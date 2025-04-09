import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

# 直接从 config.py 中导入 FlashSTUConfig
from flash_stu.config import FlashSTUConfig
# 假设 get_spectral_filters 用于生成 phi 参数
from flash_stu.utils.stu_utils import get_spectral_filters
# 导入我们改造后的模型（例如 FlashSTU_ViT）
from flash_stu.FlashSTU_ViT import FlashSTU_ViT


def setup_distributed():
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, local_rank, world_size


def main():
    # 直接实例化 FlashSTUConfig，所有默认参数直接从 config.py 中获取
    config = FlashSTUConfig()
    print(config)

    device, local_rank, world_size = setup_distributed()

    # 计算视觉任务的序列长度（ViT: patch 数量加 cls token）
    num_patches = (config.img_size // config.patch_size) ** 2  # 例如 14^2 = 196
    vision_seq_len = num_patches + 1  # 加上 cls token，即 197
    phi = get_spectral_filters(vision_seq_len, config.num_eigh, config.use_hankel_L, device, config.torch_dtype)
    
    # 构造模型
    model = FlashSTU_ViT(config, phi)
    # 将模型移动到目标设备，并统一转换数据类型到 config.torch_dtype（例如 bfloat16）
    model = model.to(device, dtype=config.torch_dtype)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 数据预处理（此处以 CIFAR10 为例，将图像调整到 config.img_size 大小）
    transform_train = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.RandomCrop(config.img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=config.bsz, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.bsz, shuffle=False,
                             sampler=test_sampler, num_workers=4, pin_memory=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.max_lr)

    num_epochs = config.num_epochs
    print("Start Training...")
    for epoch in range(num_epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        running_loss = 0.0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            # 将 images 转换到目标设备，并统一为 config.torch_dtype
            images = images.to(device, dtype=config.torch_dtype)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {elapsed:.2f} sec, average loss: {epoch_loss:.4f}")

        # 验证过程
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, dtype=config.torch_dtype)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Validation Accuracy after Epoch {epoch+1}: {acc:.2f}%\n")

    # 保存模型（仅在主进程保存）
    if world_size == 1 or local_rank == 0:
        torch.save(model.state_dict(), "flash_stu_vit.pth")
        print("Model saved.")


if __name__ == "__main__":
    main()