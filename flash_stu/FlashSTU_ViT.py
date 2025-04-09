import torch
import torch.nn as nn
from transformers import PreTrainedModel

from flash_stu.modules.stu import STU
from flash_stu.modules.attention import Attention
from flash_stu.utils.numerics import nearest_power_of_two
from flash_stu.config import FlashSTUConfig
from flash_stu.layers.stu_layer import STULayer
from flash_stu.layers.attention_layer import AttentionLayer

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm as TritonNorm
    triton_norm = True
except ImportError as e:
    print(
        f"Unable to import Triton-based RMSNorm: {e}. Falling back to PyTorch implementation."
    )
    from torch.nn import RMSNorm

    triton_norm = False


class FlashSTU_ViT(PreTrainedModel):
    config_class = FlashSTUConfig

    def __init__(self, config, phi) -> None:
        """
        参数说明（除原有参数外，新增加图像相关参数）：
          - config.img_size: 输入图像尺寸（假设为正方形，如224）
          - config.patch_size: patch 尺寸（如16）
          - config.in_chans: 输入图像通道数，通常为3
          - config.num_classes: 分类任务类别数
        """
        super(FlashSTU_ViT, self).__init__(config)

        self.n_layers = config.n_layers
        # 此处的 n 可用于 STU 内部计算（依赖于序列长度），
        # 这里令序列长度为 (num_patches + 1)
        num_patches = (config.img_size // config.patch_size) ** 2
        seq_len = num_patches + 1  # +1为cls token
        self.n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)
        self.phi = phi
        self.use_approx = config.use_approx
        self.use_hankel_L = config.use_hankel_L

        # 使用卷积层实现 patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=config.in_chans,
            out_channels=config.n_embd,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
            # 注意：这里我们不设置 dtype（后续统一转到指定的 torch_dtype）
        )
        self.num_patches = num_patches

        # 分类 token 和位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.n_embd))
        self.pos_drop = nn.Dropout(config.dropout)

        # 构造 Transformer 层序列，交替使用 STULayer 和 AttentionLayer（如果启用 attention）
        self.layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            if layer_idx % 2 == 0:
                self.layers.append(STULayer(config, self.phi, self.n))
            else:
                self.layers.append(
                    AttentionLayer(config)
                    if config.use_attn
                    else STULayer(config, self.phi, self.n)
                )

        # 使用归一化层
        self.norm = (
            TritonNorm(config.n_embd)
            if triton_norm
            else RMSNorm(config.n_embd, dtype=config.torch_dtype)
        )
        self.norm = self.norm.to(dtype=config.torch_dtype)

        # 分类头
        self.head = nn.Linear(
            config.n_embd, config.num_classes, bias=config.bias, dtype=config.torch_dtype
        )

        # 用于参数初始化的标准差
        self.std = (config.n_embd) ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 的 shape 为 [B, in_chans, img_size, img_size]
        B = x.size(0)
        # patch embedding：输出 [B, n_embd, H', W']
        x = self.patch_embed(x)
        # 展平后得到 [B, num_patches, n_embd]
        x = x.flatten(2).transpose(1, 2)
        # 将分类 token 添加到序列最前面
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 加入位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer 层：交替使用 STU 与 Attention 模块（或全部为 STU）
        for layer in self.layers:
            x = layer(x)

        # 归一化后提取分类 token（第一个 token）
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        # 此处可根据需要排除某些参数（如位置嵌入）避免重复计数
        if hasattr(self, "pos_embed") and self.pos_embed is not None:
            n_params -= self.pos_embed.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 如果模块有 SCALE_INIT 属性，调整标准差
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            if self.use_approx:
                torch.nn.init.xavier_normal_(module.M_inputs)
                torch.nn.init.xavier_normal_(module.M_filters)
            else:
                torch.nn.init.xavier_normal_(module.M_phi_plus)
                if not self.use_hankel_L:
                    torch.nn.init.xavier_normal_(module.M_phi_minus)
        elif isinstance(module, Attention):
            torch.nn.init.xavier_normal_(module.c_attn.weight)
            torch.nn.init.xavier_normal_(module.c_proj.weight)
            if module.c_attn.bias is not None:
                torch.nn.init.zeros_(module.c_attn.bias)
            if module.c_proj.bias is not None:
                torch.nn.init.zeros_(module.c_proj.bias)