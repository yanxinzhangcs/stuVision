import torch
from transformers import PretrainedConfig

class FlashSTUConfig(PretrainedConfig):
    model_type = "FlashSTU"

    def __init__(
        self,
        bsz: int = 1,
        n_embd: int = 1536,
        n_heads: int = 8,
        n_layers: int = 26,
        seq_len: int = 8192,
        window_size: int = 1024,
        vocab_size: int = 200064,
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_flash_fft: bool = True,
        use_approx: bool = True,
        use_attn: bool = True,
        softcap: float = 50.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        # 新增视觉任务相关参数:
        img_size: int = 224,        # 输入图像尺寸（正方形）
        patch_size: int = 16,       # 每个 patch 的尺寸
        in_chans: int = 3,          # 图像通道数
        num_classes: int = 1000,    # 分类任务的类别数
        multimodal: bool = False,   # 是否为视觉语言多模态任务
        # 新增训练相关参数:
        max_lr: float = 0.001,      # 学习率
        max_norm: float = 1.0,      # 梯度裁剪阈值
        num_epochs: int = 10,       # 训练轮数
        **kwargs,
    ):
        super().__init__(**kwargs)
        # NLP 及模型基础设置
        self.bsz = bsz
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.hidden_size = n_embd
        self.intermediate_size = n_embd * mlp_scale
        self.hidden_act = "swish"
        self.bias = bias
        self.dropout = dropout
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.use_flash_fft = use_flash_fft
        self.use_approx = use_approx
        self.use_attn = use_attn
        self.softcap = softcap
        self.torch_dtype = torch_dtype

        # 图像相关配置
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.multimodal = multimodal

        # 训练相关配置
        self.max_lr = max_lr
        self.max_norm = max_norm
        self.num_epochs = num_epochs

    def __repr__(self):
        return (
            f"FlashSTUConfig(n_layers={self.n_layers}, n_embd={self.n_embd}, seq_len={self.seq_len}, "
            f"vocab_size={self.vocab_size}, img_size={self.img_size}, patch_size={self.patch_size}, "
            f"in_chans={self.in_chans}, num_classes={self.num_classes}, multimodal={self.multimodal}, "
            f"max_lr={self.max_lr}, num_epochs={self.num_epochs})"
        )