# utils/torch_helper.py

import torch

__all__ = ["eye_like", "zeros_like_shape"]

def eye_like(x: torch.Tensor, size: int = None) -> torch.Tensor:
    device = x.device
    dtype  = x.dtype
    n = size if size is not None else x.shape[-1]
    return torch.eye(n, device=device, dtype=dtype)

def zeros_like_shape(shape, ref_tensor=None, *,
                     device: torch.device = None,
                     dtype: torch.dtype = None) -> torch.Tensor:
    """
    返回一个指定形状的零张量。
    - 如果 ref_tensor 是 Tensor，则优先从它继承 device 和 dtype；
    - 否则可通过关键字 device/dtype 指定；
    - shape: int 或 tuple(int)，输出张量形状。
    """
    # 如果传进来的第二个位置参数是 Tensor，就把它当 ref_tensor
    if isinstance(ref_tensor, torch.Tensor):
        device = ref_tensor.device
        dtype  = ref_tensor.dtype

    # 如果还是没指定 device/dtype，就用默认
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32

    return torch.zeros(shape, device=device, dtype=dtype)
