# simple_cnn_embedder.py
# -*- coding: utf-8 -*-
"""
一个简单的 CNN 嵌入模型：将 RGB 图片转换为定长向量（embedding）。
- 支持 GPU/CPU 自动选择
- 支持单图 / 批量（文件夹）提取
- 输出向量可选 L2 归一化，方便相似度计算（cosine）
"""

from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---- 可按需修改的默认参数 ----
DEFAULT_IMAGE_SIZE = 224            # 输入尺寸 (H=W)
DEFAULT_EMBED_DIM = 256            # 向量维度
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 使用 ImageNet 常用的归一化参数（适配 RGB 0-1）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------- CNN 定义 ----------------
class SimpleCNNEmbedder(nn.Module):
    """
    一个轻量级 CNN：Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> 池化(自适应 GAP) -> 全连接 -> 向量
    - 使用 AdaptiveAvgPool2d 保证任意输入尺寸都能变成固定维度
    - 输出向量可进一步做 L2 归一化，便于计算余弦相似度
    """
    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        # 3x224x224 -> 通道增长
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 3->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),# 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # 自适应 GAP 到 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 投影到 embed_dim
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)         # [B, 128, H', W']
        x = self.gap(x)              # [B, 128, 1, 1]
        x = torch.flatten(x, 1)      # [B, 128]
        x = self.fc(x)               # [B, embed_dim]
        return x


# ---------------- 预处理 ----------------
def build_transform(img_size: int = DEFAULT_IMAGE_SIZE):
    # 仅用 PyTorch/Tensor API，避免依赖 torchvision.transforms
    def _transform(pil_img: Image.Image) -> torch.Tensor:
        # 转 RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        # resize 到正方形（双三次插值）
        pil_img = pil_img.resize((img_size, img_size), resample=Image.BICUBIC)
        # HWC[0..255] -> CHW[0..1]
        x = torch.from_numpy(np.array(pil_img)).float() / 255.0  # type: ignore
        x = x.permute(2, 0, 1)  # HWC->CHW
        # 标准化
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
        return x
    return _transform

# 为了不引入额外依赖，这里用 numpy；若你环境里本就有 torchvision 可换成 transforms
import numpy as np  # noqa


# ---------------- 抽取函数 ----------------
@torch.no_grad()
def image_to_vec(
    image_path: str,
    model: Optional[SimpleCNNEmbedder] = None,
    embed_dim: int = DEFAULT_EMBED_DIM,
    img_size: int = DEFAULT_IMAGE_SIZE,
    device: str = DEFAULT_DEVICE,
    l2_normalize: bool = True,
) -> np.ndarray:
    """
    单张图片 -> 向量
    """
    device = torch.device(device)
    if model is None:
        model = SimpleCNNEmbedder(embed_dim=embed_dim)
    model = model.to(device).eval()

    tfm = build_transform(img_size)
    img = Image.open(image_path)
    x = tfm(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    vec = model(x)                        # [1, D]
    if l2_normalize:
        vec = F.normalize(vec, p=2, dim=1)
    return vec.squeeze(0).cpu().numpy()   # [D]


@torch.no_grad()
def folder_to_vecs(
    folder: str,
    patterns: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    embed_dim: int = DEFAULT_EMBED_DIM,
    img_size: int = DEFAULT_IMAGE_SIZE,
    device: str = DEFAULT_DEVICE,
    l2_normalize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    文件夹内所有图片 -> 向量矩阵
    返回:
        embs: [N, D] 的 numpy 数组
        paths: 对应的图片路径列表
    """
    device = torch.device(device)
    model = SimpleCNNEmbedder(embed_dim=embed_dim).to(device).eval()
    tfm = build_transform(img_size)

    folder = str(folder)
    paths = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in patterns:
            paths.append(str(p))

    if not paths:
        return np.empty((0, embed_dim), dtype=np.float32), []

    batch_tensors: List[torch.Tensor] = []
    embs_list: List[torch.Tensor] = []
    bs = 32  # 批大小，可按显存调整

    for path in paths:
        try:
            img = Image.open(path)
            x = tfm(img)  # [3, H, W]
            batch_tensors.append(x)
        except Exception as e:
            print(f"[WARN] 跳过无法读取的图片: {path} ({e})")

        if len(batch_tensors) == bs:
            batch = torch.stack(batch_tensors, dim=0).to(device)  # [B, 3, H, W]
            out = model(batch)
            if l2_normalize:
                out = F.normalize(out, p=2, dim=1)
            embs_list.append(out.cpu())
            batch_tensors.clear()

    # 处理残留
    if batch_tensors:
        batch = torch.stack(batch_tensors, dim=0).to(device)
        out = model(batch)
        if l2_normalize:
            out = F.normalize(out, p=2, dim=1)
        embs_list.append(out.cpu())

    embs = torch.cat(embs_list, dim=0).numpy().astype(np.float32)  # [N, D]
    return embs, paths


# ---------------- 小测试 / 示例 ----------------
if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser(description="Simple CNN Embedder")
    parser.add_argument("--image", type=str, default="", help="单张图片路径")
    parser.add_argument("--folder", type=str, default="", help="批量提取的文件夹")
    parser.add_argument("--dim", type=int, default=DEFAULT_EMBED_DIM, help="向量维度")
    parser.add_argument("--size", type=int, default=DEFAULT_IMAGE_SIZE, help="输入尺寸")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="'cuda' 或 'cpu'")
    parser.add_argument("--no_l2", action="store_true", help="不做 L2 归一化")
    args = parser.parse_args()

    if args.image:
        t0 = time.time()
        v = image_to_vec(
            args.image,
            embed_dim=args.dim,
            img_size=args.size,
            device=args.device,
            l2_normalize=not args.no_l2,
        )
        print(f"向量形状: {v.shape}, 耗时: {time.time()-t0:.3f}s")
        print(v[:10], "...")  # 打印前10维

    if args.folder:
        t0 = time.time()
        embs, paths = folder_to_vecs(
            args.folder,
            embed_dim=args.dim,
            img_size=args.size,
            device=args.device,
            l2_normalize=not args.no_l2,
        )
        print(f"共提取 {len(paths)} 张图，向量形状: {embs.shape}, 耗时: {time.time()-t0:.3f}s")
