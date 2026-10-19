"""
Sinh anh suong mu dua tren ban do do sau (Depth Anything V2) + mo hinh tan xa khi quyen.

Mo hinh tan xa khi quyen (Koschmieder):
    I(x) = J(x) * t(x) + A * (1 - t(x))
    t(x) = exp(-alpha * d(x))
trong do:
    J     : anh goc (sach)
    d(x)  : do sau chuan hoa ve [0, 1] (0 = gan, 1 = xa nhat)
    alpha : he so tan xa (cang lon suong cang day)
    A     : anh sang khi quyen (mau/do sang cua suong)

Vi du:
    python fog_depth.py --img-path images/ --outdir fog_out --alpha 1.5
    python fog_depth.py --img-path a.jpg --alpha 0.5 1.0 2.0 3.0 --A 0.85 --save-depth
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
HF_REPOS = {
    'vits': 'depth-anything/Depth-Anything-V2-Small',
    'vitb': 'depth-anything/Depth-Anything-V2-Base',
    'vitl': 'depth-anything/Depth-Anything-V2-Large',
}
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_depth_model(encoder, device):
    """Tai Depth Anything V2; tu dong download checkpoint tu HuggingFace neu chua co."""
    ckpt = ROOT / 'Depth-Anything-V2' / 'checkpoints' / f'depth_anything_v2_{encoder}.pth'
    if not ckpt.exists():
        from huggingface_hub import hf_hub_download
        print(f'Downloading checkpoint {ckpt.name} ...')
        hf_hub_download(repo_id=HF_REPOS[encoder], filename=ckpt.name, local_dir=ckpt.parent)
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model.to(device).eval()


@torch.no_grad()
def estimate_depth(model, img_bgr, input_size=518):
    """
    Tra ve do sau chuan hoa [0, 1] cung kich thuoc anh (0 = gan, 1 = xa).
    Depth Anything V2 xuat ra disparity tuong doi (gia tri lon = gan), nen can dao nguoc.
    """
    disp = model.infer_image(img_bgr, input_size).astype(np.float32)
    lo, hi = np.percentile(disp, 0.5), np.percentile(disp, 99.5)  # bo outlier
    disp = np.clip((disp - lo) / (hi - lo + 1e-8), 0, 1)
    depth = 1.0 - disp
    # Lam min nhe de tranh vien rang cua khi upsample len 2048x2048
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=max(img_bgr.shape[:2]) / 1024)
    return depth


def estimate_atmospheric_light(img, depth, top=0.001):
    """Uoc luong A: trung binh cac pixel sang nhat trong vung xa nhat (top 0.1% do sau)."""
    flat_d = depth.reshape(-1)
    n = max(int(flat_d.size * top), 1)
    idx = np.argpartition(flat_d, -n)[-n:]
    A = img.reshape(-1, 3)[idx].mean(axis=0)
    return np.clip(A, 0.7, 1.0)  # dam bao suong co mau sang


def add_fog(img_bgr, depth, alpha=1.0, A=0.9, depth_gamma=1.0):
    """
    Ap dung mo hinh tan xa khi quyen.
        img_bgr     : anh uint8 BGR
        depth       : do sau [0, 1] (0 = gan, 1 = xa)
        alpha       : he so tan xa beta (0.5 nhe, 1-2 vua, >3 day dac)
        A           : anh sang khi quyen, float hoac mang 3 kenh BGR trong [0, 1]
        depth_gamma : <1 lam suong lan ra gan camera hon, >1 tap trung o xa
    """
    J = img_bgr.astype(np.float32) / 255.0
    d = np.power(depth, depth_gamma)
    t = np.exp(-alpha * d)[..., None]
    A = np.asarray(A, dtype=np.float32).reshape(1, 1, -1)
    I = J * t + A * (1.0 - t)
    return (np.clip(I, 0, 1) * 255).round().astype(np.uint8), t[..., 0]


def main():
    ap = argparse.ArgumentParser(description='Sinh suong mu bang Depth Anything V2 + tan xa khi quyen')
    ap.add_argument('--img-path', required=True, help='anh hoac thu muc anh')
    ap.add_argument('--outdir', default='fog_out')
    ap.add_argument('--alpha', type=float, nargs='+', default=[1.0], help='he so tan xa (co the nhieu gia tri)')
    ap.add_argument('--A', default='0.9', help="anh sang khi quyen: so trong [0,1], 3 so 'B,G,R', hoac 'auto'")
    ap.add_argument('--depth-gamma', type=float, default=1.0)
    ap.add_argument('--size', type=int, default=2048, help='resize anh ve size x size (0 = giu nguyen)')
    ap.add_argument('--input-size', type=int, default=1036,
                    help='do phan giai dua vao Depth Anything (boi so cua 14; 518 nhanh, 1036 chi tiet hon)')
    ap.add_argument('--encoder', default='vitl', choices=list(MODEL_CONFIGS))
    ap.add_argument('--save-depth', action='store_true', help='luu them ban do do sau va transmission')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_depth_model(args.encoder, device)

    src = Path(args.img_path)
    files = sorted(p for p in src.rglob('*') if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for k, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            print(f'Skip (cannot read): {f}')
            continue
        if args.size and img.shape[:2] != (args.size, args.size):
            img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_CUBIC)

        depth = estimate_depth(model, img, args.input_size)
        if args.A == 'auto':
            A = estimate_atmospheric_light(img.astype(np.float32) / 255.0, depth)
        else:
            A = [float(v) for v in args.A.split(',')]

        for alpha in args.alpha:
            fog, t = add_fog(img, depth, alpha, A, args.depth_gamma)
            cv2.imwrite(str(out / f'{f.stem}_fog_a{alpha:g}.png'), fog)
            if args.save_depth:
                cv2.imwrite(str(out / f'{f.stem}_trans_a{alpha:g}.png'), (t * 255).astype(np.uint8))
        if args.save_depth:
            cv2.imwrite(str(out / f'{f.stem}_depth.png'), (depth * 255).astype(np.uint8))
        print(f'[{k}/{len(files)}] {f.name} -> alpha={args.alpha}, A={np.round(A, 3).tolist()}')


if __name__ == '__main__':
    main()
