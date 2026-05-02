import argparse
import random
from pathlib import Path

import numpy as np
import scipy.io
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from model import HMSTUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_part_dir(data_root: Path, part: str) -> Path:
    part = part.upper()
    candidates = {
        "A": ["part_A_final", "part_A"],
        "B": ["part_B_final", "part_B"],
    }
    for name in candidates[part]:
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find ShanghaiTech part {part} under {data_root}. "
        f"Expected one of: {candidates[part]}"
    )


def load_gt_points(gt_mat_path: Path) -> np.ndarray:
    data = scipy.io.loadmat(str(gt_mat_path))
    if "image_info" in data:
        points = data["image_info"][0, 0][0, 0][0]
        return np.asarray(points, dtype=np.float32)
    if "annPoints" in data:
        return np.asarray(data["annPoints"], dtype=np.float32)
    if "points" in data:
        return np.asarray(data["points"], dtype=np.float32)
    raise KeyError(f"Unsupported ground-truth MAT format: {gt_mat_path}")


def density_map_path(dm_dir: Path, image_path: Path) -> Path:
    return dm_dir / f"{image_path.stem}_dm.npy"


def gt_mat_path(gt_dir: Path, image_path: Path) -> Path:
    candidates = [
        gt_dir / f"GT_{image_path.stem}.mat",
        gt_dir / f"{image_path.stem}.mat",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Ground-truth .mat not found for {image_path.name} in {gt_dir}")


def generate_density_map(height: int, width: int, points: np.ndarray, sigma: float) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    if points.size == 0:
        return density
    for x, y in points:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        xi = max(0, min(width - 1, xi))
        yi = max(0, min(height - 1, yi))
        density[yi, xi] += 1.0
    return scipy.ndimage.gaussian_filter(density, sigma=sigma, mode="constant")


def ensure_density_maps(img_dir: Path, gt_dir: Path, dm_dir: Path, sigma: float, force: bool = False) -> None:
    dm_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    generated = 0
    skipped = 0
    for idx, img_path in enumerate(img_paths, start=1):
        out_path = density_map_path(dm_dir, img_path)
        if out_path.is_file() and not force:
            skipped += 1
            continue
        with Image.open(img_path) as img:
            width, height = img.size
        points = load_gt_points(gt_mat_path(gt_dir, img_path))
        dm = generate_density_map(height, width, points, sigma=sigma)
        np.save(out_path, dm)
        generated += 1
        if idx % 50 == 0:
            print(f"[density] {img_dir.name}: processed {idx}/{len(img_paths)}")
    print(
        f"[density] {img_dir.name}: generated={generated}, skipped={skipped}, total={len(img_paths)}"
    )


class CrowdDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        img_dir: Path,
        dm_dir: Path,
        crop_size: int,
        is_train: bool,
        downsample: int = 4,
    ):
        self.img_paths = sorted(img_dir.glob("*.jpg"))
        self.dm_dir = dm_dir
        self.crop_size = crop_size
        self.is_train = is_train
        self.downsample = downsample
        self.normalize = T.Normalize(mean=self.MEAN, std=self.STD)
        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        dm_path = density_map_path(self.dm_dir, img_path)
        if not dm_path.is_file():
            raise FileNotFoundError(f"Density map missing: {dm_path}")

        img = Image.open(img_path).convert("RGB")
        dm = np.load(dm_path).astype(np.float32)
        width, height = img.size

        if height < self.crop_size or width < self.crop_size:
            new_h = max(height, self.crop_size)
            new_w = max(width, self.crop_size)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            dm_t = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0)
            dm = (
                F.interpolate(dm_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
                .squeeze()
                .numpy()
            )
            height, width = new_h, new_w

        if self.is_train:
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
            img = TF.crop(img, top, left, self.crop_size, self.crop_size)
            dm = dm[top : top + self.crop_size, left : left + self.crop_size]
            if random.random() > 0.5:
                img = TF.hflip(img)
                dm = dm[:, ::-1].copy()
            img = self.jitter(img)
        else:
            new_h = max(32, (height // 32) * 32)
            new_w = max(32, (width // 32) * 32)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            dm_t = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0)
            dm = (
                F.interpolate(dm_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
                .squeeze()
                .numpy()
            )

        img_t = self.normalize(TF.to_tensor(img))
        dm_t = torch.from_numpy(dm.copy()).unsqueeze(0)

        out_h = dm_t.shape[1] // self.downsample
        out_w = dm_t.shape[2] // self.downsample
        dm_t = (
            F.interpolate(
                dm_t.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False
            ).squeeze(0)
            * (self.downsample**2)
        )
        return img_t, dm_t


class CrowdLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred.sum(dim=(1, 2, 3)), target.sum(dim=(1, 2, 3)))
        return self.alpha * mse + self.beta * mae


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    maes = []
    mses = []
    for imgs, tgts in loader:
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)
        preds = model(imgs)
        pred_count = preds.sum(dim=(1, 2, 3))
        true_count = tgts.sum(dim=(1, 2, 3))
        diff = pred_count - true_count
        maes.extend(diff.abs().cpu().tolist())
        mses.extend((diff**2).cpu().tolist())
    mae = float(np.mean(maes))
    rmse = float(np.sqrt(np.mean(mses)))
    return mae, rmse


def build_dataloaders(args, data_root: Path):
    part_dir = resolve_part_dir(data_root, args.part)
    train_img_dir = part_dir / "train_data" / "images"
    train_gt_dir = part_dir / "train_data" / "ground_truth"
    train_dm_dir = part_dir / "train_data" / "density_maps"
    test_img_dir = part_dir / "test_data" / "images"
    test_gt_dir = part_dir / "test_data" / "ground_truth"
    test_dm_dir = part_dir / "test_data" / "density_maps"

    print(f"[dataset] lane part: {args.part.upper()}")
    print(f"[dataset] using: {part_dir}")

    if not args.skip_density_gen:
        ensure_density_maps(train_img_dir, train_gt_dir, train_dm_dir, sigma=args.sigma, force=args.force_density)
        ensure_density_maps(test_img_dir, test_gt_dir, test_dm_dir, sigma=args.sigma, force=args.force_density)

    if args.generate_density_only:
        return None, None

    train_dataset = CrowdDataset(
        img_dir=train_img_dir,
        dm_dir=train_dm_dir,
        crop_size=args.train_crop,
        is_train=True,
        downsample=args.downsample,
    )
    test_dataset = CrowdDataset(
        img_dir=test_img_dir,
        dm_dir=test_dm_dir,
        crop_size=args.val_crop,
        is_train=False,
        downsample=args.downsample,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(
        f"[data] train images={len(train_dataset)}, train batches={len(train_loader)}, "
        f"test images={len(test_dataset)}"
    )
    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Train HMSTUNet on ShanghaiTech crowd counting.")
    parser.add_argument("--data-root", type=str, default="data", help="ShanghaiTech dataset root")
    parser.add_argument("--part", type=str, choices=["A", "B", "a", "b"], default="A")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-crop", type=int, default=384)
    parser.add_argument("--val-crop", type=int, default=512)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=15.0, help="Gaussian sigma for GT density maps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--best-name", type=str, default="best.pth")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--pretrained-encoder", action="store_true", help="Use timm pretrained ConvNeXt encoder")
    parser.add_argument("--skip-density-gen", action="store_true", help="Skip density map generation step")
    parser.add_argument("--force-density", action="store_true", help="Regenerate density maps even if present")
    parser.add_argument("--generate-density-only", action="store_true", help="Only prepare density maps and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    args.part = args.part.upper()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")

    data_root = Path(args.data_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / args.best_name

    train_loader, test_loader = build_dataloaders(args, data_root)
    if args.generate_density_only:
        print("[done] density maps generated.")
        return

    model = HMSTUNet(pretrained=args.pretrained_encoder).to(device)
    criterion = CrowdLoss(alpha=1.0, beta=0.1)

    encoder_params = list(model.enc.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    new_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": args.lr * args.encoder_lr_mult},
            {"params": new_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_mae = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "mae" in ckpt:
            best_mae = float(ckpt["mae"])
        print(f"[resume] loaded {resume_path} at epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, tgts in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss = criterion(preds, tgts)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_mae, val_rmse = evaluate(model, test_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(
            f"[{epoch:03d}/{args.epochs:03d}] loss={avg_loss:.4f} "
            f"val_MAE={val_mae:.2f} val_RMSE={val_rmse:.2f}"
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "mae": val_mae,
            "mse": val_rmse,
            "part": args.part,
            "args": vars(args),
        }
        torch.save(checkpoint, checkpoint_dir / "last.pth")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(checkpoint, best_path)
            print(f"[best] saved {best_path} (MAE={best_mae:.2f})")

    print(f"[done] training complete. best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
