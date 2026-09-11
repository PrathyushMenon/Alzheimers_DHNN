from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from model import MySwinUNETR  # noqa: E402
from utils.data_utils import MultiModalDataset  # noqa: E402


def load_model(checkpoint: Path, device: torch.device) -> MySwinUNETR:
    model = MySwinUNETR(
        img_size=(32, 32, 32),
        in_channels=3,
        out_channels=3,
        feature_size=24,
        modality="fod",
    ).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def extract_split(model: MySwinUNETR, csv_path: Path, split: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    ds = MultiModalDataset(str(csv_path), dataset=split, modality="fod")
    feats: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for idx in range(len(ds)):
            fod_o0, fod_o2, fod_o4, label = ds[idx]
            tensors = [
                torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                for x in (fod_o0, fod_o2, fod_o4)
            ]
            split_feats = model.get_features(*tensors)
            # The first Linear output is the 64-D penultimate feature used by
            # the official Swin-FOD head before the 3-class classifier.
            feat = split_feats[0].detach().cpu().numpy()
            feats.append(feat)
            labels.append(int(label))
    return np.concatenate(feats, axis=0), np.asarray(labels, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract local Swin-FOD features from the official model.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("data/official_inputs/swin_fod_local.csv"), type=Path)
    parser.add_argument("--output-dir", default=Path("features/fod"), type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)

    for split in ("train", "val", "test"):
        x, y = extract_split(model, args.manifest, split, device)
        np.save(args.output_dir / f"{split}_features.npy", x)
        np.save(args.output_dir / f"{split}_labels.npy", y)
        print(f"{split}: features={x.shape}, labels={y.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
