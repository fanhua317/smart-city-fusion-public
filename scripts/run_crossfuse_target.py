from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def to_gray_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32)
    return torch.from_numpy(arr[None, None, :, :]).to(device)


def save_gray_like(tensor: torch.Tensor, out_path: Path) -> None:
    arr = tensor.detach().float().cpu().numpy()[0, 0]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        Image.fromarray(arr, mode="L").save(out_path, quality=100, subsampling=0, optimize=False)
    else:
        Image.fromarray(arr, mode="L").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--vi-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from network.net_autoencoder import Auto_Encoder_single
    from network.net_conv_trans import Trans_FuseNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CrossFuse target runner requires CUDA for this workflow")

    custom_config_auto = {
        "in_channels": 1,
        "out_channels": 1,
        "en_out_channels1": 32,
        "en_out_channels": 64,
        "num_layers": 3,
        "dense_out": 128,
        "part_out": 128,
        "train_flag": False,
    }
    custom_config_trans = {
        "en_out_channels1": 32,
        "out_channels": 1,
        "part_out": 128,
        "train_flag": False,
        "img_size": 32,
        "patch_size": 2,
        "depth_self": 1,
        "depth_cross": 1,
        "n_heads": 16,
        "qkv_bias": True,
        "mlp_ratio": 4,
        "p": 0.0,
        "attn_p": 0.0,
    }

    auto_ir = Auto_Encoder_single(**custom_config_auto).to(device).eval()
    auto_vi = Auto_Encoder_single(**custom_config_auto).to(device).eval()
    trans = Trans_FuseNet(**custom_config_trans).to(device).eval()
    auto_ir.load_state_dict(torch.load(repo / "models/autoencoder/auto_encoder_epoch_4_ir.model", map_location=device))
    auto_vi.load_state_dict(torch.load(repo / "models/autoencoder/auto_encoder_epoch_4_vi.model", map_location=device))
    trans.load_state_dict(
        torch.load(repo / "models/transfuse/fusetrans_epoch_32_bs_8_num_20k_lr_0.1_s1_c1.model", map_location=device)
    )

    ir_dir = Path(args.ir_dir)
    vi_dir = Path(args.vi_dir)
    out_dir = Path(args.out_dir)
    names = sorted(p.name for p in ir_dir.iterdir() if p.is_file())
    with torch.no_grad():
        for idx, name in enumerate(names, start=1):
            vi_path = vi_dir / name
            if not vi_path.exists():
                raise FileNotFoundError(vi_path)
            ir = to_gray_tensor(ir_dir / name, device)
            vi = to_gray_tensor(vi_path, device)
            ir_sh, ir_de = auto_ir(ir)
            vi_sh, vi_de = auto_vi(vi)
            out = trans(ir_de, ir_sh, vi_de, vi_sh, True)["out"]
            save_gray_like(out, out_dir / name)
            print(f"{idx}/{len(names)} {name}")


if __name__ == "__main__":
    main()
