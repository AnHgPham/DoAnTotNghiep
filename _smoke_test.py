"""Smoke test: import, DSCNN forward, checkpoint loading."""
import sys
from pathlib import Path

ROOT = Path(r"D:\Downloads\DoAnTotNghiep")
sys.path.insert(0, str(ROOT))

import torch
from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN
from src.enhancements.denoiser import Denoiser

print("Import OK")

m = DSCNN(model_size="L", input_shape=(47, 10))
x = torch.randn(1, 1, 47, 10)
y = m(x)
print(f"DSCNN-L output: {y.shape}")

CKPT = ROOT / "checkpoints" / "best.pt"
ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
m.load_state_dict(ckpt["model_state_dict"])
print(f"Checkpoint loaded: epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?'):.6f}")

print("Smoke test PASSED")