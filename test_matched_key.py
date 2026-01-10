
import sys
import torch
from mmgp import fp8_quanto_bridge

# Mock state dict representing a ComfyUI FP8 model
input_sd = {
    "model.layers.0.weight": torch.zeros((16, 16), dtype=torch.float8_e4m3fn),
    "model.layers.0.weight_scale": torch.tensor(2.0)
}

print("Testing detect() return values...")
res = fp8_quanto_bridge.detect(input_sd)
print(f"Result: {res}")

if res.get("matched") is True:
    print("✅ SUCCESS: 'matched': True found")
else:
    print("❌ FAILURE: 'matched' key missing or False")
    sys.exit(1)
