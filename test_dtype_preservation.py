
import sys
import os
import torch
from mmgp import fp8_quanto_bridge

# Mock input dict with FP16 bias and FP8 weight
# Mimics a mixed-precision ComfyUI checkpoint
input_sd = {
    # An FP16 bias (should stay FP16)
    "model.layers.0.bias": torch.randn(16, dtype=torch.float16),
    # An FP8 weight (will become QTensor parts)
    "model.layers.0.weight": torch.zeros((16, 16), dtype=torch.float8_e4m3fn),
    "model.layers.0.weight_scale": torch.tensor(2.0),
    # Edge Case: uint8 metadata (must NOT be cast to float)
    "model.extra_metadata": torch.tensor([1, 2, 3], dtype=torch.uint8),
    # Edge Case: float8 non-weight (must be preserved)
    "model.some_float8": torch.tensor([1.0], dtype=torch.float8_e4m3fn),
}

# Simulate Wan2GP calling with default_dtype=bfloat16
print("Testing dtype preservation with default_dtype=bfloat16...")
res = fp8_quanto_bridge.convert_scaled_fp8_to_quanto(
    input_sd,
    dtype=torch.bfloat16, 
    fp8_format="e4m3fn"
)

out_sd = res.state_dict

# 1. Verify bias is STILL float16 (was float16)
bias = out_sd["model.layers.0.bias"]
print(f"Bias dtype: {bias.dtype}")

if bias.dtype == torch.float16:
    print("✅ SUCCESS: Bias preserved as float16")
elif bias.dtype == torch.bfloat16:
    print("❌ FAILURE: Bias cast to bfloat16 (Upcasting occurred)")
    sys.exit(1)
else:
    print(f"❌ FAILURE: Bias is {bias.dtype}, expected float16")
    sys.exit(1)

# 2. Verify quantized weight is handled (just existence check)
if "model.layers.0.weight._data" in out_sd:
    print("✅ SUCCESS: FP8 weight converted to quanto parts")
else:
    print("❌ FAILURE: FP8 weight conversion failed")
    sys.exit(1)

# 3. Verify uint8 preservation
meta = out_sd["model.extra_metadata"]
print(f"Metadata dtype: {meta.dtype}")
if meta.dtype == torch.uint8:
    print("✅ SUCCESS: uint8 metadata preserved")
else:
    print(f"❌ FAILURE: uint8 metadata cast to {meta.dtype}")
    sys.exit(1)

# 4. Verify float8 preservation
f8 = out_sd["model.some_float8"]
print(f"Float8 dtype: {f8.dtype}")
if f8.dtype == torch.float8_e4m3fn:
     print("✅ SUCCESS: float8 non-weight preserved")
else:
     print(f"❌ FAILURE: float8 non-weight cast to {f8.dtype}")
     sys.exit(1)
