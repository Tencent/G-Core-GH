"""
Hardware-specific peak TFLOPs entries for domestic accelerators.
This file is excluded from public releases.
"""

# Additional entries to merge into _DEVICE_FLOPS in flops_counter.py.
DEVICE_FLOPS_PRIV = {
    "910B": 354e12,
    "Ascend910": 354e12,
    "MLU590-M9DG": 220e12,
}
