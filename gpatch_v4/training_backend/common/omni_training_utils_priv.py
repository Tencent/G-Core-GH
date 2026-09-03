"""
Hardware-specific peak TFLOPs lookup table (BF16).
This file is excluded from public releases.
"""

# Maps substring of device name (uppercased) to BF16 peak TFLOPs.
PEAK_TFLOPS_TABLE = {
    "MI300X": 1336.0,
    "H100": 989.0,
    "H800": 989.0,
    "H200": 989.0,
    "A100": 312.0,
    "A800": 312.0,
    "L40": 181.05,
    "L20": 119.5,
    "H20": 148.0,
    "910B": 354.0,
    "RTX 3070 TI": 21.75,
    "MLU590-M9DG": 314.0,
}
