import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from nba_api.stats.endpoints import leaguegamefinder

print("=== HoopIQ Environment Check ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"xgboost: {xgb.__version__}")
print("nba_api: installed")
print("=== All systems go ===")