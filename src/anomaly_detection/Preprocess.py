import numpy as np, joblib
from pathlib import Path
import pandas as pd

ROOT   = Path(__file__).parent
IDX    = np.load(ROOT / "idx.npy")               
scaler = joblib.load(ROOT / "scaler.pkl")  
ALL_COLS = np.load(ROOT / "all_cols.npy")

def transform(raw_list: list[float]) -> np.ndarray:
   
    if len(raw_list) != len(ALL_COLS):
        raise ValueError(f"Input length {len(raw_list)} does not match expected {len(ALL_COLS)}")

    
    cleaned = [
        0.0 if (x in ("", None) or (isinstance(x, float) and np.isnan(x))) else float(x)
        for x in raw_list
    ] 

    x_full = pd.DataFrame([cleaned], columns=ALL_COLS, dtype=np.float32)
    x_sel  = x_full.iloc[:, IDX]  
    x_standard = scaler.transform(x_sel)                           
    return x_standard  

