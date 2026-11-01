#Real Time Testing
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
shap.initjs()
from catboost import CatBoostClassifier
from src.anomaly_detection.Preprocess import transform
from pathlib import Path
from datetime import datetime
from typing import Optional
import asyncio
import logging
import requests
import threading


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
PLOT_DIR = ROOT / "plots"; PLOT_DIR.mkdir(exist_ok=True)
TEST_DATA_PATH = ROOT / "test_data.csv"

test_data = pd.read_csv(TEST_DATA_PATH)
data_iterator = test_data.iterrows()  # 迭代器用於循序讀取

# ----- 1) Model & Explainer -----
model = CatBoostClassifier()
model.load_model(ROOT / "model.cbm")
threshold = float(np.load(ROOT / "threshold.npy"))
background = np.load(ROOT / "bg_1000.npy")
explainer = shap.TreeExplainer(model, background)
IDX = np.load(ROOT / "idx.npy")

# ----- 2) FastAPI -----
app = FastAPI(title="Festo Anomaly API")

class LineIn(BaseModel):
    line: str  

class ResultOut(BaseModel):
    score: float
    is_anomaly: bool
    plot_file: Optional[str] = None

def generate_waterfall_plot(shap_val, score, plot_fp):
    plt.figure()
    shap.plots.waterfall(shap_val, show=False)
    plt.title(f"score={score:.3f}")
    plt.savefig(plot_fp, bbox_inches="tight")
    plt.close()

def run_inference(x: np.ndarray):
    try:
        score = float(model.predict_proba(x)[0, 1])
        is_abn = score >= threshold
        plot_nm = None
        if is_abn:
            x_df = pd.DataFrame(x, columns=IDX.astype(str))
            shap_val = explainer(x_df)
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            plot_nm = f"{ts}.png"
            plot_fp = PLOT_DIR / plot_nm
            generate_waterfall_plot(shap_val[0], score, plot_fp)
        return score, is_abn, plot_nm
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 2-A  HTTP 端點 ---
@app.post("/predict", response_model=ResultOut)
def predict(item: LineIn):
    try:
        raw_list = item.line.split(",")
        x_std = transform(raw_list)
        score, is_abn, plot_nm = run_inference(x_std)
        return {"score": score, "is_anomaly": is_abn, "plot_file": plot_nm}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/plot/{file}")
def get_plot(file: str):       
    fp = PLOT_DIR / file
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(fp, media_type="image/png")

# --- 2-B  自動數據傳輸 ---

async def send_test_data():
    url = "http://127.0.0.1:8000/predict"
    global data_iterator
    for _ in range(len(test_data)):  
        try:
            index, row = next(data_iterator)  
            raw_data_array = row.drop(['Pass/Fail', 'Time']).values  
            clean = ["" if pd.isna(v) else v for v in raw_data_array]
            logger.info(f"Raw data shape before transform: {raw_data_array.shape}")
            raw_data = ",".join(map(str, clean))  
            data = {"line": raw_data}
            response = requests.post(url, json=data, timeout=5)
            logger.info(f"Sent (Index {index}): {raw_data}, Response: {response.json()}")
        except StopIteration:
            logger.warning("Unexpected StopIteration, data may be exhausted.")
            break  
        except Exception as e:
            logger.error(f"Send error: {e}")
            break  
        await asyncio.sleep(3)  
    logger.info("All 10 test data sent, transmission completed.")

def start_data_transmission():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_test_data())


if __name__ == "__main__":
    import uvicorn
    
    transmission_thread = threading.Thread(target=start_data_transmission, daemon=True)
    transmission_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)