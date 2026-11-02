# Real Time Testing
# 啟動 FastAPI 伺服器，並在伺服器就緒後，依序把 10 筆保留的原始資料送到 /predict
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from anomaly_detection.Preprocess import transform


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- 0) Settings -----
HOST = "127.0.0.1"          # 只允許本機連線；要讓其他裝置連線再改成 "0.0.0.0"
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"
SEND_INTERVAL = 3           # 每筆資料間隔秒數
REQUEST_TIMEOUT = 30        # 偵測到異常時要算 SHAP + 畫圖，給足時間

ROOT = Path(__file__).resolve().parent.parent.parent
PLOT_DIR = ROOT / "results" / "plots_output"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
TEST_DATA_PATH = ROOT / "data" / "streaming" / "test_data.csv"

model_path = ROOT / "artifacts" / "model" / "model.cbm"
threshold_path = ROOT / "artifacts" / "model" / "threshold.npy"
background_path = ROOT / "artifacts" / "model" / "bg_1000.npy"
idx_path = ROOT / "artifacts" / "preprocessing" / "idx.npy"

test_data = pd.read_csv(TEST_DATA_PATH)

# ----- 1) Model & Explainer -----
model = CatBoostClassifier()
model.load_model(model_path)
threshold = float(np.load(threshold_path))
background = np.load(background_path)
explainer = shap.TreeExplainer(model, background)
IDX = np.load(idx_path)

# ----- 2) FastAPI -----
app = FastAPI(title="UCI-SECOM Anomaly Detection API")


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
    score = float(model.predict_proba(x)[0, 1])
    is_abn = score >= threshold
    plot_nm = None
    if is_abn:
        x_df = pd.DataFrame(x, columns=IDX.astype(str))
        shap_val = explainer(x_df)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")   # 加上微秒，避免同一秒內檔名重複
        plot_nm = f"{ts}.png"
        generate_waterfall_plot(shap_val[0], score, PLOT_DIR / plot_nm)
    return score, is_abn, plot_nm


# --- 2-A  HTTP 端點 ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ResultOut)
def predict(item: LineIn):
    # 輸入格式錯誤 → 400；模型推論出錯 → 500
    try:
        x_std = transform(item.line.split(","))
    except Exception as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    try:
        score, is_abn, plot_nm = run_inference(x_std)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"score": score, "is_anomaly": is_abn, "plot_file": plot_nm}


@app.get("/plot/{file}")
def get_plot(file: str):
    fp = PLOT_DIR / file
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(fp, media_type="image/png")


# --- 2-B  自動數據傳輸 ---
def wait_until_ready(timeout=60):
    """每 0.5 秒檢查一次 /health，直到伺服器回應或超時。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{BASE_URL}/health", timeout=1).ok:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def stream_test_data():
    if not wait_until_ready():
        logger.error("API did not start within 60 s, streaming aborted.")
        return

    sent = 0
    for index, row in test_data.iterrows():
        raw_data_array = row.drop(['Pass/Fail', 'Time']).values
        line = ",".join("" if pd.isna(v) else str(v) for v in raw_data_array)
        true_label = "anomaly" if row['Pass/Fail'] == 1 else "normal"
        try:
            response = requests.post(f"{BASE_URL}/predict", json={"line": line}, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Index {index}: send failed: {e}")
            continue   # 單筆失敗就跳過，繼續送下一筆
        sent += 1
        logger.info(f"Index {index} (true: {true_label}) -> {response.json()}")
        time.sleep(SEND_INTERVAL)

    logger.info(f"Streaming finished: {sent}/{len(test_data)} records sent. "
                f"Server is still running, press Ctrl+C to stop.")


if __name__ == "__main__":
    import uvicorn

    threading.Thread(target=stream_test_data, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT)