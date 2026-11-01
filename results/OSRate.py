import pandas as pd
import matplotlib.pyplot as plt

# ---------- Data ------------------------------------------------------
df = pd.DataFrame({
    "OS Ratio": ["0%", "30%", "50%", "80%", "100%"],
    "Precision": [0.0, 0.964, 0.984, 0.986, 0.996],
    "Recall":    [0.0, 0.788, 0.871, 0.939, 0.953],
    "F1‑score":  [0.0, 0.867, 0.924, 0.962, 0.974],
    "ROC‑AUC":   [0.74, 0.922, 0.922, 0.974, 0.99],
    "PR‑AUC":    [0.13, 0.904, 0.962, 0.980, 0.993],
})

df.set_index("OS Ratio", inplace=True)
df_T = df.T

# ---------- Plot ------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

df_T.plot(kind="barh",
          ax=ax,
          edgecolor="black"
          )

# Reverse order so Precision at top
ax.invert_yaxis()

# Add number labels
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", padding=4, label_type="edge", fontsize=8)

# Axis & title
ax.set_ylabel("Evaluation Metric", fontsize=12)
ax.set_xlabel("Score", fontsize=12)
ax.set_title("Evaluation Metrics Scores vs. Oversampling Ratio", pad=10)
ax.set_xlim(0, df_T.values.max() * 1.25)     # 25 % headroom

# 右上內側 (axes 座標 0~1)
ax.legend(title="OS Ratio",
          loc="upper right",         
          bbox_to_anchor=(0.98, 0.98),
          frameon=True)

plt.tight_layout()
plt.show()
