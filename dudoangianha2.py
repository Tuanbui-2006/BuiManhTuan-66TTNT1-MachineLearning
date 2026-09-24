from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
RANDOM_SEED = 42
SPLIT_SEED = 10
N_SAMPLES = 90

rng = np.random.default_rng(RANDOM_SEED)
x = np.sort(rng.uniform(-3, 3, N_SAMPLES))
y_true = np.sin(1.5 * x) + 0.25 * x
y = y_true + rng.normal(loc=0, scale=0.25, size=N_SAMPLES)

# Tạo ba tập dữ liệu theo chỉ số; test được giữ lại để đánh giá cuối cùng.
all_idx = np.arange(N_SAMPLES)
train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.40, random_state=SPLIT_SEED
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=SPLIT_SEED
)

split = np.empty(N_SAMPLES, dtype=object)
split[train_idx] = "train"
split[val_idx] = "validation"
split[test_idx] = "test"

df = pd.DataFrame({"x": x, "y": y, "y_true": y_true, "split": split})
data_path = Path("du_lieu_overfitting.csv")
df.to_csv(data_path, index=False)

print(f"Đã tạo: {data_path.resolve()}")
print(df["split"].value_counts())
df.head(10)
colors = {"train": "#2563EB", "validation": "#F59E0B", "test": "#DC2626"}

fig, ax = plt.subplots(figsize=(10, 5.5))
for group in ["train", "validation", "test"]:
    part = df[df["split"] == group]
    ax.scatter(part["x"], part["y"], s=48, alpha=0.78,
               label=group.capitalize(), color=colors[group])

ax.plot(x, y_true, color="black", linewidth=2.2, label="Quan hệ thật")
ax.set(title="Bộ dữ liệu phi tuyến có nhiễu", xlabel="x", ylabel="y")
ax.legend(ncol=4)
plt.show()