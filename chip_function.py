import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 參數設定 ===
filename = r"C:\Users\USER\Downloads\function.csv"
df = pd.read_csv(filename)

# 清除欄位名稱空白與 BOM 字元
df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True)

# 限制範圍 0–14000
df = df[df['sampleTime1'] <= 14000]

# Downsample（每 20 筆取一筆）
df_down = df.iloc[::20, :].reset_index(drop=True)

# === Baseline 校正（每條 Intensity 扣掉自己 baseline）===
intensity_cols = ['Intensity1', 'Intensity2', 'Intensity3', 'Intensity4']
baseline_region = df_down[(df_down['sampleTime1'] >= 0) & (df_down['sampleTime1'] <= 300)]
baseline_values = baseline_region[intensity_cols].mean()

for col in intensity_cols:
    df_down[col] = df_down[col] - baseline_values[col]

# === 計算 ΔRIU 前，先轉換 Intensity 為 RIU ===
for col in intensity_cols:
    df_down[col] = (df_down[col] + 22.6942) / 175611.1693

# === 計算 baseline 平均值（轉 RIU 後）===
region1 = df_down[(df_down['sampleTime1'] >= 0) & (df_down['sampleTime1'] <= 300)]
avg1 = region1[intensity_cols].mean().mean()

# === 將整組資料平移，使 baseline 為 0 ===
for col in intensity_cols:
    df_down[col] = df_down[col] - avg1

# === Step 2: 畫出多條 RIU 線 ===
plt.figure(figsize=(15, 7))
for col in intensity_cols:
    plt.plot(df_down['sampleTime1'], df_down[col], label=col, linewidth=1)

# === Step 3: 標記功能化步驟區段 ===
highlight_regions = [
    (169, 1377, "2M NaCl + 0.1M NaOH"),
    (5575, 7533, "0.4M EDC+0.1M NHS\n (MES Buffer pH5)"),
    (8607, 9922, "Antibody immobilization"),
    (11402, 12829, "Ethanolamine\n hydrochloride")
]

ymin, ymax = plt.ylim()
text_y_ratios = [0.95, 0.95, 0.75, 0.85]

for i, (start, end, label) in enumerate(highlight_regions):
    plt.axvspan(start, end, color='orange', alpha=0.1)
    mid = (start + end) / 2
    text_y = ymax * text_y_ratios[i]
    plt.text(mid, text_y, label,
             ha='center', va='top', fontsize=13, color='black',
             bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))

# === Step 4: 平均線與 Δ ===
region2 = df_down[(df_down['sampleTime1'] >= 13067) & (df_down['sampleTime1'] <= 13167)]
region3 = df_down[(df_down['sampleTime1'] >= 8307) & (df_down['sampleTime1'] <= 8607)]
region4 = df_down[(df_down['sampleTime1'] >= 10848) & (df_down['sampleTime1'] <= 11148)]

avg2 = region2[intensity_cols].mean().mean()
avg3 = region3[intensity_cols].mean().mean()
avg4 = region4[intensity_cols].mean().mean()
delta = avg2 - 0  # 因為 avg1 已設為 0
delta2 = avg4 - avg3

x_start = df_down['sampleTime1'].min()
x_end = df_down['sampleTime1'].max()

plt.hlines(0, x_start, x_end, colors='blue', linestyles='dashed', linewidth=1.5, label='before functionalization')
plt.hlines(avg2, x_start, x_end, colors='red', linestyles='dashed', linewidth=1.5, label='after functionalization')
plt.hlines(avg3, x_start, x_end, colors='green', linestyles='dashed', linewidth=1.5, label='before antibody immobilization')
plt.hlines(avg4, x_start, x_end, colors='purple', linestyles='dashed', linewidth=1.5, label='after antibody immobilization')

mid_x = (x_start + x_end) / 2
mid_y = (0 + avg2) / 2
plt.text(mid_x + 9500, mid_y - 0.0006, f'Δ between functionalization\n = {delta:.2e} RIU', ha='center', fontsize=15, color='black',
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
plt.text(mid_x + 9500, mid_y + 0.0002, f'Δ between immobilization\n= {delta2:.2e} RIU', ha='center', fontsize=15, color='black',
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# === Step 5: 美化 ===
plt.xlabel("Time (s)", fontsize=15)
plt.ylabel("ΔRIU", fontsize=15)
plt.title("Immobilization", fontsize=15)
plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.grid(False)
plt.show()
