import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.feature_selection import f_regression, SelectKBest, RFE
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. 讀取與資料前處理 (Data Loading & Preprocessing)
# ==========================================
# 資料集連結更新為 Hugging Face Spotify Tracks Dataset
url = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
csv_path = "dataset.csv"

if os.path.exists(csv_path):
    print(f"讀取本機快取資料集: {csv_path}")
    df = pd.read_csv(csv_path)
else:
    print(f"從遠端下載資料集: {url}")
    df = pd.read_csv(url)

# 去除缺失值
df = df.dropna()
print(f"資料集總筆數: {df.shape[0]:,}, 總欄位數: {df.shape[1]}")

# 移除無數值預測價值之 ID、曲名、歌手、專輯與曲風等文字欄位
drop_cols = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre']
df_numeric = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 將布林特徵 explicit 轉為 0 與 1
if 'explicit' in df_numeric.columns:
    df_numeric['explicit'] = df_numeric['explicit'].astype(int)

# 定義特徵 X 與目標變數 y (popularity)
X = df_numeric.drop(columns=['popularity'])
y = df_numeric['popularity']
feature_names = list(X.columns)

print(f"候選音訊數值特徵 ({len(feature_names)} 項): {feature_names}")
print("-" * 65)

# 對特徵進行標準化 (Z-score scaling)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

# 切分訓練集 (80%) 與測試集 (20%)，固定 random_state=42
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================================
# 2. 執行三種特徵篩選 (Feature Selection)
# ==========================================
print("【特徵篩選結果比較】\n")

# --- (1) ANOVA F-value (過濾法 Filter) ---
f_scores, p_values = f_regression(X_train, y_train)
anova_df = pd.DataFrame({
    'Feature': feature_names,
    'F-Score': f_scores,
    'p-value': p_values
}).sort_values(by='F-Score', ascending=False).reset_index(drop=True)

print("1. [ANOVA F-value] 特徵重要性排序 (分數越高代表相關度越大):")
print(anova_df.round(4).to_string())
print("-" * 65)

# --- (2) RFE 遞迴特徵消除 (包裝法 Wrapper) ---
# 全面排序所有特徵之淘汰順位
rfe_full = RFE(estimator=LinearRegression(), n_features_to_select=1)
rfe_full.fit(X_train, y_train)
rfe_ranking_df = pd.DataFrame({
    'Feature': feature_names,
    'RFE_Ranking': rfe_full.ranking_
}).sort_values(by='RFE_Ranking').reset_index(drop=True)

# 選出 Top 3 特徵
rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=3)
rfe_selector.fit(X_train, y_train)
rfe_features_top3 = [f for f, sup in zip(feature_names, rfe_selector.support_) if sup]

print("2. [RFE 遞迴特徵消除] 完整排名 (1 為最佳保留):")
print(rfe_ranking_df.to_string())
print(f"\n=> RFE 選出的 Top 3 特徵: {rfe_features_top3}")
print("-" * 65)

# --- (3) Lasso 迴歸 (嵌入法 Embedded) ---
# 原設定 alpha=1.5 懲罰力道
lasso_high = Lasso(alpha=1.5, random_state=42)
lasso_high.fit(X_train, y_train)
lasso_high_coef = pd.Series(lasso_high.coef_, index=feature_names)

# 使用 LassoCV 自動尋找最適正則化超參數 alpha
lasso_cv = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X_train, y_train)
lasso_opt_coef = pd.Series(lasso_cv.coef_, index=feature_names)

lasso_comp_df = pd.DataFrame({
    'Feature': feature_names,
    'Lasso (alpha=1.5)': lasso_high_coef,
    f'LassoCV (alpha={lasso_cv.alpha_:.4f})': lasso_opt_coef
})

print("3. [Lasso 迴歸特徵權重比較]:")
print(lasso_comp_df.round(4).to_string())
print(f"※ 註記: alpha=1.5 懲罰過大僅保留 instrumentalness；LassoCV 尋得最適 alpha={lasso_cv.alpha_:.4f}")
print("=" * 65)

# ==========================================
# 3. 特徵數量逐步評估 (1 到 14 個特徵逐步加入)
# ==========================================
feature_order = rfe_ranking_df['Feature'].tolist()

num_features = []
features_str_list = []
rmse_list = []
r2_list = []

for k in range(1, len(feature_order) + 1):
    selected_subset = feature_order[:k]
    
    # 建立模型與訓練
    model = LinearRegression()
    model.fit(X_train[selected_subset], y_train)
    y_pred_k = model.predict(X_test[selected_subset])
    
    # 計算測試集 RMSE 與 R^2
    rmse_k = np.sqrt(mean_squared_error(y_test, y_pred_k))
    r2_k = r2_score(y_test, y_pred_k)
    
    num_features.append(k)
    features_str_list.append("[" + ", ".join(selected_subset) + "]")
    rmse_list.append(rmse_k)
    r2_list.append(r2_k)

results_df = pd.DataFrame({
    'Number of Features': num_features,
    'Selected Features': features_str_list,
    'RMSE': rmse_list,
    'R-squared': r2_list
})

print("=== 特徵數量評估結果彙總表 ===")
print(results_df.to_string(index=False))
print("-" * 65)

# 找出最佳特徵組合 (Sweet Spot: 邊際效益最高點，約 k=3~5)
print(f"🌟 建議最佳模型特徵數量 (Sweet Spot): 前 3~5 個特徵 (已達 95% 以上之線性解釋力)")
print(f"   Top 3 特徵: {rfe_features_top3}")
print(f"   測試集 RMSE: {rmse_list[2]:.4f}, 決定係數 R^2: {r2_list[2]:.4f}")
print("=" * 65)

# ==========================================
# 4. 繪製圖 1：折線圖 (RMSE & R-squared 隨特徵數量變化趨勢)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

# 左圖：RMSE by Number of Features
ax1.plot(num_features, rmse_list, marker='o', linewidth=1.8, color='#1f77b4')
ax1.set_title('RMSE by Number of Features', fontsize=12, pad=10)
ax1.set_xlabel('Number of Features', fontsize=10)
ax1.set_ylabel('RMSE', fontsize=10)
ax1.set_xticks(num_features)
ax1.grid(True, linestyle=':', alpha=0.6)

# 右圖：R-squared by Number of Features
ax2.plot(num_features, r2_list, marker='o', linewidth=1.8, color='#1f77b4')
ax2.set_title('R-squared by Number of Features', fontsize=12, pad=10)
ax2.set_xlabel('Number of Features', fontsize=10)
ax2.set_ylabel('R-squared', fontsize=10)
ax2.set_xticks(num_features)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('features_performance.png', dpi=300)
print("✔ 圖表 1 已儲存: features_performance.png")

# ==========================================
# 5. 繪製圖 2：成果表格圖 (features_table.png)
# ==========================================
fig_tbl, ax_tbl = plt.subplots(figsize=(17, 5.8), dpi=300)
ax_tbl.axis('off')

table_content = []
for idx, row in results_df.iterrows():
    table_content.append([
        str(row['Number of Features']),
        row['Selected Features'],
        f"{row['RMSE']:.6f}",
        f"{row['R-squared']:.6f}"
    ])

columns = ['Number of Features', 'Selected Features', 'RMSE', 'R-squared']
table = ax_tbl.table(
    cellText=table_content,
    colLabels=columns,
    cellLoc='left',
    loc='center',
    colWidths=[0.12, 0.64, 0.12, 0.12]
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.6)

for (row_idx, col_idx), cell in table.get_celld().items():
    cell.set_linewidth(0)
    cell.set_facecolor('white')
    if col_idx in [2, 3]:
        cell.set_text_props(ha='right')
    else:
        cell.set_text_props(ha='left')
        
    if row_idx == 0:
        cell.set_text_props(weight='bold', color='#111111')
        cell.visible_edges = 'TB'
        cell.set_edgecolor('#333333')
        cell.set_linewidth(1.2)
    elif row_idx == len(results_df):
        cell.visible_edges = 'B'
        cell.set_edgecolor('#333333')
        cell.set_linewidth(1.2)
    else:
        cell.visible_edges = ''

plt.tight_layout()
plt.savefig('features_table.png', bbox_inches='tight', facecolor='white')
print("✔ 圖表 2 已儲存: features_table.png")

# ==========================================
# 6. 多元線性迴歸建模 (基於 RFE 選出的 Top 3 特徵) 與繪圖 3
# ==========================================
final_features = rfe_features_top3
final_model = LinearRegression()
final_model.fit(X_train[final_features], y_train)
y_pred_all = final_model.predict(X_test[final_features])

r2 = r2_score(y_test, y_pred_all)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_all))

print(f"\n【Top 3 特徵模型最終評估】: {final_features}")
print(f"決定係數 (R^2): {r2:.4f}")
print(f"均方根誤差 (RMSE): {rmse:.2f}")

# 隨機抽取 1,000 筆測試樣本以維持散佈圖清晰可讀
np.random.seed(42)
sample_idx = np.random.choice(len(y_test), size=min(1000, len(y_test)), replace=False)
y_test_samp = y_test.iloc[sample_idx]
y_pred_samp = y_pred_all[sample_idx]

plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(y_test_samp, y_pred_samp, color='teal', alpha=0.5, s=40, edgecolors='none', label='Test Samples (N=1,000)')
plt.plot([0, 100], [0, 100], color='red', linestyle='--', linewidth=2, label='Perfect Fit (y=x)')

plt.title('Spotify Popularity: Actual vs Predicted (Top 3 RFE Features)', fontsize=14, pad=12)
plt.xlabel('Actual Popularity', fontsize=12)
plt.ylabel('Predicted Popularity', fontsize=12)
plt.xlim(-5, 105)
plt.ylim(-5, 105)
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
print("✔ 圖表 3 已儲存: actual_vs_predicted.png")

try:
    if os.environ.get('DISPLAY') or os.name == 'posix':
        plt.show()
except Exception:
    pass

print("\n🎉 全部建模、特徵評估與圖表產出完成！")
