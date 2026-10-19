# 🎵 Spotify Top 50 音訊特徵工程與熱門度多元線性回歸分析

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?style=for-the-badge&logo=github)](https://chenyichencyc.github.io/HW2_multiple-linear-regression-Spotify-/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F89939?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

本專案針對經典的 **Spotify 2019 Top 50 全球串流榜單** 進行深入的特徵工程（Feature Engineering）與多元線性回歸（Multiple Linear Regression）建模。實作並比較了 **ANOVA F-test (Filter Method 過濾法)**、**RFE 遞迴特徵消除 (Wrapper Method 包裝法)** 以及 **Lasso 正則化 (Embedded Method 嵌入法)** 三種核心特徵篩選技術，評估隨特徵數量增加時模型在未見測試集（20% Test Set）上的泛化能力與偏差-變異權衡（Bias-Variance Tradeoff）。

---

## 🌐 線上互動展示儀表板 (Live Interactive Demo)

本專案已支援部署於 GitHub Pages，提供動態下拉式選單與即時計算功能：
👉 **[點此造訪 GitHub Pages 線上儀表板](https://chenyichencyc.github.io/HW2_multiple-linear-regression-Spotify-/)**

### 網頁核心架構與功能：
- **首頁即時呈現核心分析與預測器**：
  - 📈 **特徵數量評估與最佳化 (Sweet Spot)**：提供互動式 Chart.js 折線圖（雙 Y 軸展示 RMSE 與 $R^2$）與 Matplotlib 原生 PNG 圖（`features_performance.png`、`features_table.png`）無縫切換，附完整 1~9 特徵評估指標數據表。
  - 🎧 **Spotify 即時熱門度預測計算機**：支援調整正向度 (Valence)、說話度 (Speechiness)、現場感 (Liveness)、BPM、能量與舞曲度滑桿，即時利用多元迴歸方程式計算熱門度（Popularity 0~100）並同步繪製音訊風向雷達圖。
  - 🎵 **熱門名曲一鍵帶入**：可直接載入 *Billie Eilish - bad guy*、*Shawn Mendes - Señorita*、*Ariana Grande - 7 rings*、*Lil Nas X - Old Town Road* 的真實音訊參數。
- **三大特徵篩選方法深度分析（專屬下拉式選單切換）**：
  1. 🔬 **ANOVA F-test (Filter Method 過濾法)**：9 項音訊特徵之 F-Score 條形圖與 p-value 統計顯著性分析
  2. 🔄 **RFE 遞迴特徵消除 (Wrapper Method 包裝法)**：特徵淘汰輪次、保留排名與標準化說明
  3. 🎯 **Lasso L1 正則化 (Embedded Method 嵌入法)**：支援動態滑動調整 $\alpha$ 懲罰力度，動態展示特徵權重收縮過程，深入剖析為何 $\alpha=1.5$ 懲罰過大導致權重全數歸零，以及 `LassoCV` 尋獲的最佳 $\alpha \approx 0.612$ 之稀疏化效果。
- **50 首榜單歌曲原始資料集檢視器**：支援歌名/歌手即時搜尋與曲風過濾，點擊「導入預測器」可直接將該曲參數載入預測模型。

---

## 📊 核心實驗成果 (Experimental Results)

### 1. 特徵數量逐步評估對照表

模型在切分後的測試集（20% Test Set，固定 `random_state=42`）上進行逐步加入特徵（1 至 9 個）之成效評估：

| 特徵數 (k) | 選取特徵組合 (依 RFE 排名由先至後) | 測試集 RMSE | 測試集 R-squared | 說明 |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `[Valence.]` (正向度) | `6.651936` | `-0.137780` | 單一情緒特徵，具統計顯著性 |
| **2** | `[Valence., Speechiness.]` | **`6.468910`** | **`-0.076030`** | 🌟 **最佳模型 (Sweet Spot)**，測試集誤差最低 |
| **3** | 加入 `Liveness` (現場感) | `6.489122` | `-0.082764` | 🎯 **RFE Top 3 推薦組合**，表現平穩 |
| **4** | 加入 `Length.` (時長) | `6.493394` | `-0.084190` | 誤差持平 |
| **5** | 加入 `Loudness..dB..` (響度) | `6.505599` | `-0.088270` | 誤差開始微幅增加 |
| **6** | 加入 `Beats.Per.Minute` (節奏) | `6.486942` | `-0.082037` | 無實質改善 |
| **7** | 加入 `Acousticness..` (原聲度) | `6.969816` | `-0.249121` | ⚠️ **開始出現顯著過擬合 (Overfitting)** |
| **8** | 加入 `Energy` (能量強度) | `7.317351` | `-0.376797` | 誤差劇烈飆升 |
| **9** | 加入 `Danceability` (全特徵) | `7.336715` | `-0.384093` | 泛化能力最差 |

---

### 2. 成果圖表展示

| 圖表 1：特徵數量評估折線圖 (`features_performance.png`) | 圖表 2：評估數據對照表 (`features_table.png`) |
| :---: | :---: |
| ![features_performance](features_performance.png) | ![features_table](features_table.png) |

| 圖表 3：實際熱門度 vs 預測熱門度散佈圖 (`actual_vs_predicted.png`) |
| :---: |
| ![actual_vs_predicted](actual_vs_predicted.png) |

---

## 🔬 深入理論與機制探討 (Key Insights)

### 1. 最適特徵數量 (Sweet Spot) 與過擬合現象
- 當使用 **2 個特徵 (`Valence.` 與 `Speechiness.`)** 或 **3 個特徵 (加入 `Liveness`)** 時，測試集 RMSE 達到全場最低值 **6.4689 ~ 6.4891**。
- **維度災難與噪聲累積**：當加入超過 6 個特徵後，特別是加入 `Acousticness`、`Energy`、`Danceability` 時，RMSE 從 6.48 大幅飆升至 **7.3367**。這是因為 Top 50 歌曲的 Energy 與 Danceability 普遍偏高，與熱門度缺乏線性區分度；強行引入弱相關特徵會讓模型強行配適訓練集噪聲，使模型方差（Variance）急遽放大，在未見測試集上產生嚴重的泛化崩潰。

### 2. 為什麼測試集 $R^2$ 會呈現負值？
- 在 Scikit-Learn 中，決定係數計算公式為：
  $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y}_{\text{test}})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
- 當樣本規模較小（測試集僅 10 首歌曲）且目標變數方差有限時，若加入過多無效特徵使模型預測出現較大殘差，$\text{SS}_{\text{res}}$ 就可能超過基準線 $\text{SS}_{\text{tot}}$，導致 $R^2 < 0$。
- **這恰恰證明了「特徵篩選 (Feature Selection)」在實際機器學習應用中的不可或缺性**：在特徵過多的情況下，未經篩選的模型在測試集上的表現甚至劣於直接猜測均值！

### 3. 三大特徵篩選方法特點比較

| 特徵篩選方法 | 所屬類別 | 核心優點 | 本資料集關鍵發現 |
| :--- | :--- | :--- | :--- |
| **ANOVA F-test** | Filter Method (過濾法) | 計算速度最快，獨立於特定模型 | `Valence.` ($F=5.39, p=0.0245$) 為唯一達統計顯著之特徵，呈負相關 |
| **RFE** | Wrapper Method (包裝法) | 考慮特徵與模型的整體配適度與互補性 | 成功辨識出 `Liveness` 與 `Speechiness` 的互補性，選出 Top 3 |
| **Lasso 迴歸** | Embedded Method (嵌入法) | 內建 L1 正則化稀疏性，自動壓縮無效權重 | 原始 $\alpha=1.5$ 懲罰過大將所有權重歸零；經 `LassoCV` 最佳化 $\alpha \approx 0.612$，成功精準保留 `Valence`、`Speechiness` 與 `BPM` |

---

## 🛠️ 專案檔案結構 (Project Structure)

```
(Hw2)L05_spotify/
├── top50.csv                  # Spotify Top 50 2019 原始資料集
├── main.py                    # 完整特徵工程、篩選與回歸分析執行主程式
├── index.html                 # 部署於 GitHub Pages 的互動式現代化前端儀表板
├── songs_data.json            # 50 首歌曲結構化資料 (供前端檢視與預測器載入)
├── lasso_path.json            # Lasso 正則化係數路徑數據 (供動態滑桿即時渲染)
├── features_performance.png   # 特徵數量評估折線圖 (300 DPI)
├── features_table.png         # 特徵評估結果數據表格圖 (300 DPI)
├── actual_vs_predicted.png    # 實際熱門度 vs 預測熱門度診斷圖 (300 DPI)
├── requirements.txt           # Python 套件依賴清單
└── README.md                  # 專案完整研究分析報告
```

---

## 🚀 快速開始 (Quick Start)

### 1. 安裝環境依賴
```bash
pip install -r requirements.txt
```

### 2. 執行回歸分析與特徵評估
```bash
python3 main.py
```
執行後終端將印出：
- ANOVA 特徵重要性排序與 p-value
- RFE 9 特徵排名與選出之 Top 3 特徵
- Lasso 權重比較（原 $\alpha=1.5$ vs `LassoCV` 最佳 $\alpha=0.612$）
- 1~9 個特徵逐步加入之評估指標彙總表
- 自動輸出並儲存三張 300 DPI 高解析度圖表

### 3. 開啟互動式網頁儀表板
直接於瀏覽器中開啟 `index.html`，或啟動本機伺服器：
```bash
python3 -m http.server 8080
```
造訪 [http://localhost:8080](http://localhost:8080) 即可體驗互動式熱門度預測器與特徵篩選比較切換！

---

## 📝 授權 (License)
本專案遵循 MIT License 開源協議。
