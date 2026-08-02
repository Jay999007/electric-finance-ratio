# 台灣電金比風險偏好指標

每天從臺灣證券交易所（TWSE）抓取：

- 電子類指數
- 金融保險類指數

並計算：

```text
電金比 = 電子類指數 ÷ 金融保險類指數
```

系統會自動產生黑底圖表、每日 CSV、反轉訊號表，並發布到 GitHub Pages。

## 指標規則

- 電金比高於均線：`Risk On`
- 電金比低於均線：`Risk Off`
- 紅點：均線斜率由負轉正
- 綠點：均線斜率由正轉負
- 預設同時繪製 20 日與 120 日均線

目前 `buffer_pct` 預設為 `0`，也就是直接以電金比是否高於／低於均線判斷。若要降低貼線反覆穿越，可改成 `0.005`，代表上下各 0.5% 緩衝。

---

# 一、建立 GitHub 自動更新版

## 1. 建立新的 GitHub Repository

建議名稱：

```text
electric-finance-ratio
```

公開或私人皆可，但 GitHub Pages 的可用條件會依 GitHub 帳戶方案而不同。

## 2. 上傳本資料夾內全部檔案

必須包含隱藏資料夾：

```text
.github/workflows/daily.yml
```

完整結構如下：

```text
electric-finance-ratio/
├─ .github/
│  └─ workflows/
│     └─ daily.yml
├─ data/
├─ docs/
├─ src/
│  └─ update_chart.py
├─ config.json
├─ requirements.txt
├─ run_local.bat
└─ README.md
```

## 3. 開啟 GitHub Actions 寫入權限

進入：

```text
Settings
→ Actions
→ General
→ Workflow permissions
→ Read and write permissions
→ Save
```

這個權限用來把每日新增的 CSV 與圖表提交回 Repository。

## 4. 開啟 GitHub Pages

進入：

```text
Settings
→ Pages
→ Build and deployment
→ Source
→ GitHub Actions
```

## 5. 第一次手動執行

進入：

```text
Actions
→ Daily Electric-Finance Ratio
→ Run workflow
```

第一次會回補預設 500 個日曆天，請等待數分鐘。之後每天只會更新最近資料。

## 6. 開啟網站

執行成功後，在：

```text
Settings → Pages
```

可以看到網址，通常格式是：

```text
https://你的帳號.github.io/electric-finance-ratio/
```

---

# 二、自動執行時間

預設為台北時間每週一至週五：

```text
18:23
```

設定位置：

```yaml
on:
  schedule:
    - cron: "23 18 * * 1-5"
      timezone: "Asia/Taipei"
```

若遇台股休市，程式不會新增錯誤資料，只會沿用最近一個交易日。

---

# 三、本機 Windows 執行

雙擊：

```text
run_local.bat
```

程式會自動：

1. 建立 `.venv`
2. 安裝 Python 套件
3. 抓取 TWSE 資料
4. 產生圖表
5. 開啟 `docs/index.html`

需要先安裝 Python 3.11 以上版本。

---

# 四、主要輸出

```text
data/electronics_finance_ratio.csv
```

保存原始每日資料：

- 日期
- 電子類指數
- 金融保險類指數
- 電金比

```text
docs/latest.png
```

最新版圖表。

```text
docs/index.html
```

GitHub Pages 首頁。

```text
docs/data.csv
```

包含均線、斜率、多空狀態與穿越訊號的完整資料。

```text
docs/signals.csv
```

只保留均線斜率反轉日期。

---

# 五、修改參數

編輯 `config.json`：

```json
{
  "backfill_days": 500,
  "refresh_days": 10,
  "chart_days": 260,
  "moving_averages": [20, 120],
  "buffer_pct": 0.0,
  "request_interval_seconds": 0.25
}
```

參數說明：

| 參數 | 用途 |
|---|---|
| `backfill_days` | 第一次執行向前抓取的日曆天數 |
| `refresh_days` | 每次重抓最近幾天，補發布延遲或修正 |
| `chart_days` | 圖上顯示的最近交易日數 |
| `moving_averages` | 要畫的移動平均線 |
| `buffer_pct` | 多空判斷緩衝比例 |
| `request_interval_seconds` | 每次 TWSE 請求間隔 |

例如只畫 20 日線：

```json
"moving_averages": [20]
```

例如加入上下 0.5% 緩衝：

```json
"buffer_pct": 0.005
```

---

# 六、資料來源與限制

程式使用 TWSE 官方每日收盤行情，掃描其中的「電子類指數」與「金融保險類指數」。第一次回補需要逐日查詢，因此時間較長；之後每日只抓最近資料。

這個指標反映電子與金融的相對強弱，適合觀察市場風格與風險偏好，但不代表加權指數必然上漲或下跌。
