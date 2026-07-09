# RailVolt MVP — 綠色的夥伴（HR-04370）

軌道列車空調預測式節能與虛擬電廠決策系統的最小可行性產品。
目前以**模擬資料**展示完整管線；欄位結構對齊 HackRail 釋出資料，
真實資料到手後只需替換資料來源即可產出真實回測數字。

## 執行方式

```
pip install streamlit plotly pandas numpy
streamlit run app.py
```

瀏覽器開啟 http://localhost:8501

## 儀表板分頁（7 個）

1. **單一班次模擬** — 選車型／方向／班次，看整趟逐站逐區間的空調策略與省電量
2. **行控即時監控** — 時間軸重播全天，逐車廂載客／溫度／功率與最佳化建議
3. **節能回測** — 傳統固定 24°C vs RailVolt 全天功率曲線與節能減碳數字
4. **空調最佳化** — 手動調 7 個控制參數即時看結果，或自動搜尋最省能策略
5. **載客預測驗證** — 航廈人流領先指標預測載客之準確度散點
6. **牽引電力反推載客** — 由牽引功率反演列車質量→人數（純物理，MAE≈11 人、R²≈0.99）
7. **虛擬電廠試算** — 車隊聚合可調度電力容量估算

## 程式架構

- `simulation.py` — 資料引擎與物理模型
  - `gen_train_day()`：全日模擬資料產生器（列車遙測、航廈人流、氣溫）
  - `simulate(day, policy)`：車廂 RC 熱力模型 ＋ 參數化空調策略
  - `optimize_policy()`：舒適約束下最小化能耗之隨機搜尋（MPC 最佳化雛型）
  - `build_trip()` / `run_trip()` / `trip_compare()`：單一班次逐站逐區間模擬
  - `fit_predictor()`：航廈人流載客預測（最小平方法）
  - `infer_load_from_traction()`：牽引電力反推載客
- `app.py` — Streamlit 儀表板（上述 7 分頁）
- `dataio.py` — 真實資料接入層；`data/` 內有真檔即讀，否則回傳同欄位模擬資料

## 真實資料接入待辦

1. 桃捷「列車資料」→ 放進 `data/`，`dataio.load_train_telemetry()` 自動改讀真檔；
   以實測電壓電流校準 `C_TH`、`UA`、`AC_MAX` 等物理參數
2. 「一二航廈旅客出入境分時量」→ `dataio.load_airport_flow()`
3. 北捷「列車擁擠度」載重 → `dataio.load_crowding()`，作為載客預測真值重新驗證
4. 回測數字確認後，回填構想書圖二／圖三的示意值

> 註：`data/` 已列入 `.gitignore`；競賽釋出資料僅限競賽用途、禁止外流。
