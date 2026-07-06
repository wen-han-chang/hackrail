# RailVolt MVP — 綠色的夥伴（HR-04370）

軌道列車空調預測式節能與虛擬電廠決策系統的最小可行性產品。
目前以**模擬資料**展示完整管線；欄位結構對齊 HackRail 釋出資料，
真實資料到手後只需替換 `simulation.py` 的資料來源函式。

## 執行方式

```
pip install streamlit plotly pandas numpy
streamlit run app.py
```

瀏覽器自動開啟 http://localhost:8501

## 架構（對應構想書圖一）

- `simulation.py` — 資料層＋模型層
  - `gen_train_day()`：模擬資料產生器（列車遙測、航廈人流、氣溫；真實資料接入點）
  - `simulate()`：車廂熱力模型（RC 熱網路）＋兩種空調策略（傳統 vs RailVolt）
  - `fit_predictor()`：載客預測模組（航廈人流領先指標，最小平方法）
- `app.py` — 應用層（Streamlit 儀表板，四個分頁：即時監控／節能回測／預測驗證／VPP 試算）

## 真實資料接入待辦

1. 桃捷「列車資料」→ 取代模擬遙測，校準 C_TH、UA、AC_MAX 等物理參數
2. 「一二航廈旅客出入境分時量」→ 取代 `gen_env()` 的航廈人流
3. 北捷「列車擁擠度」載重 → `fit_predictor()` 的真值重新驗證
4. 回測數字確認後，回填構想書圖二／圖三的示意值
