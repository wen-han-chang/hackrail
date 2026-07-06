# -*- coding: utf-8 -*-
"""RailVolt MVP 儀表板 — streamlit run app.py"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import simulation as sim

st.set_page_config(page_title="RailVolt 軌道虛擬電廠 MVP", page_icon="🚈", layout="wide")


@st.cache_data
def load():
    day = sim.gen_train_day(42)
    base = sim.simulate(day, "baseline")
    rv = sim.simulate(day, "railvolt")
    pred = sim.fit_predictor()
    return day, base, rv, pred


day, base, rv, pred = load()
MIN, STATIONS = sim.MIN, sim.STATIONS


def hhmm(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


st.title("🚈 RailVolt 軌道虛擬電廠 — MVP 展示")
st.caption("隊名：綠色的夥伴｜參賽編號 HR-04370｜目前使用**模擬資料**展示完整管線，"
           "競賽資料到手後直接替換資料源（欄位結構已對齊釋出資料）")

tab1, tab2, tab3, tab4 = st.tabs(["📍 行控即時監控", "📊 節能回測", "🎯 載客預測驗證", "⚡ 虛擬電廠試算"])

# ---------- Tab 1 即時監控 ----------
with tab1:
    idx = st.select_slider("模擬日時刻（06:00–23:00，每分鐘一步）",
                           options=list(range(sim.N)), value=300,
                           format_func=lambda i: hhmm(MIN[i]),
                           help="拖動觀察一天營運中任一時刻的列車狀態")
    st.subheader(f"⏱ {hhmm(MIN[idx])}｜車次 1236（模擬）")
    ph, s_now, s_next = day["route"][idx]
    pos_txt = f"停靠 {STATIONS[s_now]}" if ph == "dwell" else \
        f"{STATIONS[s_now]} → {STATIONS[s_next]}（{ {'accel':'加速','cruise':'巡航','brake':'煞車・再生窗口'}[ph] }）"
    st.markdown(f"**列車位置：** {pos_txt}")

    cols = st.columns(sim.CARS)
    for c, col in enumerate(cols):
        occ = day["occ"][idx, c]
        dens = occ / sim.CAR_CAP
        with col:
            st.metric(f"車廂 {c+1}", f"載客 {dens:.0%}",
                      f"{rv['T'][idx, c]:.1f} °C")
            st.progress(min(dens, 1.0))

    c1, c2, c3 = st.columns(3)
    c1.metric("RailVolt 全車空調電力", f"{rv['P'][idx]:.1f} kW",
              f"{rv['P'][idx] - base['P'][idx]:+.1f} kW vs 傳統", delta_color="inverse")
    c2.metric("今日累計節電（至此刻）",
              f"{max(0, np.sum(base['P'][:idx]) - np.sum(rv['P'][:idx])) / 60:.1f} kWh")
    surge = rv["pred"][idx].sum() - day["occ"][idx].sum()
    c3.metric("30 分鐘載客預測", f"{rv['pred'][idx].sum():.0f} 人",
              f"{surge:+.0f} 人", delta_color="normal")

    st.info("💡 **最佳化建議**：" + (
        "偵測到航廈人流尖峰將至 → 已啟動預冷排程；" if surge > 80 else
        "載客平穩 → 各車廂依 PMV 舒適帶調整設定；") +
        ("壓縮機高載排入煞車窗口（使用再生電能）。" if ph == "brake" else
         "加速時段壓縮機降載讓路。" if ph == "accel" else "維持巡航節能模式。"))

# ---------- Tab 2 節能回測 ----------
with tab2:
    st.subheader("代表日回測：傳統固定 24°C vs RailVolt 策略")
    x = [hhmm(m) for m in MIN]
    fig = go.Figure()
    fig.add_scatter(x=x, y=base["P"], name="傳統反應式", line=dict(color="#c55a11"))
    fig.add_scatter(x=x, y=rv["P"], name="RailVolt", line=dict(color="#538135"))
    fig.update_layout(height=340, yaxis_title="全車空調電力 (kW)",
                      margin=dict(t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    save_kwh = base["kwh"] - rv["kwh"]
    save_pct = save_kwh / base["kwh"]
    yearly = save_kwh * 365 * 22   # 22 列車隊年化
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("傳統能耗（列車・日）", f"{base['kwh']:.0f} kWh")
    m2.metric("RailVolt 能耗", f"{rv['kwh']:.0f} kWh", f"-{save_pct:.1%}",
              delta_color="inverse")
    m3.metric("全車隊年化節電（22列）", f"{yearly/1000:,.0f} MWh")
    m4.metric("年減碳", f"{yearly * sim.CO2_FACTOR / 1000:,.0f} t CO₂e")

    c1, c2 = st.columns(2)
    c1.metric("乘客舒適時間比（傳統）", f"{base['comfort']:.0%}",
              help="PMV 簡化指標：含擁擠體感修正")
    c2.metric("乘客舒適時間比（RailVolt）", f"{rv['comfort']:.0%}",
              f"{rv['comfort'] - base['comfort']:+.0%}")
    st.caption(f"另計：煞車窗口再生電能折抵約 {rv['regen']:.1f} kWh/日（試算值）。"
               "所有參數為文獻量級，實測資料到手後以真實電壓電流回測取代。")

# ---------- Tab 3 預測驗證 ----------
with tab3:
    st.subheader("載客預測模組驗證（航廈站上車人數）")
    st.markdown("特徵：**航廈人流（滯後 30 分）＋時段**｜方法：最小平方法（物理×統計，非黑箱）"
                "｜訓練 12 個模擬日、測試 1 日。*正式版將以北捷逐車廂實測載重為真值重新驗證。*")
    fig2 = go.Figure()
    lim = float(max(pred["y_true"].max(), pred["y_pred"].max())) * 1.05
    fig2.add_scatter(x=pred["y_true"], y=pred["y_pred"], mode="markers",
                     marker=dict(color="#2e75b6", size=9), name="站停事件")
    fig2.add_scatter(x=[0, lim], y=[0, lim], mode="lines",
                     line=dict(dash="dash", color="gray"), name="完美預測")
    fig2.update_layout(height=380, xaxis_title="實際上車人數",
                       yaxis_title="預測上車人數", margin=dict(t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.metric("測試日平均絕對誤差 (MAE)", f"{pred['mae']:.1f} 人／班次")

# ---------- Tab 4 VPP ----------
with tab4:
    st.subheader("虛擬電廠聚合試算（大膽假設層）")
    n_trains = st.slider("聚合列車數", 4, 160, 22, help="22＝機場線量級；160＝全台捷運量級")
    peak = float(np.max(base["P"]))
    curtail = peak * 0.5 * n_trains / 1000
    st.markdown(f"""
| 項目 | 數值 |
|---|---|
| 單列車空調尖峰電力 | {peak:.0f} kW |
| 聚合 {n_trains} 列車可抑低容量（50%、30分鐘） | **{curtail:.2f} MW** |
| 相當於 | 約 {curtail*1000/5:.0f} 戶家庭冷氣同時關機 |
""")
    st.markdown("**機制**：車廂熱慣性＝儲能。用電尖峰事件前 20 分鐘預冷全車隊，"
                "事件中壓縮機集體降載，車溫僅緩升 1°C 內，PMV 仍在舒適帶——"
                "軌道系統從用電大戶變成電網的穩定器。")
    st.caption("台電需量反應獎勵制度下之收益另行試算；本頁為機制展示。")
