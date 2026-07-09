# -*- coding: utf-8 -*-
"""RailVolt MVP 儀表板 — streamlit run app.py"""
import datetime as dt

import numpy as np
import pandas as pd
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


@st.cache_data
def load_infer():
    return sim.infer_load_from_traction()


day, base, rv, pred = load()
MIN, STATIONS = sim.MIN, sim.STATIONS


def hhmm(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


st.title("🚈 RailVolt 軌道虛擬電廠 — MVP 展示")
st.caption("隊名：綠色的夥伴｜參賽編號 HR-04370｜目前使用**模擬資料**展示完整管線，"
           "競賽資料到手後直接替換資料源（欄位結構已對齊釋出資料）")

tab0, tab1, tab2, tab6, tab3, tab5, tab4 = st.tabs(
    ["🚆 單一班次模擬", "📍 行控即時監控", "📊 節能回測", "🎛️ 空調最佳化",
     "🎯 載客預測驗證", "🔩 牽引電力反推載客", "⚡ 虛擬電廠試算"])

# ---------- Tab 0 單一班次模擬 ----------
with tab0:
    st.subheader("選一班實際班次，看它整趟怎麼開、省多少電")
    c1, c2, c3 = st.columns(3)
    veh_name = c1.selectbox("車型", list(sim.VEHICLES.keys()))
    direction = c2.selectbox("方向", ["南下（往環北）", "北上（往台北）"])
    tt = sim.timetable()
    svc_labels = [f"{hhmm(s['dep'])}　{s['kind']}" for s in tt]
    default_i = next((i for i, s in enumerate(tt)
                      if s["dep"] == 8 * 60 and s["kind"] == "直達"), 0)
    svc_i = c3.selectbox("班次（時刻表）", range(len(tt)),
                         index=default_i, format_func=lambda i: svc_labels[i])

    veh = sim.VEHICLES[veh_name]
    cmp = sim.trip_compare(tt[svc_i], veh, "南下" if "南下" in direction else "北上")
    r = cmp["route"]

    st.markdown(
        f"**{hhmm(r['dep'])} 發車　{tt[svc_i]['kind']}車　{direction}**　｜　"
        f"{r['n_stops']} 停靠　｜　{r['trip_km']:.1f} km　｜　行駛 {r['arr'] - r['dep']} 分　｜　"
        f"車型：{veh['cars']} 節・每車 {veh['cap']} 人・空載 {veh['tare_t']} 噸　｜　"
        f"尖峰載客 {r['occ_peak']:.0f} 人/車")

    m = st.columns(5)
    m[0].metric("本趟傳統耗電", f"{cmp['base']['total_kwh']:.1f} kWh",
                help="固定設定 24°C")
    m[1].metric("本趟 RailVolt 耗電", f"{cmp['rv']['total_kwh']:.1f} kWh",
                f"-{cmp['saved_pct']:.1%}", delta_color="inverse")
    m[2].metric("本趟省電", f"{cmp['saved_kwh']:.2f} kWh")
    m[3].metric("本趟減碳", f"{cmp['saved_co2']:.2f} kg CO₂e")
    m[4].metric("再生煞車折抵", f"{cmp['rv']['total_regen']:.2f} kWh",
                help="煞車回饋電能覆蓋的空調用電（供電策略）")

    # 沿線功率剖面
    labels = [row["label"] for row in cmp["rv"]["rows"]]
    st.markdown("**沿線空調功率**")
    fig = go.Figure()
    fig.add_bar(x=labels, y=[row["power_kw"] for row in cmp["base"]["rows"]],
                name="傳統固定", marker_color="#c55a11")
    fig.add_bar(x=labels, y=[row["power_kw"] for row in cmp["rv"]["rows"]],
                name="RailVolt", marker_color="#538135")
    fig.update_layout(barmode="group", height=250, xaxis_tickangle=-50,
                      yaxis_title="平均空調功率 kW", margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=1.2))
    st.plotly_chart(fig, use_container_width=True)

    # 沿線車廂溫度（每站）
    st.markdown("**沿線車廂溫度（逐站模擬）**")
    figt = go.Figure()
    figt.add_hrect(y0=24.5, y1=27.0, fillcolor="#538135", opacity=0.08,
                   line_width=0, annotation_text="PMV 舒適帶", annotation_position="top left")
    figt.add_scatter(x=labels, y=[row["temp"] for row in cmp["base"]["rows"]],
                     name="傳統固定 24°C", line=dict(color="#c55a11"))
    figt.add_scatter(x=labels, y=[row["temp"] for row in cmp["rv"]["rows"]],
                     name="RailVolt", line=dict(color="#538135"))
    figt.update_layout(height=250, xaxis_tickangle=-50, yaxis_title="車廂溫度 °C",
                       margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.2))
    st.plotly_chart(figt, use_container_width=True)
    st.caption("傳統固定設定使車廂長期偏冷（浪費電）；RailVolt 讓溫度貼著舒適帶上緣走，"
               "省電同時避免過冷。溫度由車廂熱力模型逐分鐘模擬。")

    # 逐站逐區間明細
    tbl = []
    for b, rr in zip(cmp["base"]["rows"], cmp["rv"]["rows"]):
        if rr["kind"] == "stop":
            phase, note = "停站", "🚪 開門熱交換・乘降"
        elif rr["surge"]:
            phase, note = "行駛", "❄️ 航廈預冷啟動（人流領先指標）"
        else:
            phase, note = "行駛", "🚄 加速降載＋煞車再生回充"
        tbl.append({
            "位置／區間": rr["label"], "相位": phase,
            "分": str(rr["minutes"]), "載客/車": str(round(rr["occ"])),
            "設定°C": f"{rr['setpoint']:.1f}", "功率kW": f"{rr['power_kw']:.1f}",
            "本段kWh": f"{rr['kwh']:.2f}", "傳統kWh": f"{b['kwh']:.2f}",
            "再生kWh": f"{rr['regen']:.2f}", "空調策略": note})
    with st.expander("📋 展開逐站逐區間明細", expanded=True):
        st.table(pd.DataFrame(tbl).set_index("位置／區間"))
    st.caption("同一趟班次、相同載客下比較：傳統固定 24°C vs RailVolt 依載客/相位/"
               "航廈預測動態調節。物理參數為車型標稱量級，真實資料到手後以桃捷實測"
               "電壓電流校準。")

# ---------- Tab 1 即時監控 ----------
with tab1:
    t_sel = st.slider("模擬日時刻", min_value=dt.time(6, 0),
                      max_value=dt.time(22, 58), value=dt.time(11, 0),
                      step=dt.timedelta(minutes=2),
                      help="拖動觀察一天營運中任一時刻的列車狀態")
    idx = min(sim.N - 1, t_sel.hour * 60 + t_sel.minute - sim.T_START)
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

# ---------- Tab 6 空調最佳化 ----------
with tab6:
    st.subheader("空調節能最佳化：手動調參數，或讓系統跑出最佳解")
    KEYMAP = {"base_sp": "o_base", "load_gain": "o_gain", "empty_sp": "o_empty",
              "precool_thresh": "o_pth", "precool_sp": "o_psp",
              "brake_gain": "o_bg", "accel_relief": "o_ar"}
    for pk, kk in KEYMAP.items():
        st.session_state.setdefault(kk, float(sim.DEFAULT_POLICY[pk]))

    left, right = st.columns(2)
    with left:
        st.markdown("**① 手動調整控制參數（即時看結果）**")
        pol = dict(
            base_sp=st.slider("基準設定溫度 °C（滿載目標）", 25.5, 27.0,
                              key="o_base", step=0.1),
            load_gain=st.slider("載客敏感度（越大越隨人調溫）", 0.0, 3.5,
                                key="o_gain", step=0.1),
            empty_sp=st.slider("空車漂移設定 °C", 26.5, 28.0, key="o_empty", step=0.1),
            precool_thresh=st.slider("預冷觸發門檻（預測增量・人）", 20.0, 90.0,
                                     key="o_pth", step=1.0),
            precool_sp=st.slider("預冷目標 °C", 24.5, 26.0, key="o_psp", step=0.1),
            brake_gain=st.slider("煞車耦合倍數（1＝不用再生）", 1.0, 2.0,
                                 key="o_bg", step=0.05),
            accel_relief=st.slider("加速降載係數", 0.3, 1.0, key="o_ar", step=0.05),
        )
        r = sim.simulate(day, policy=pol)
        save = (base["kwh"] - r["kwh"]) / base["kwh"]
        mm = st.columns(3)
        mm[0].metric("本策略日耗電", f"{r['kwh']:.0f} kWh",
                     f"-{save:.1%}", delta_color="inverse")
        mm[1].metric("乘客舒適時間比", f"{r['comfort']:.0%}")
        mm[2].metric("再生折抵", f"{r['regen']:.1f} kWh")
        if r["comfort"] < 0.85:
            st.warning("⚠️ 舒適度偏低（車廂過熱或過冷），建議調整參數")

    with right:
        st.markdown("**② 自動搜尋最佳解**（隨機搜尋，物理模擬為評估器）")
        ctarget = st.slider("舒適度下限約束", 0.85, 1.0, 0.92, 0.01)
        if st.button("🔍 執行最佳化搜尋（約 8 秒）", type="primary"):
            with st.spinner("在參數空間搜尋能耗最低且滿足舒適約束的策略…"):
                st.session_state.opt = sim.optimize_policy(
                    day, comfort_min=ctarget, n_iter=180)
        if "opt" in st.session_state:
            b = st.session_state.opt["best"]
            if b:
                st.success(f"✅ 最佳解：省電 **{b['save']:.1%}**，舒適 {b['comfort']:.0%}")
                if st.button("套用最佳策略到左側滑桿"):
                    for pk, kk in KEYMAP.items():
                        st.session_state[kk] = float(b["policy"][pk])
                    st.rerun()
                tr = st.session_state.opt["trials"]
                figo = go.Figure()
                figo.add_scatter(
                    x=[t["comfort"] for t in tr if not t["feasible"]],
                    y=[t["save"] for t in tr if not t["feasible"]],
                    mode="markers", name="不符舒適", marker=dict(color="#cccccc", size=6))
                figo.add_scatter(
                    x=[t["comfort"] for t in tr if t["feasible"]],
                    y=[t["save"] for t in tr if t["feasible"]],
                    mode="markers", name="可行解", marker=dict(color="#538135", size=6))
                figo.add_scatter(x=[b["comfort"]], y=[b["save"]], mode="markers",
                                 name="最佳", marker=dict(color="#c00000", size=15,
                                                        symbol="star"))
                figo.update_layout(height=300, xaxis_title="舒適時間比",
                                   yaxis_title="節能率", margin=dict(t=10, b=10),
                                   legend=dict(orientation="h", y=1.25),
                                   yaxis_tickformat=".0%", xaxis_tickformat=".0%")
                st.plotly_chart(figo, use_container_width=True)
            else:
                st.warning("此約束下找不到可行解，請放寬舒適度下限再搜尋。")
    st.caption("最佳化目標：在固定 PMV 舒適標準（24.5–27°C）下最小化能耗；決策變數為左側 7 個"
               "控制參數。真實資料到手後評估器由模擬換成桃捷實測回測，即為 MPC 最佳化雛型。")

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

# ---------- Tab 5 牽引電力反推載客 ----------
with tab5:
    st.subheader("牽引電力反推列車載客（特色二・電力數據為 mock）")
    st.markdown(
        "原理：出站加速時 **牽引功率 P =(m·a + 阻力)·v**，量測 P、由速度求加速度 a，"
        "反解車重 m，扣除空車重即得乘客數。乘客僅占全車質量約 3–4%，單點量測噪聲"
        "會放大成大幅人數誤差，故對整段加速多點平均以壓低噪聲。*純物理反演，非黑箱 AI。*")
    inf = load_infer()
    sp = inf["sample"]
    cc = st.columns(4)
    cc[0].metric("反推 MAE", f"{inf['mae']:.1f} 人／列")
    cc[1].metric("R²（判定係數）", f"{inf['r2']:.3f}")
    cc[2].metric("驗證樣本數", f"{len(inf['true'])} 筆")
    cc[3].metric("量測噪聲設定", f"±{inf['noise']:.0%}")

    figi = go.Figure()
    lim = float(max(inf["true"].max(), inf["est"].max())) * 1.05
    figi.add_scatter(x=inf["true"], y=inf["est"], mode="markers",
                     marker=dict(color="#2e75b6", size=6, opacity=0.45),
                     name="出站事件")
    figi.add_scatter(x=[0, lim], y=[0, lim], mode="lines",
                     line=dict(dash="dash", color="gray"), name="完美反推")
    figi.update_layout(height=380, xaxis_title="實際載客（人/列）",
                       yaxis_title="牽引電力反推載客（人/列）",
                       margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(figi, use_container_width=True)

    st.info(
        f"範例｜真實 {sp['true_pax']:.0f} 人（車重 {sp['m_true']:.1f} 噸，空車 "
        f"{sp['tare']:.0f} 噸）→ 量測牽引功率 {sp['p_meas']:.0f} kW → 反解 "
        f"{sp['m_est']:.1f} 噸 → 反推 {sp['est_pax']:.0f} 人")
    st.caption("桃捷未釋出逐車廂載客數，本法用已釋出的電力（電壓×電流）自建載客標籤，"
               "餵給空調決策；正式版並以北捷『各車廂實測載重』交叉驗證。目前電力為 mock。")

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
