# -*- coding: utf-8 -*-
"""RailVolt MVP — 模擬資料引擎與物理模型
產生符合競賽釋出資料欄位結構的模擬資料（列車遙測、航廈人流、氣溫），
並以車廂熱力模型比較「傳統固定設定」與「RailVolt 預測式策略」的能耗。
真實資料到手後，只需替換 gen_env / gen_train_day 的資料來源。
"""
import numpy as np

# ---- 時間軸：06:00–23:00，1 分鐘一步 ----
T_START, T_END = 6 * 60, 23 * 60
MIN = np.arange(T_START, T_END)
N = len(MIN)

# ---- 路線（機場線代表站）----
STATIONS = ["A1 台北車站", "A3 新北產業園區", "A8 長庚醫院",
            "A12 機場第一航廈", "A13 機場第二航廈", "A18 高鐵桃園站", "A21 環北"]
LEG_MIN = [8, 7, 9, 3, 8, 6]          # 站間行駛分鐘數
AIRPORT_ST = {3, 4}                    # A12 / A13

# ---- 車廂物理參數（700T 級通勤車廂量級，文獻參數，待實測校準）----
CARS = 4
CAR_CAP = 230          # 每車廂滿載人數
C_TH = 2.0e6           # J/K 車廂等效熱容
UA = 380.0             # W/K 車體傳導
UA_DOOR = 2500.0       # W/K 停站開門之額外熱交換
Q_PERSON = 100.0       # W/人 乘客體熱
AC_MAX = 40000.0       # W 每車廂最大冷房能力
COP = 3.0              # 空調能效比
T_INIT = 26.0
CO2_FACTOR = 0.474     # kgCO2e/kWh（經濟部能源署公告量級）
CAR_W = np.array([0.30, 0.26, 0.21, 0.23])  # 各車廂載客分布權重


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def gen_env(rng):
    """外氣溫（°C）、航廈出境人流（人/時）、通勤強度（0–1）"""
    t = MIN / 60.0
    tout = 27 + 6 * _gauss(t, 14.5, 3.5) + rng.normal(0, 0.25, N)
    airport = 300 + 900 * _gauss(t, 9.5, 1.5) + 1100 * _gauss(t, 17.5, 2.0) \
        + rng.normal(0, 40, N)
    commuter = 0.15 + 1.0 * _gauss(t, 8.0, 1.0) + 0.9 * _gauss(t, 18.0, 1.2)
    return tout, np.clip(airport, 0, None), commuter


def build_route():
    """整日運行序列：每分鐘 (phase, 現站, 次站)；phase ∈ dwell/accel/cruise/brake"""
    seq, s, d = [], 0, 1
    while len(seq) < N:
        seq.append(("dwell", s, s))
        if s + d < 0 or s + d > len(STATIONS) - 1:
            d = -d
        nxt = s + d
        leg = LEG_MIN[min(s, nxt)]
        for k in range(leg):
            ph = "accel" if k == 0 else ("brake" if k == leg - 1 else "cruise")
            seq.append((ph, s, nxt))
        s = nxt
    return seq[:N]


def gen_train_day(seed):
    """產生一列車整日的逐分鐘資料：載客、路線相位、環境；並收集預測樣本"""
    rng = np.random.default_rng(seed)
    tout, airport, commuter = gen_env(rng)
    route = build_route()
    lag = 30  # 航廈人流領先車站進站約 30 分鐘

    occ = np.zeros((N, CARS))
    cur = np.zeros(CARS)
    events = []  # (minute_idx, station, boarding) 供預測模型用
    for i, (ph, s, nxt) in enumerate(route):
        if ph == "dwell":
            if s in AIRPORT_ST:
                j = max(0, i - lag)
                inflow = airport[j] / 60.0 * 0.55        # 人/分
            else:
                base = {0: 3.2, 5: 2.2, 6: 1.8}.get(s, 1.2)
                inflow = base * commuter[i] * 60 / 60
            board = max(0.0, inflow * 15 * rng.normal(1.0, 0.15))  # 班距 15 分
            alight_f = 0.55 if s in (0, 6) else (0.35 if s in AIRPORT_ST else 0.18)
            cur = cur * (1 - alight_f)
            add = board * CAR_W * rng.normal(1.0, 0.08, CARS)
            cur = np.clip(cur + add, 0, CAR_CAP)
            events.append((i, s, float(board)))
        occ[i] = cur
    return dict(occ=occ, route=route, tout=tout, airport=airport,
                commuter=commuter, events=events)


def simulate(day, mode):
    """回傳逐分鐘：各車廂溫度、空調電力(kW)、累計 kWh、再生折抵 kWh、舒適比例"""
    occ, route, tout = day["occ"], day["route"], day["tout"]
    # 預測載客：30 分鐘航廈領先資訊 + 誤差（demo 用；正式版接預測模組輸出）
    rng = np.random.default_rng(7)
    pred = np.roll(occ, -10, axis=0) * rng.normal(1.0, 0.06, occ.shape)

    tin = np.full(CARS, T_INIT)
    T_hist = np.zeros((N, CARS))
    P_hist = np.zeros(N)              # 全列車空調電力 kW
    kwh = regen = 0.0
    comfort_ok = comfort_tot = 0
    for i in range(N):
        ph = route[i][0]
        ua = UA + (UA_DOOR if ph == "dwell" else 0.0)
        p_train = 0.0
        for c in range(CARS):
            dens = occ[i, c] / CAR_CAP
            if mode == "baseline":
                sp = 24.0
            else:
                sp = 26.3 - 2.0 * dens                # PMV 帶內隨載客調整
                if occ[i, c] < 5:
                    sp = 27.0                          # 空車讓溫度漂移
                if pred[i, c] - occ[i, c] > 40:
                    sp = min(sp, 25.0)                 # 預冷啟動
            q = float(np.clip(22000 * (tin[c] - sp), 0, AC_MAX))
            if mode == "railvolt":
                if ph == "brake":
                    q = min(q * 1.6 + (4000 if q > 0 else 0), AC_MAX)
                elif ph == "accel":
                    q = q * 0.45                       # 加速時壓縮機讓路
            elec = q / COP / 1000 / 60                 # kWh / 分
            kwh += elec
            if mode == "railvolt" and ph == "brake":
                regen += elec * 0.45                   # 再生電能覆蓋比例（試算）
            p_train += q / COP / 1000
            tin[c] += 60.0 / C_TH * (occ[i, c] * Q_PERSON + ua * (tout[i] - tin[c]) - q)
            if occ[i, c] > 10:                          # 有乘客時計舒適
                feel = tin[c] + 1.2 * dens             # 擁擠體感修正
                comfort_tot += 1
                comfort_ok += int(24.6 <= feel <= 27.0)  # 低於下限＝過冷不適
        T_hist[i] = tin
        P_hist[i] = p_train
    comfort = comfort_ok / max(comfort_tot, 1)
    return dict(T=T_hist, P=P_hist, kwh=kwh, regen=regen, comfort=comfort, pred=pred)


def fit_predictor(n_days=12, test_seed=42):
    """載客預測模組：航廈人流(滯後30分)+時段特徵 → 航廈站上車人數
    最小平方法（物理×統計，非黑箱），回傳測試日 MAE 與散點資料。"""
    def feats(day):
        X, y = [], []
        for i, s, board in day["events"]:
            if s not in AIRPORT_ST:
                continue
            t = MIN[i] / 1440.0
            ap = day["airport"][max(0, i - 30)] / 1000.0
            X.append([1.0, ap, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
            y.append(board)
        return np.array(X), np.array(y)

    Xs, ys = [], []
    for k in range(n_days):
        X, y = feats(gen_train_day(100 + k))
        Xs.append(X); ys.append(y)
    Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    Xte, yte = feats(gen_train_day(test_seed))
    yhat = np.clip(Xte @ w, 0, None)
    mae = float(np.mean(np.abs(yhat - yte)))
    return dict(w=w, y_true=yte, y_pred=yhat, mae=mae)


# ==================================================================
#  單一班次（實際班表）模擬 — 逐站逐區間的空調策略、供電與省電量
# ==================================================================
# 機場捷運全線車站（代碼、站名、累積里程 km）。里程為近似值，
# 正式版以 TDX「捷運車站間旅行時間」與「站間資料」取代。
LINE = [
    ("A1", "台北車站", 0.0), ("A2", "三重", 3.6), ("A3", "新北產業園區", 5.2),
    ("A4", "新莊副都心", 6.8), ("A5", "泰山", 8.6), ("A6", "泰山貴和", 10.0),
    ("A7", "體育大學", 13.2), ("A8", "長庚醫院", 15.1), ("A9", "林口", 17.2),
    ("A10", "山鼻", 19.7), ("A11", "坑口", 22.6), ("A12", "機場第一航廈", 25.9),
    ("A13", "機場第二航廈", 27.3), ("A14a", "機場旅館", 28.8), ("A15", "大園", 31.6),
    ("A16", "橫山", 33.6), ("A17", "領航", 35.2), ("A18", "高鐵桃園站", 37.0),
    ("A19", "桃園體育園區", 39.0), ("A20", "興南", 40.8), ("A21", "環北", 42.5),
]
EXPRESS_STOPS = [0, 2, 7, 11, 12, 17, 20]   # 直達車停靠：A1 A3 A8 A12 A13 A18 A21
AIRPORT_IDX = {11, 12}                        # A12 / A13

# 車型參數（機捷直達車／普通車量級；標稱值，待桃捷實測校準）
VEHICLES = {
    "直達車（Express，座位為主・2門）": dict(
        cars=4, cap=170, tare_t=140, ac_max=42000,
        c_th=2.2e6, ua=360, ua_door=1800, k_ctrl=22000),
    "普通車（Commuter，站位為主・3門）": dict(
        cars=4, cap=250, tare_t=132, ac_max=40000,
        c_th=2.0e6, ua=380, ua_door=2600, k_ctrl=22000),
}


def timetable():
    """整日班表：直達每 30 分、普通每 15 分（示意；正式版接 TDX 定期站別時刻表）"""
    svcs = []
    for h in range(6, 23):
        svcs += [dict(dep=h * 60, kind="直達"), dict(dep=h * 60 + 30, kind="直達")]
        for m in (7, 22, 37, 52):
            svcs.append(dict(dep=h * 60 + m, kind="普通"))
    return sorted(svcs, key=lambda s: s["dep"])


def _tout_at(minute, rng):
    return 27 + 6 * _gauss(minute / 60.0, 14.5, 3.5) + rng.normal(0, 0.2)


def _board_alight(idx, minute, rng):
    t = minute / 60.0
    peak = _gauss(t, 8, 1.2) + _gauss(t, 18, 1.4)
    if idx in AIRPORT_IDX:
        board, alight = 70 + 150 * (_gauss(t, 10, 2.5) + _gauss(t, 20, 2.5)), 0.35
    elif idx in (0, len(LINE) - 1):
        board, alight = 45 + 130 * peak, 0.55
    else:
        board, alight = 18 + 55 * peak, 0.20
    return max(0.0, board * rng.normal(1, 0.15)), alight


def build_trip(service, veh, direction, seed=0):
    """建立一趟班次的逐分鐘序列（載客、外氣、相位），與空調策略無關。"""
    rng = np.random.default_rng(seed or service["dep"] * 7 + 1)
    stops = EXPRESS_STOPS if service["kind"] == "直達" else list(range(len(LINE)))
    if direction == "北上":
        stops = stops[::-1]
    cars, cap = veh["cars"], veh["cap"]
    occ = np.zeros(cars)
    clock, leg, steps = service["dep"], 0, []
    for si, idx in enumerate(stops):
        board, alight = _board_alight(idx, clock, rng)
        occ = np.clip(occ * (1 - alight) + board * CAR_W * rng.normal(1, 0.08, cars),
                      0, cap)
        steps.append(dict(leg=leg, kind="stop", idx=idx, phase="dwell",
                          label=f"{LINE[idx][0]} {LINE[idx][1]}",
                          occ=occ.copy(), tout=_tout_at(clock, rng),
                          board=board, alight=alight, surge=False))
        leg += 1; clock += 1
        if si < len(stops) - 1:
            nxt = stops[si + 1]
            dist = abs(LINE[nxt][2] - LINE[idx][2])
            tmin = max(2, round(dist / (90 if service["kind"] == "直達" else 60) * 60))
            phases = ["accel"] + ["cruise"] * (tmin - 2) + ["brake"]
            nb, _ = _board_alight(nxt, clock + tmin, rng)
            surge = nxt in AIRPORT_IDX and nb > 120
            for ph in phases:
                steps.append(dict(leg=leg, kind="seg", idx=idx, phase=ph,
                                  label=f"{LINE[idx][0]}→{LINE[nxt][0]}",
                                  occ=occ.copy(), tout=_tout_at(clock, rng),
                                  surge=surge, dist=dist))
                clock += 1
            leg += 1
    trip_km = abs(LINE[stops[-1]][2] - LINE[stops[0]][2])
    return dict(steps=steps, service=service, veh=veh, direction=direction,
                n_stops=len(stops), trip_km=trip_km, dep=service["dep"],
                arr=clock, occ_peak=float(max(s["occ"].max() for s in steps)))


def _car_minute(tin, occ_c, cap, tout, phase, mode, veh, surge):
    dens = occ_c / cap
    if mode == "baseline":
        sp = 24.0
    else:
        sp = 26.3 - 2.0 * dens
        if occ_c < 5:
            sp = 27.0
        if surge:
            sp = min(sp, 25.0)
    ua = veh["ua"] + (veh["ua_door"] if phase == "dwell" else 0.0)
    q = float(np.clip(veh["k_ctrl"] * (tin - sp), 0, veh["ac_max"]))
    if mode == "railvolt":
        if phase == "brake":
            q = min(q * 1.6 + (4000 if q > 0 else 0), veh["ac_max"])
        elif phase == "accel":
            q = q * 0.45
    elec = q / COP / 1000 / 60
    regen = elec * 0.45 if (mode == "railvolt" and phase == "brake") else 0.0
    tin_new = tin + 60.0 / veh["c_th"] * (occ_c * Q_PERSON + ua * (tout - tin) - q)
    return tin_new, elec, regen, q / COP / 1000, sp


def run_trip(route, mode):
    """在給定路線上跑一種空調策略，回傳逐區間彙整與全趟總計。"""
    veh, cars, cap = route["veh"], route["veh"]["cars"], route["veh"]["cap"]
    tin = np.full(cars, T_INIT)
    legs, total, totregen, peak = {}, 0.0, 0.0, 0.0
    for s in route["steps"]:
        m_kwh = m_regen = m_pw = 0.0
        sps = []
        for c in range(cars):
            tin[c], elec, regen, pw, sp = _car_minute(
                tin[c], s["occ"][c], cap, s["tout"], s["phase"], mode, veh, s["surge"])
            m_kwh += elec; m_regen += regen; m_pw += pw; sps.append(sp)
        peak = max(peak, m_pw); total += m_kwh; totregen += m_regen
        a = legs.setdefault(s["leg"], dict(
            kind=s["kind"], label=s["label"], phase=s["phase"], surge=s["surge"],
            minutes=0, kwh=0.0, regen=0.0, pw_sum=0.0, sp_sum=0.0,
            occ=float(s["occ"].mean())))
        a["minutes"] += 1; a["kwh"] += m_kwh; a["regen"] += m_regen
        a["pw_sum"] += m_pw; a["sp_sum"] += float(np.mean(sps))
    rows = []
    for lid in sorted(legs):
        a = legs[lid]
        rows.append(dict(kind=a["kind"], label=a["label"], phase=a["phase"],
                         surge=a["surge"], minutes=a["minutes"], kwh=a["kwh"],
                         regen=a["regen"], power_kw=a["pw_sum"] / a["minutes"],
                         setpoint=a["sp_sum"] / a["minutes"], occ=a["occ"]))
    return dict(rows=rows, total_kwh=total, total_regen=totregen, peak_kw=peak)


def trip_compare(service, veh, direction, seed=0):
    """同一趟班次上比較傳統固定 24°C 與 RailVolt 策略。"""
    route = build_trip(service, veh, direction, seed)
    base, rv = run_trip(route, "baseline"), run_trip(route, "railvolt")
    saved = base["total_kwh"] - rv["total_kwh"]
    return dict(route=route, base=base, rv=rv, saved_kwh=saved,
                saved_pct=saved / base["total_kwh"] if base["total_kwh"] else 0.0,
                saved_co2=saved * CO2_FACTOR)
