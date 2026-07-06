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
