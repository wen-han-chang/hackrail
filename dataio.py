# -*- coding: utf-8 -*-
"""RailVolt 真實資料接入層（loader 骨架）
競賽資料開放後，把檔案放進 data/ 並呼叫對應 load_* 函式即可；
未提供檔案時自動回傳與釋出資料「相同欄位結構」的模擬資料，讓管線先跑通。

釋出資料對照：
- 桃捷「列車資料」        -> load_train_telemetry()  空調/電壓電流/車門/動力
- 桃捷「一二航廈分時量」  -> load_airport_flow()      日期/小時/入出境人數
- 桃捷「北北桃空品氣候」  -> load_weather()           地區/測項/數值/時間
- 北捷「列車擁擠度」      -> load_crowding()          車次/車廂/實測載重人數
"""
import os

import numpy as np
import pandas as pd

import simulation as sim

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _has(fname):
    return os.path.exists(os.path.join(DATA_DIR, fname))


def load_train_telemetry(fname="train_telemetry.xlsx"):
    """列車遙測：時間、車廂、載客、車廂溫度、空調功率、牽引功率(電壓×電流)。
    真檔存在則讀 Excel；否則由模擬產生同欄位 DataFrame。"""
    cols = ["timestamp", "car", "occupancy", "cabin_temp_c",
            "ac_power_kw", "traction_power_kw", "door_open"]
    if _has(fname):
        df = pd.read_excel(os.path.join(DATA_DIR, fname))
        return df.reindex(columns=cols)
    day = sim.gen_train_day(42)
    rv = sim.simulate(day, "railvolt")
    rows = []
    for i in range(0, sim.N, 1):
        ph = day["route"][i][0]
        for c in range(sim.CARS):
            rows.append(dict(
                timestamp=int(sim.MIN[i]), car=c + 1,
                occupancy=float(day["occ"][i, c]),
                cabin_temp_c=round(float(rv["T"][i, c]), 2),
                ac_power_kw=round(float(rv["P"][i]) / sim.CARS, 2),
                traction_power_kw=np.nan, door_open=int(ph == "dwell")))
    return pd.DataFrame(rows, columns=cols)


def load_airport_flow(fname="airport_flow.xlsx"):
    """一二航廈旅客出入境分時量：日期、小時、入境人數、出境人數。"""
    cols = ["date", "hour", "arrival", "departure"]
    if _has(fname):
        return pd.read_excel(os.path.join(DATA_DIR, fname)).reindex(columns=cols)
    day = sim.gen_train_day(42)
    rows = []
    for h in range(6, 23):
        idx = min(sim.N - 1, h * 60 - sim.T_START)
        flow = float(day["airport"][idx])
        rows.append(dict(date="2026-08-15", hour=h,
                         arrival=round(flow * 0.5), departure=round(flow * 0.5)))
    return pd.DataFrame(rows, columns=cols)


def load_weather(fname="weather.csv"):
    """北北桃空品氣候：地區、測項、數值、時間（此處取氣溫）。"""
    cols = ["region", "item", "value", "timestamp"]
    if _has(fname):
        return pd.read_csv(os.path.join(DATA_DIR, fname)).reindex(columns=cols)
    day = sim.gen_train_day(42)
    rows = [dict(region="桃園", item="氣溫", value=round(float(day["tout"][i]), 1),
                 timestamp=int(sim.MIN[i])) for i in range(0, sim.N, 10)]
    return pd.DataFrame(rows, columns=cols)


def load_crowding(fname="crowding.csv"):
    """北捷列車擁擠度：車次、車廂、離站時間、各車廂實測載重人數（預測驗證真值）。"""
    cols = ["train_no", "car", "depart_time", "load_persons"]
    if _has(fname):
        return pd.read_csv(os.path.join(DATA_DIR, fname)).reindex(columns=cols)
    inf = sim.infer_load_from_traction(n_trips=20)
    rows = [dict(train_no=1000 + i // sim.CARS, car=i % sim.CARS + 1,
                 depart_time=int(360 + i), load_persons=round(v / sim.CARS))
            for i, v in enumerate(inf["true"])]
    return pd.DataFrame(rows, columns=cols)


def status():
    """回報各資料集目前是真檔或模擬。"""
    files = {"列車遙測": "train_telemetry.xlsx", "航廈分時量": "airport_flow.xlsx",
             "空品氣候": "weather.csv", "北捷擁擠度": "crowding.csv"}
    return {k: ("真實檔案" if _has(v) else "模擬") for k, v in files.items()}


if __name__ == "__main__":
    for name, df in [("列車遙測", load_train_telemetry()),
                     ("航廈分時量", load_airport_flow()),
                     ("空品氣候", load_weather()),
                     ("北捷擁擠度", load_crowding())]:
        print(f"[{name}] {df.shape} 欄位: {list(df.columns)}")
    print("狀態:", status())
