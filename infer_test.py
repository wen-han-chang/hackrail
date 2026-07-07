# -*- coding: utf-8 -*-
import simulation as s

# 溫度追蹤
veh = s.VEHICLES["直達車（Express，座位為主・2門）"]
cmp = s.trip_compare(dict(dep=8 * 60, kind="直達"), veh, "南下")
temps = [round(r["temp"], 1) for r in cmp["rv"]["rows"]]
print("RailVolt 逐區間車廂溫度：", temps)
btemps = [round(r["temp"], 1) for r in cmp["base"]["rows"]]
print("傳統逐區間車廂溫度：", btemps)

# 牽引電力反推載客
inf = s.infer_load_from_traction()
print(f"\n反推載客：樣本 {len(inf['true'])} 筆，MAE {inf['mae']:.1f} 人，R² {inf['r2']:.3f}")
sp = inf["sample"]
print(f"範例：真實 {sp['true_pax']:.0f} 人 (質量 {sp['m_true']:.1f}t) → "
      f"量測牽引功率 {sp['p_meas']:.0f} kW → 反推 {sp['m_est']:.1f}t → "
      f"{sp['est_pax']:.0f} 人")
