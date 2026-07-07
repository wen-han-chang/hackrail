# -*- coding: utf-8 -*-
import simulation as s

veh = s.VEHICLES["直達車（Express，座位為主・2門）"]
svc = dict(dep=8 * 60, kind="直達")
cmp = s.trip_compare(svc, veh, "南下")
r = cmp["route"]
print(f"班次 08:00 直達 南下｜{r['n_stops']} 停靠｜{r['trip_km']:.1f} km｜"
      f"行駛 {r['arr'] - r['dep']} 分｜尖峰載客 {r['occ_peak']:.0f}/車")
print(f"傳統 {cmp['base']['total_kwh']:.1f} kWh｜RailVolt {cmp['rv']['total_kwh']:.1f} kWh"
      f"｜省 {cmp['saved_kwh']:.1f} kWh ({cmp['saved_pct']:.1%})｜"
      f"減碳 {cmp['saved_co2']:.2f} kg｜再生 {cmp['rv']['total_regen']:.1f} kWh")
print(f"路段數 {len(cmp['rv']['rows'])}，峰值功率 {cmp['rv']['peak_kw']:.1f} kW")
print("\n普通車對照：")
veh2 = s.VEHICLES["普通車（Commuter，站位為主・3門）"]
c2 = s.trip_compare(dict(dep=8 * 60 + 7, kind="普通"), veh2, "南下")
print(f"普通車 {c2['route']['n_stops']} 停靠｜傳統 {c2['base']['total_kwh']:.1f}"
      f"｜RailVolt {c2['rv']['total_kwh']:.1f}｜省 {c2['saved_pct']:.1%}")
print(f"\n班表共 {len(s.timetable())} 班次")
