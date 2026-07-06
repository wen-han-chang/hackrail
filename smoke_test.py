# -*- coding: utf-8 -*-
import simulation as s

d = s.gen_train_day(42)
b = s.simulate(d, "baseline")
r = s.simulate(d, "railvolt")
p = s.fit_predictor()
print(f"baseline {b['kwh']:.1f} kWh/day, railvolt {r['kwh']:.1f} kWh/day, "
      f"saving {(b['kwh'] - r['kwh']) / b['kwh']:.1%}")
print(f"comfort base {b['comfort']:.0%} -> rv {r['comfort']:.0%}, "
      f"regen {r['regen']:.1f} kWh, predictor MAE {p['mae']:.1f} persons")
print(f"occ max {d['occ'].max():.0f}, tin range "
      f"{r['T'].min():.1f}-{r['T'].max():.1f} degC, peak power {b['P'].max():.1f} kW")
