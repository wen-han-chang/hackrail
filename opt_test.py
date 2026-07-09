# -*- coding: utf-8 -*-
import time
import simulation as s

day = s.gen_train_day(42)
base = s.simulate(day, "baseline")
rv = s.simulate(day, "railvolt")
print(f"baseline {base['kwh']:.1f} | default railvolt {rv['kwh']:.1f} "
      f"({(base['kwh']-rv['kwh'])/base['kwh']:.1%}), comfort {rv['comfort']:.0%}")

t0 = time.time()
opt = s.optimize_policy(day, comfort_min=0.90, n_iter=200)
dt = time.time() - t0
b = opt["best"]
print(f"\n最佳化 (200 次, {dt:.1f}s)：省 {b['save']:.1%}, 舒適 {b['comfort']:.0%}")
print("最佳策略：")
for k, v in b["policy"].items():
    print(f"  {k}: {v:.2f}")
feas = sum(t["feasible"] for t in opt["trials"])
print(f"可行解 {feas}/{len(opt['trials'])}")
