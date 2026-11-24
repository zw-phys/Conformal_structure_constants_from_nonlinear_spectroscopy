from pathlib import Path
import numpy as np
import pandas as pd
from pulse_calc import build_model,forward_pulse
OUTPUT=Path(__file__).with_name('st_5.csv')
LENGTH=12
J2=0.05
FIELD=1.08385
STATES_PER_PARITY=64
CYCLES=96
RAMP=8
AREA=0.02
STEPS=128
BOUNDS={'intermediate':0.001,'output':0.0005}
DRIFT_UNITS=[0.0,-0.001,0.001,-0.0005,0.0005]
def calculate():
 model=build_model(LENGTH,J2,FIELD,STATES_PER_PARITY)
 pulse=dict(cycles=CYCLES,ramp=RAMP,area=AREA,steps=STEPS)
 nominal=forward_pulse(model,**pulse)
 reference=nominal['denominator']
 baseline=nominal['numerator']/reference
 resolution=model.gap_epsilon/CYCLES
 rows=[]
 for coordinate,bound in BOUNDS.items():
  changes=[]
  offsets=[u for u in DRIFT_UNITS if abs(u)<=bound]
  for u in offsets:
   if coordinate=='intermediate':
    drift1,drift2=(u*resolution,-u*resolution)
   else:
    drift1,drift2=(0.0,u*resolution)
   measured=forward_pulse(model,**pulse,drift1=drift1,drift2=drift2)
   changes.append(abs(measured['numerator']/reference-baseline))
  rows.append((coordinate,bound,max((abs(u)for u in offsets))*resolution,max(changes)))
 return pd.DataFrame(rows,columns=['coordinate','bound_resolution_units','bound_J1_over_hbar','max_complex_change'])
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
