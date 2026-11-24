from pathlib import Path
import numpy as np
import pandas as pd
import tci_model_calc as tci
LENGTH=14
G=0.856
STATES=96
CYCLES=(12,24,48,96,128)
RAMP=6
AREA=0.005
STEPS=128
OUTPUT=Path(__file__).with_name('st_25.csv')
def calculate():
 energy,spin,readout,static=tci.build_model(LENGTH,G,STATES)
 rows=[]
 for cycles in CYCLES:
  pulse=tci.finite_pulse(energy,spin,readout,static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'],steps_per_cycle=STEPS,cycles=cycles,ramp_cycles=RAMP,area=AREA)
  C=complex(pulse['C_real'],pulse['C_imag'])
  rows.append((cycles,C.real,C.imag,abs(C-static['C_direct'])))
 return pd.DataFrame(rows,columns=['cycles','C_real','C_imag','complex_error'])
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
