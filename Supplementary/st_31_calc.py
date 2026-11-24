from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from st_29_calc import build_low_energy_model
from st_30_calc import precompute_kubo,chi_two_tone,laurent_design_row,solve_lstsq
OUTPUT=Path(__file__).with_name('st_31.csv')
LENGTHS=(12,14)
STATE_COUNTS=(8,24,48)
def recalculate_truncation(lengths=(12,14),state_counts=(8,24,48)):
 offsets=(-6.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,6.0)
 eta,degree=(0.005,2)
 rows=[]
 for length in lengths:
  for states in state_counts:
   model=build_low_energy_model(length,1.0,1.0,states,0,512,1e-10)
   precomputed=precompute_kubo(model)
   coordinates=[(x*eta,y*eta)for x in offsets for y in offsets]
   design=np.array([laurent_design_row(x,y,eta,degree)for x,y in coordinates])
   values=[]
   for x,y in coordinates:
    first=model.gap_sigma+x
    second=model.gap_epsilon+y-first
    values.append(chi_two_tone(precomputed,precomputed,first,second,eta))
   coefficients,_=solve_lstsq(design,np.array(values))
   coefficient=coefficients[0]/(model.M_epsilon*model.M_sigma**2)
   rows.append((length,states,coefficient.real,coefficient.imag,model.energies[model.n_even-1]))
 return pd.DataFrame(rows,columns=['L','states_per_parity','C_real','C_imag','cutoff_even'])
def calculate(lengths=LENGTHS,state_counts=STATE_COUNTS):
 return recalculate_truncation(lengths,state_counts)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
