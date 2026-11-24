from pathlib import Path
import numpy as np
import pandas as pd
from pulse_calc import build_ising_model,ising_make_time_grid,ising_phase_cycled_cross,ising_lockin_full
OUTPUT=Path(__file__).with_name('st_15.csv')
LENGTH=12
J=1.0
FIELD=1.0
STATES_PER_PARITY=12
DENSE_CUTOFF=1024
EIGEN_TOLERANCE=1e-10
CYCLES=48
RAMP_CYCLES=8
STEPS_PER_CYCLE=64
AMPLITUDE=0.0015
PHASE_SUMS=np.arange(5)*np.pi/4
def calculate():
 model=build_ising_model(LENGTH,J,FIELD,STATES_PER_PARITY,DENSE_CUTOFF,EIGEN_TOLERANCE)
 grid=ising_make_time_grid(model.gap_epsilon,CYCLES,RAMP_CYCLES,STEPS_PER_CYCLE)
 rows=[]
 for phase in PHASE_SUMS:
  signal=ising_phase_cycled_cross(ising_lockin_full,model,grid,AMPLITUDE,AMPLITUDE,phase,0.0)
  rows.append((phase,abs(signal),np.angle(signal)))
 return pd.DataFrame(rows,columns=['phi_sum_rad','S_pc_abs','arg_S_pc_rad'])
def fingerprint_slope(table=None):
 table=calculate()if table is None else table
 x=table.phi_sum_rad.to_numpy()
 y=np.unwrap(table.arg_S_pc_rad.to_numpy())
 design=np.column_stack([np.ones(len(x)),x])
 norms=np.linalg.norm(design,axis=0)
 coefficients=np.linalg.lstsq(design/norms,y,rcond=None)[0]/norms
 residual=y-design@coefficients
 return{'slope':float(coefficients[1]),'rms':float(np.sqrt(np.mean(residual**2)))}
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
