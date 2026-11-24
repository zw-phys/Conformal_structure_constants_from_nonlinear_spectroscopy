from pathlib import Path
import numpy as np
import pandas as pd
from pulse_calc import build_ising_model,ising_make_time_grid,ising_phase_cycled_cross,ising_lockin_full
OUTPUT=Path(__file__).with_name('st_14.csv')
LENGTH=12
J=1.0
FIELD=1.0
STATES_PER_PARITY=12
DENSE_CUTOFF=1024
EIGEN_TOLERANCE=1e-10
CYCLES=48
RAMP_CYCLES=8
STEPS_PER_CYCLE=64
AMPLITUDES=[0.0005,0.00075,0.001,0.00125,0.0015]
def calculate():
 model=build_ising_model(LENGTH,J,FIELD,STATES_PER_PARITY,DENSE_CUTOFF,EIGEN_TOLERANCE)
 grid=ising_make_time_grid(model.gap_epsilon,CYCLES,RAMP_CYCLES,STEPS_PER_CYCLE)
 rows=[]
 for amplitude in AMPLITUDES:
  signal=ising_phase_cycled_cross(ising_lockin_full,model,grid,amplitude,amplitude,0.0,0.0)
  rows.append((amplitude,abs(signal),abs(signal)/amplitude**2))
 return pd.DataFrame(rows,columns=['lambda','S_pc_abs','S_pc_abs_over_lambda_squared'])
def fingerprint_slope(table=None):
 table=calculate()if table is None else table
 x=np.log(table['lambda'].to_numpy())
 y=np.log(table.S_pc_abs.to_numpy())
 design=np.column_stack([np.ones(len(x)),x])
 norms=np.linalg.norm(design,axis=0)
 coefficients=np.linalg.lstsq(design/norms,y,rcond=None)[0]/norms
 residual=y-design@coefficients
 return{'slope':float(coefficients[1]),'rms':float(np.sqrt(np.mean(residual**2)))}
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
