from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_17.csv')
TAG='phase_fingerprint'
SCAN_PARAMETERS={'projection':'full_parity','tag':'phase_fingerprint','L':12,'J2':0.05,'h':1.0825,'n':20,'cycles':32.0,'ramp':8.0,'area':0.02,'steps':128,'quadrature_stride':1,'envelope_scale':1.0,'phase_offset':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
PHASE_SUMS=np.arange(5)*np.pi/4
SCAN_POINTS=[dict(phi1=0.37*phase,phi2=0.63*phase)for phase in PHASE_SUMS]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.loc[pulse.tag==TAG].copy()
 pulse['phi_sum_rad']=pulse.phi1+pulse.phi2
 pulse=pulse.sort_values('phi_sum_rad')
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 result=pd.DataFrame({'phi_sum_rad':pulse.phi_sum_rad,'S_pc_abs':np.abs(S_pc),'arg_S_pc_rad':np.angle(S_pc)})
 return result
def fingerprint_slope(table=None):
 table=calculate()if table is None else table
 x=table.phi_sum_rad.to_numpy()
 y=np.unwrap(table.arg_S_pc_rad.to_numpy())
 design=np.column_stack([np.ones(len(x)),x])
 norms=np.linalg.norm(design,axis=0)
 coefficients=np.linalg.lstsq(design/norms,y,rcond=None)[0]/norms
 residual=y-design@coefficients
 return{'slope':float(coefficients[1]),'rms':float(np.sqrt(np.mean(residual**2)))}
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
