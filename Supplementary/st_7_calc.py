from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_7.csv')
MODELS=[(powers,lower)for lower in(6,8,10,12,14)for powers in((1,),(2,),(3,),(2,4))]
MAX_LENGTH=26
SCAN_PARAMETERS={'tag':'matched_main','J2':0.05,'h':1.08385,'cycles':96.0,'ramp':8.0,'area':0.02,'steps':128,'quadrature_stride':1,'phi1':0.0,'phi2':0.0,'envelope_scale':1.0,'phase_offset':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'L':6,'n':8},{'L':8,'n':20},{'L':10,'n':56},{'L':12,'n':64},{'L':14,'n':64},{'L':16,'n':64},{'L':18,'n':64},{'L':20,'n':64},{'L':22,'n':64},{'L':24,'n':64},{'L':26,'n':64}]
def calculate():
 data=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 C=(data.numerator_real.to_numpy()+1j*data.numerator_imag.to_numpy())/(data.denominator_real.to_numpy()+1j*data.denominator_imag.to_numpy())
 data['C_real']=C.real
 data['C_imag']=C.imag
 rows=[]
 for powers,minimum_length in MODELS:
  window=data.loc[(data.L>=minimum_length)&(data.L<=MAX_LENGTH)]
  L=window.L.to_numpy(dtype=float)
  C_static=window.C_direct.to_numpy()
  C_pulse=window.C_real.to_numpy()+1j*window.C_imag.to_numpy()
  design=np.column_stack([np.ones(len(L)),*(L**(-p)for p in powers)])
  scale=np.linalg.norm(design,axis=0)
  scaled_design=design/scale
  alpha=np.linalg.lstsq(scaled_design,C_static,rcond=None)[0]/scale
  beta=np.linalg.lstsq(scaled_design,C_pulse,rcond=None)[0]/scale
  rows.append(('+'.join(map(str,powers)),minimum_length,alpha[0],np.max(np.abs(C_static-design@alpha)),beta[0].real,beta[0].imag,beta[0].real-alpha[0],np.max(np.abs(C_pulse-design@beta)),len(L)-design.shape[1]))
 result=pd.DataFrame(rows,columns=['powers','L_min','C_infinity','static_max_residual','pulse_intercept_real','pulse_intercept_imag','delta_0','pulse_max_complex_residual','dof'])
 return result
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
