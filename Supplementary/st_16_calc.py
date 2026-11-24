from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_16.csv')
TAG='amplitude_fingerprint'
SCAN_PARAMETERS={'projection':'full_parity','tag':'amplitude_fingerprint','L':12,'J2':0.05,'h':1.0825,'n':20,'cycles':32.0,'ramp':8.0,'steps':128,'quadrature_stride':1,'phi1':0.0,'phi2':0.0,'envelope_scale':1.0,'phase_offset':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'area':0.005},{'area':0.01},{'area':0.015},{'area':0.02},{'area':0.025},{'area':0.03}]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.loc[pulse.tag==TAG].sort_values('area').copy()
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 S_1=pulse.denominator_real.to_numpy()+1j*pulse.denominator_imag.to_numpy()
 C=S_pc/S_1
 pulse['C_real']=C.real
 pulse['C_imag']=C.imag
 pulse['S_pc_abs']=np.abs(S_pc)
 result=pulse[['area','S_pc_abs','C_real','C_imag']]
 return result
def fingerprint_slope(table=None):
 table=calculate()if table is None else table
 x=np.log(table.area.to_numpy())
 y=np.log(table.S_pc_abs.to_numpy())
 design=np.column_stack([np.ones(len(x)),x])
 norms=np.linalg.norm(design,axis=0)
 coefficients=np.linalg.lstsq(design/norms,y,rcond=None)[0]/norms
 residual=y-design@coefficients
 return{'slope':float(coefficients[1]),'rms':float(np.sqrt(np.mean(residual**2)))}
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
