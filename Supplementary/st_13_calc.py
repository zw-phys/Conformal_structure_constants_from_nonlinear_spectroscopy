from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_13.csv')
TAG='apodization'
SCAN_PARAMETERS={'projection':'full_parity','tag':'apodization','L':12,'J2':0.05,'h':1.0825,'n':20,'cycles':32.0,'ramp':8.0,'area':0.02,'steps':128,'quadrature_stride':1,'phi1':0.0,'phi2':0.0,'envelope_scale':1.0,'phase_offset':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0}
SCAN_POINTS=[{'gamma_units':0.0},{'gamma_units':0.25},{'gamma_units':0.5},{'gamma_units':1.0}]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.loc[pulse.tag==TAG].sort_values('gamma').copy()
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 S_1=pulse.denominator_real.to_numpy()+1j*pulse.denominator_imag.to_numpy()
 C=S_pc/S_1
 pulse['C_real']=C.real
 pulse['C_imag']=C.imag
 pulse['gamma_resolution_units']=pulse.gamma/pulse.resolution
 pulse['S_pc_abs']=np.abs(S_pc)
 pulse['S_1_abs']=np.abs(S_1)
 result=pulse[['gamma_resolution_units','S_pc_abs','S_1_abs','C_real','C_imag']]
 return result
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
