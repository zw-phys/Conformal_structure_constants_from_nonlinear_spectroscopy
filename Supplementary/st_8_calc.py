from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_8.csv')
J2=0.05
LENGTHS=[12,16]
SCAN_PARAMETERS={'tag':'q0_truncation','J2':0.05,'h':1.0825,'cycles':32.0,'ramp':8.0,'area':0.02,'steps':128,'quadrature_stride':1,'phi1':0.0,'phi2':0.0,'envelope_scale':1.0,'phase_offset':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'L':12,'n':8},{'L':12,'n':16},{'L':12,'n':32},{'L':12,'n':64},{'L':12,'n':96},{'L':16,'n':8},{'L':16,'n':16},{'L':16,'n':32},{'L':16,'n':64},{'L':16,'n':96}]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.loc[(pulse.tag=='q0_truncation')&(pulse.J2==J2)&pulse.L.isin(LENGTHS)].sort_values(['L','n']).copy()
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 S_1=pulse.denominator_real.to_numpy()+1j*pulse.denominator_imag.to_numpy()
 C=S_pc/S_1
 pulse['C_real']=C.real
 pulse['C_imag']=C.imag
 result=pulse[['L','n','spectral_cutoff_even','spectral_cutoff_odd','C_real','C_imag']]
 return result
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
