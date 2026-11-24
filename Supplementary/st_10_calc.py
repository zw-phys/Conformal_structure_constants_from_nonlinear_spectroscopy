from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_10.csv')
SCAN_PARAMETERS={'tag':'retuned_area_duration','L':12,'J2':0.05,'h':1.08385,'n':64,'ramp':8.0,'steps':128,'quadrature_stride':1,'phi1':0.0,'phi2':0.0,'envelope_scale':1.0,'phase_offset':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'cycles':32.0,'area':0.02},{'cycles':32.0,'area':0.01},{'cycles':32.0,'area':0.005},{'cycles':96.0,'area':0.02},{'cycles':96.0,'area':0.01},{'cycles':96.0,'area':0.005},{'cycles':192.0,'area':0.02},{'cycles':192.0,'area':0.01},{'cycles':192.0,'area':0.005},{'cycles':384.0,'area':0.02},{'cycles':384.0,'area':0.01},{'cycles':384.0,'area':0.005}]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.loc[pulse.tag=='retuned_area_duration'].sort_values(['cycles','area'],ascending=[True,False]).copy()
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 S_1=pulse.denominator_real.to_numpy()+1j*pulse.denominator_imag.to_numpy()
 C=S_pc/S_1
 pulse['C_real']=C.real
 pulse['C_imag']=C.imag
 pulse['real_error']=C.real-pulse.C_direct
 result=pulse[['cycles','area','C_real','C_imag','real_error']]
 return result
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
