from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sf_2_c.csv'
SCAN_PARAMETERS={'projection':'full_parity','tag':'apodization','L':12,'J2':0.05,'h':1.0825,'n':20,'cycles':32.0,'ramp':8.0,'area':0.02,'steps':128,'phi1':0.0,'phi2':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0}
SCAN_POINTS=[{'gamma_units':0.0},{'gamma_units':0.25},{'gamma_units':0.5},{'gamma_units':1.0}]
def estimate_from_lockin(data,numerator='numerator',denominator='denominator'):
 signal=data[numerator+'_real'].to_numpy()+1j*data[numerator+'_imag'].to_numpy()
 reference=data[denominator+'_real'].to_numpy()+1j*data[denominator+'_imag'].to_numpy()
 if np.any(reference==0)or not np.isfinite(signal/reference).all():
  raise ValueError('Invalid raw lock-in signal or reference')
 estimate=signal/reference
 data=data.copy()
 data['C_real'],data['C_imag']=(estimate.real,estimate.imag)
 if'C_direct'in data:
  data['error_real']=estimate.real-data.C_direct
  data['error_complex']=abs(estimate-data.C_direct)
 return data
def calculate():
 window=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS]).sort_values('gamma')
 kernel_abs=np.hypot(window.denominator_real,window.denominator_imag)
 rectangular_abs=kernel_abs.loc[window.gamma==0].item()
 return pd.DataFrame({'x':window.gamma*window.T_measure/(2*np.pi),'y':kernel_abs/rectangular_abs,'gamma':window.gamma,'T_measure':window.T_measure,'denominator_abs':kernel_abs,'rectangular_denominator_abs':rectangular_abs})
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
