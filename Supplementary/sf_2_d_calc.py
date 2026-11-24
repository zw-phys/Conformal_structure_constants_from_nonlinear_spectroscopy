from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sf_2_d.csv'
SCAN_PARAMETERS={'projection':'full_parity','tag':'amplitude_fingerprint','L':12,'J2':0.05,'h':1.0825,'n':20,'cycles':32.0,'ramp':8.0,'steps':128,'phi1':0.0,'phi2':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'area':0.005},{'area':0.01},{'area':0.015},{'area':0.02},{'area':0.025},{'area':0.03}]
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
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS]).sort_values('area')
 signal_abs=np.hypot(pulse.numerator_real,pulse.numerator_imag)
 reference=signal_abs.iloc[0]*(pulse.area/pulse.area.iloc[0])**2
 return pd.DataFrame({'x':pulse.area,'y':signal_abs,'quadratic_reference':reference,'numerator_real':pulse.numerator_real,'numerator_imag':pulse.numerator_imag})
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
