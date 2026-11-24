from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sf_2_f.csv'
LENGTH=16
REFERENCE_N=96
LOG_FLOOR=1e-12
SCAN_PARAMETERS={'tag':'q0_truncation','L':16,'cycles':32.0,'ramp':8.0,'area':0.02,'steps':128,'phi1':0.0,'phi2':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'J2':0.0,'h':1.0,'n':8},{'J2':0.0,'h':1.0,'n':16},{'J2':0.0,'h':1.0,'n':32},{'J2':0.0,'h':1.0,'n':64},{'J2':0.0,'h':1.0,'n':96},{'J2':0.05,'h':1.0825,'n':8},{'J2':0.05,'h':1.0825,'n':16},{'J2':0.05,'h':1.0825,'n':32},{'J2':0.05,'h':1.0825,'n':64},{'J2':0.05,'h':1.0825,'n':96}]
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
 scan=estimate_from_lockin(pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS]))
 rows=[]
 for J2,series,label in[(0.0,'ising','Ising'),(0.05,'nnn','NNN')]:
  pulse=scan.loc[scan.J2==J2].sort_values('n')
  reference=pulse.loc[pulse.n==REFERENCE_N,'C_imag'].item()
  for n,imaginary in pulse.loc[pulse.n<REFERENCE_N,['n','C_imag']].itertuples(index=False,name=None):
   rows.append((series,label,n,abs(imaginary-reference)+LOG_FLOOR,J2,LENGTH,imaginary,REFERENCE_N,reference,LOG_FLOOR))
 return pd.DataFrame(rows,columns=['series','label','x','y','J2','L','C_imag','reference_n','reference_C_imag','log_plot_floor'])
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
