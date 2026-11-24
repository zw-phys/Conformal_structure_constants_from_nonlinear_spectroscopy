from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sf_2_b.csv'
SCAN_PARAMETERS={'projection':'full_parity','L':12,'J2':0.05,'h':1.0825,'n':20,'cycles':32.0,'ramp':8.0,'area':0.02,'steps':128,'phi1':0.0,'phi2':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'tag':'detuning_intermediate','intermediate_detuning_units':-1.0,'output_detuning_units':0.0},{'tag':'detuning_intermediate','intermediate_detuning_units':-0.5,'output_detuning_units':0.0},{'tag':'detuning_intermediate','intermediate_detuning_units':-0.25,'output_detuning_units':0.0},{'tag':'detuning_intermediate','intermediate_detuning_units':-0.1,'output_detuning_units':0.0},{'tag':'detuning_intermediate','intermediate_detuning_units':0.0,'output_detuning_units':0.0},{'tag':'detuning_intermediate','intermediate_detuning_units':0.1,'output_detuning_units':0.0},{'tag':'detuning_intermediate','intermediate_detuning_units':0.25,'output_detuning_units':0.0},{'tag':'detuning_intermediate','intermediate_detuning_units':0.5,'output_detuning_units':0.0},{'tag':'detuning_intermediate','intermediate_detuning_units':1.0,'output_detuning_units':0.0},{'tag':'detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':-1.0},{'tag':'detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':-0.5},{'tag':'detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':-0.25},{'tag':'detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':-0.1},{'tag':'detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':0.0},{'tag':'detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':0.1},{'tag':'detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':0.25},{'tag':'detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':0.5},{'tag':'detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':1.0}]
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
 curves=[]
 for series in('intermediate','output'):
  points=scan.loc[scan.tag=='detuning_'+series].copy()
  points['x']=points[series+'_detuning']/points.resolution
  zero=points.loc[points.x.abs().idxmin()]
  points['series']=series
  points['label']=series
  points['baseline_C_real']=zero.C_real
  points['y']=points.C_imag
  curves.append(points)
 result=pd.concat(curves,ignore_index=True)
 return result[['series','label','x','resolution','baseline_C_real','C_real','C_imag','y']]
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
