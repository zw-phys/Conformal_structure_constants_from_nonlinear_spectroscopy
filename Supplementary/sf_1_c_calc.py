from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sf_1_c.csv'
AREAS=[0.02,0.01,0.005]
SCAN_PARAMETERS={'tag':'retuned_area_duration','L':12,'J2':0.05,'h':1.08385,'n':64,'ramp':8.0,'steps':128,'phi1':0.0,'phi2':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'cycles':32.0,'area':0.02},{'cycles':32.0,'area':0.01},{'cycles':32.0,'area':0.005},{'cycles':96.0,'area':0.02},{'cycles':96.0,'area':0.01},{'cycles':96.0,'area':0.005},{'cycles':192.0,'area':0.02},{'cycles':192.0,'area':0.01},{'cycles':192.0,'area':0.005},{'cycles':384.0,'area':0.02},{'cycles':384.0,'area':0.01},{'cycles':384.0,'area':0.005}]
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
 for area in AREAS:
  for cycles,C_imag,L in scan.loc[scan.area==area,['cycles','C_imag','L']].itertuples(index=False,name=None):
   rows.append((str(area),f'area = {area}',cycles,C_imag,area,L))
 return pd.DataFrame(rows,columns=['series','label','x','y','area','L'])
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
