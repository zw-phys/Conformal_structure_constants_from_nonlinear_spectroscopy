from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sf_1_b.csv'
BRANCHES=[(0.0,1.0,'ising','Ising, $h=1$'),(0.05,1.0825,'detuned','NNN, $h=1.0825$'),(0.05,1.08385,'tuned','NNN, tuned $h=1.08385$')]
SCAN_PARAMETERS={'tag':'matched_main','cycles':96.0,'ramp':8.0,'area':0.02,'steps':128,'phi1':0.0,'phi2':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'L':6,'J2':0.0,'h':1.0,'n':8},{'L':8,'J2':0.0,'h':1.0,'n':20},{'L':10,'J2':0.0,'h':1.0,'n':56},{'L':12,'J2':0.0,'h':1.0,'n':64},{'L':14,'J2':0.0,'h':1.0,'n':64},{'L':16,'J2':0.0,'h':1.0,'n':64},{'L':18,'J2':0.0,'h':1.0,'n':64},{'L':20,'J2':0.0,'h':1.0,'n':64},{'L':22,'J2':0.0,'h':1.0,'n':64},{'L':24,'J2':0.0,'h':1.0,'n':64},{'L':26,'J2':0.0,'h':1.0,'n':64},{'L':6,'J2':0.05,'h':1.0825,'n':8},{'L':8,'J2':0.05,'h':1.0825,'n':20},{'L':10,'J2':0.05,'h':1.0825,'n':56},{'L':12,'J2':0.05,'h':1.0825,'n':64},{'L':14,'J2':0.05,'h':1.0825,'n':64},{'L':16,'J2':0.05,'h':1.0825,'n':64},{'L':18,'J2':0.05,'h':1.0825,'n':64},{'L':20,'J2':0.05,'h':1.0825,'n':64},{'L':22,'J2':0.05,'h':1.0825,'n':64},{'L':24,'J2':0.05,'h':1.0825,'n':64},{'L':26,'J2':0.05,'h':1.0825,'n':64},{'L':6,'J2':0.05,'h':1.08385,'n':8},{'L':8,'J2':0.05,'h':1.08385,'n':20},{'L':10,'J2':0.05,'h':1.08385,'n':56},{'L':12,'J2':0.05,'h':1.08385,'n':64},{'L':14,'J2':0.05,'h':1.08385,'n':64},{'L':16,'J2':0.05,'h':1.08385,'n':64},{'L':18,'J2':0.05,'h':1.08385,'n':64},{'L':20,'J2':0.05,'h':1.08385,'n':64},{'L':22,'J2':0.05,'h':1.08385,'n':64},{'L':24,'J2':0.05,'h':1.08385,'n':64},{'L':26,'J2':0.05,'h':1.08385,'n':64}]
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
 pulse=estimate_from_lockin(pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS]))
 rows=[]
 for J2,field,series,label in BRANCHES:
  model=pulse.loc[(pulse.J2==J2)&(pulse.h==field)]
  for L,C_imag in model[['L','C_imag']].itertuples(index=False,name=None):
   rows.append(('imaginary',series,label,L,C_imag,J2,field))
 return pd.DataFrame(rows,columns=['kind','series','label','x','y','J2','h'])
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
