from pathlib import Path
import numpy as np
import pandas as pd
import tci_model_calc as tci
LENGTHS=range(6,29,2)
G=0.856
STATES=8
CALIBRATION_LENGTH=18
OUTPUT=Path(__file__).with_name('sf_5_a.csv')
def calculate():
 alpha=tci.probe_calibration(CALIBRATION_LENGTH,g=G,states=STATES)
 rows=[]
 for length in LENGTHS:
  p=tci.probe_transitions(length,g=G,states=STATES)
  primary=p['m_z_sigma']+alpha*p['m_xz_sigma']
  secondary=p['m_z_sigma_prime']+alpha*p['m_xz_sigma_prime']
  if primary==0 or p['m_z_sigma']==0:
   raise ValueError('A primary normalization transition vanishes')
  rows.append((length,abs(p['m_z_sigma_prime']/p['m_z_sigma']),abs(secondary/primary)))
 return pd.DataFrame(rows,columns=['L','bare_secondary_ratio','fixed_secondary_ratio'])
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
