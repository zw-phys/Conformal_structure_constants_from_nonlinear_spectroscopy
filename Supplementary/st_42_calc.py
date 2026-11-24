from pathlib import Path
import numpy as np
import pandas as pd
import tci_model_calc as tci
CALIBRATION_LENGTH=18
G=0.856
STATES=8
OPERATORS=('XZ','ZZZ')
OUTPUT=Path(__file__).with_name('st_42.csv')
def calculate():
 rows=[]
 for operator in OPERATORS:
  p=tci.probe_transitions(CALIBRATION_LENGTH,operator,G,STATES)
  alpha=-p['m_z_sigma_prime']/p['m_xz_sigma_prime']
  primary=p['m_z_sigma']+alpha*p['m_xz_sigma']
  if abs(primary)<1e-14:
   raise ValueError('The calibrated primary transition vanishes')
  direct=p['middle_z']/p['m_z_sigma']
  calibrated=(p['middle_z']+alpha*p['middle_aux'])/primary
  np.testing.assert_allclose(p['m_z_sigma_prime']+alpha*p['m_xz_sigma_prime'],0,rtol=0,atol=5e-15)
  rows.append((CALIBRATION_LENGTH,operator,p['m_xz_sigma'],p['m_xz_sigma_prime'],alpha,primary,direct,calibrated))
 return pd.DataFrame(rows,columns=['L','operator','M_aux_sigma','M_aux_sigma_prime','alpha','M_calibrated_sigma','C_original','C_calibrated'])
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
