from pathlib import Path
import numpy as np
import pandas as pd
import tci_model_calc as tci
LENGTHS=range(6,27,2)
G=0.856
STATES=96
CYCLES=96
RAMP=6
AREA=0.005
STEPS=256
OUTPUT=Path(__file__).with_name('st_24.csv')
def calculate():
 data=tci.reference_series(LENGTHS,G,STATES,CYCLES,RAMP,AREA,STEPS)
 result=data[['L','gap_ratio','C_direct','C_frequency_real','C_real','C_imag']]
 return result.rename(columns={'C_frequency_real':'C_Kubo_real','C_real':'C_time_real','C_imag':'C_time_imag'})
def correction_model_diagnostics(frequency):
 families={'4/5':(0.8,),'4/5,2':(0.8,2),'4/5,6/5':(0.8,1.2),'1':(1,),'1,2':(1,2),'2':(2,)}
 lengths=frequency.L.to_numpy(dtype=float)
 values=frequency.C_Kubo_real.to_numpy()
 rows=[]
 for name,powers in families.items():
  def fit(keep):
   L=lengths[keep]
   design=np.column_stack([np.ones(len(L)),*(L**(-p)for p in powers)])
   scale=np.linalg.norm(design,axis=0)
   coeff=tci.scaled_lstsq(design,values[keep])
   return(coeff[0],np.linalg.cond(design/scale))
  intercept,condition=fit(np.ones(len(lengths),dtype=bool))
  upper,_=fit(lengths>=8)
  omitted=[fit(lengths!=length)[0]for length in lengths]
  rows.append((name,intercept,upper,condition,min(omitted),max(omitted)))
 return pd.DataFrame(rows,columns=['powers','C_infinity_L6','C_infinity_L8','condition_L6','LOO_min','LOO_max'])
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
