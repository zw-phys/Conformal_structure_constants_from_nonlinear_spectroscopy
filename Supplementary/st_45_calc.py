from pathlib import Path
import numpy as np
import pandas as pd
import tci_model_calc as tci
LENGTHS=range(6,27,2)
G=0.856
STATES=96
CYCLES=384
RAMP=6
AREA=0.0025
STEPS=256
CALIBRATION_LENGTH=18
PROBES=('bare','improved')
WINDOWS=[('two',(4/5,2),6),('three',(4/5,6/5,2),6),('three',(4/5,6/5,2),16)]
OUTPUT=Path(__file__).with_name('st_45.csv')
def calculate():
 data=tci.paired_series(LENGTHS,G,STATES,CYCLES,RAMP,AREA,STEPS,CALIBRATION_LENGTH)
 rows=[]
 for name,powers,lower in WINDOWS:
  for probe in PROBES:
   sample=data.loc[(data.L>=lower)&data.probe.eq(probe)].sort_values('L')
   L=sample.L.to_numpy(dtype=float)
   design=np.column_stack([np.ones(len(L)),*(L**(-p)for p in powers)])
   norms=np.linalg.norm(design,axis=0)
   scaled=design/norms
   if np.linalg.matrix_rank(scaled)!=design.shape[1]:
    raise ValueError('Rank-deficient size-fit design')
   weights=np.linalg.pinv(scaled)[0]/norms[0]
   static,pulse,background,higher=[weights@sample[column]for column in('C_direct','full_real','background_real','higher_real')]
   rows.append((name,int(L[0]),int(L[-1]),probe,static,pulse,pulse-static,background,higher,abs(weights).sum(),len(L)-len(norms)))
 result=pd.DataFrame(rows,columns=['model','L_min','L_max','probe','C_static','C_pulse','delta_C','delta_background','delta_higher','weight_l1','residual_dof'])
 np.testing.assert_allclose(result.delta_C,result.delta_background+result.delta_higher,rtol=0,atol=4e-13)
 return result
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
