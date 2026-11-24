from pathlib import Path
import numpy as np
import pandas as pd
from st_36_calc import solve_lstsq
from st_38_calc import pulse_series
LENGTHS=tuple(range(6,27,2))
COUPLING=0.05
FIELD=1.08387382
SPECTRAL_WINDOW=(10,26)
SPECTRAL_POWERS=(2,3,4)
MIN_LENGTHS=(6,8,10,12,14)
REPRESENTATIVE_POWER=3
OUTPUT=Path(__file__).with_name('st_40.csv')
def spectral_kappa(data,window=SPECTRAL_WINDOW,powers=SPECTRAL_POWERS):
 selected=data.loc[(data.L>=window[0])&(data.L<=window[1])].sort_values('L')
 L=selected.L.to_numpy(dtype=float)
 design=np.column_stack([np.ones(len(L)),*[L**(-p)for p in powers]])
 if len(L)<=design.shape[1]:
  raise ValueError('Spectral calibration requires positive residual degrees of freedom')
 sigma,_=solve_lstsq(design,L*selected.gap_sigma.to_numpy())
 epsilon,_=solve_lstsq(design,L*selected.gap_epsilon.to_numpy())
 velocity=4*sigma[0]/np.pi
 if velocity<=0:
  raise ValueError('The fitted spectral velocity must be positive')
 return-3*(epsilon[1]+16*sigma[1])/(8*np.pi*velocity)
def calculate(lengths=LENGTHS,minimum_lengths=MIN_LENGTHS,spectral_window=SPECTRAL_WINDOW,cycles=768,area=0.01,steps=128):
 data=pulse_series(lengths,models=((COUPLING,FIELD),),cycles=cycles,area=area,steps=steps).sort_values('L')
 kappa=spectral_kappa(data,spectral_window)
 rows=[]
 for mode in('spectral','free'):
  for extra_power in(0,3,4):
   for lower in minimum_lengths:
    selected=data.loc[data.L>=lower]
    L=selected.L.to_numpy(dtype=float)
    log_term=np.log(L)/L**2
    columns=[np.ones(len(L))+kappa*log_term,L**(-2)]if mode=='spectral'else[np.ones(len(L)),log_term,L**(-2)]
    if extra_power:
     columns.append(L**(-extra_power))
    design=np.column_stack(columns)
    if len(L)<=design.shape[1]:
     raise ValueError('Paired fits require positive residual degrees of freedom')
    scale=np.linalg.norm(design,axis=0)
    scaled=design/scale
    values=np.column_stack([selected.C_direct,selected.C_real])
    fitted,_,rank,sv=np.linalg.lstsq(scaled,values,rcond=None)
    if rank!=design.shape[1]:
     raise np.linalg.LinAlgError('Rank-deficient paired correction fit')
    coefficients=fitted/scale[:,None]
    error=np.max(abs(values-design@coefficients),axis=0)
    weights=np.linalg.pinv(scaled)[0]/scale[0]
    np.testing.assert_allclose(weights@(values[:,1]-values[:,0]),coefficients[0,1]-coefficients[0,0],rtol=0,atol=2e-12)
    rows.append((mode,extra_power,lower,int(L[-1]),*coefficients[0],coefficients[0,1]-coefficients[0,0],*error,sv[0]/sv[-1],np.abs(weights).sum(),len(L)-len(scale)))
 windows=pd.DataFrame(rows,columns=['mode','extra_power','L_min','L_max','static_C_inf','pulse_C_inf','pulse_minus_static','static_max_residual','pulse_max_residual','scaled_condition','intercept_weight_l1','residual_dof'])
 table=windows.loc[windows['mode'].eq('spectral')&windows.extra_power.eq(REPRESENTATIVE_POWER)]
 return pd.concat([table.assign(row_type='table'),windows.assign(row_type='fit_window')],ignore_index=True)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
