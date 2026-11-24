from pathlib import Path
import numpy as np
import pandas as pd
from st_36_calc import static_series,solve_lstsq
LENGTHS=tuple(range(6,27,2))
COUPLING=0.05
FIELDS=(1.0838737,1.08387382,1.0838741)
MODELS={'log2':(2,),'log23':(2,3),'log24':(2,4)}
MIN_LENGTHS=(6,8,10,12,14)
OUTPUT=Path(__file__).with_name('st_39.csv')
def field_series(lengths=LENGTHS):
 return pd.concat([static_series(lengths,COUPLING,field)for field in FIELDS],ignore_index=True)
def calculate(lengths=LENGTHS,minimum_lengths=MIN_LENGTHS):
 data=field_series(lengths)
 rows=[]
 for h,field in data.groupby('h',sort=True):
  for model,powers in MODELS.items():
   for lower in minimum_lengths:
    window=field.loc[field.L>=lower].sort_values('L')
    L=window.L.to_numpy(dtype=float)
    values=window.C_direct.to_numpy()
    design=np.column_stack([np.ones(len(L)),np.log(L)/L**2,*(L**(-p)for p in powers)])
    if len(L)<=design.shape[1]:
     raise ValueError('Correction fits require positive residual degrees of freedom')
    coefficients,meta=solve_lstsq(design,values)
    residual=values-design@coefficients
    rows.append((h,model,lower,int(L[-1]),coefficients[0],coefficients[1],np.max(abs(residual)),meta['condition_number_scaled'],len(L)-design.shape[1]))
 windows=pd.DataFrame(rows,columns=['h','model','L_min','L_max','C_infinity','log_coefficient','max_residual','scaled_condition','residual_dof'])
 table=windows.groupby(['h','model'],sort=False,as_index=False).agg(C_min=('C_infinity','min'),C_max=('C_infinity','max'),condition_max=('scaled_condition','max'),min_dof=('residual_dof','min'))
 return pd.concat([table.assign(row_type='table'),windows.assign(row_type='fit_window')],ignore_index=True)
def independently_calibrated_field_range(lengths=LENGTHS,minimum_lengths=MIN_LENGTHS):
 data=field_series(lengths)
 frames=[data.loc[data.h==h].sort_values('L')for h in FIELDS]
 low,center,high=frames
 L=center.L.to_numpy(dtype=float)
 critical=[]
 for key in('gap_ratio','gap_sigma'):
  values=[(frame.gap_epsilon/frame.gap_sigma if key=='gap_ratio'else frame.gap_sigma).to_numpy()for frame in frames]
  y=values[1]
  derivative=(values[2]-values[0])/(FIELDS[2]-FIELDS[0])
  for powers in((2,3,4),(2,3,4,5)):
   for lower in minimum_lengths:
    keep=L>=lower
    size=L[keep]
    if key=='gap_ratio':
     design=np.column_stack([derivative[keep],*[-size**(-p)for p in powers]])
     target=8-y[keep]
     index=0
    else:
     design=np.column_stack([np.ones(len(size)),*[size**(-p)for p in powers],-size*derivative[keep]])
     target=size*y[keep]
     index=-1
    if len(size)<=design.shape[1]:
     raise ValueError('Spectral field fits require positive residual degrees of freedom')
    coefficients,_=solve_lstsq(design,target)
    critical.append(FIELDS[1]+coefficients[index])
 return(min(critical),max(critical))
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
