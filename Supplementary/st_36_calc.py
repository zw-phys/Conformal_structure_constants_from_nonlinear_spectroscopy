from pathlib import Path
import numpy as np
import pandas as pd
from st_1_calc import make_orbits,projected_model
LENGTHS=tuple(range(6,27,2))
STATES_PER_PARITY=64
MODELS={'p1':(1,),'p2':(2,),'p3':(3,),'p12':(1,2),'p24':(2,4)}
MIN_LENGTHS=(6,8,10,12)
OUTPUT=Path(__file__).with_name('st_36.csv')
def static_series(lengths=LENGTHS,j2=0.0,field=1.0,states_per_parity=STATES_PER_PARITY):
 rows=[]
 for length in lengths:
  orbits=make_orbits(length)
  _,_,_,static=projected_model(length,j2,field,orbits,np.random.default_rng(2718),states_per_parity=states_per_parity)
  rows.append(dict(L=length,J2=j2,h=field,**static))
 return pd.DataFrame(rows)
def validate_design(design,values=None):
 raw=np.asarray(design)
 if raw.ndim!=2:
  raise ValueError(f'design must be a matrix, got shape {raw.shape}')
 if raw.shape[0]==0 or raw.shape[1]==0:
  raise np.linalg.LinAlgError(f'empty design {raw.shape}')
 if raw.shape[0]<raw.shape[1]:
  raise np.linalg.LinAlgError(f'underdetermined design: {raw.shape[0]} rows, {raw.shape[1]} columns')
 dtype=np.result_type(raw.dtype,np.float64)
 if values is not None:
  dtype=np.result_type(dtype,np.asarray(values).dtype)
 A=np.asarray(raw,dtype=dtype)
 if not np.all(np.isfinite(A)):
  raise ValueError('design contains nonfinite values')
 y=None
 if values is not None:
  y=np.asarray(values,dtype=dtype)
  if y.ndim!=1 or y.shape[0]!=A.shape[0]:
   raise ValueError(f'values shape {y.shape} does not match {A.shape[0]} design rows')
  if not np.all(np.isfinite(y)):
   raise ValueError('values contain nonfinite values')
 norms=np.linalg.norm(A,axis=0)
 if not np.all(np.isfinite(norms)):
  raise ValueError('nonfinite design column norms')
 if np.any(norms==0):
  raise np.linalg.LinAlgError('design contains a zero column')
 scaled=A/norms[None,:]
 sv=np.linalg.svd(scaled,compute_uv=False)
 tolerance=np.finfo(sv.dtype).eps*max(A.shape)*sv[0]
 rank=int(np.sum(sv>tolerance))
 if rank!=A.shape[1]:
  raise np.linalg.LinAlgError(f'rank-deficient design: rank {rank}, columns {A.shape[1]} (duplicate/dependent columns)')
 return(A,y,norms,scaled,sv)
def solve_lstsq(design,values):
 A,y,norms,scaled,sv=validate_design(design,values)
 bscaled,_,rank,sv_solve=np.linalg.lstsq(scaled,y,rcond=None)
 if rank!=A.shape[1]:
  raise np.linalg.LinAlgError(f'least-squares solver returned rank {rank} for {A.shape[1]} columns')
 coefficients=bscaled/norms
 if not np.all(np.isfinite(coefficients)):
  raise FloatingPointError('fit produced nonfinite coefficients')
 residual=y-A@coefficients
 rss=float(np.sum(abs(residual)**2))
 rms=float(np.sqrt(rss/len(y)))
 yrms=float(np.sqrt(np.mean(abs(y)**2)))
 meta=dict(n_rows=int(A.shape[0]),n_columns=int(A.shape[1]),rank=int(rank),singular_min_scaled=float(sv_solve[-1]),singular_max_scaled=float(sv_solve[0]),condition_number_scaled=float(sv_solve[0]/sv_solve[-1]),rms_abs=rms,relative_rms=rms/yrms if yrms else 0.0,residual_sum_squares=rss)
 return(coefficients,meta)
def checked_pinv(design):
 A,_,norms,scaled,_=validate_design(design)
 return np.linalg.pinv(scaled)/norms[:,None]
def calculate(lengths=LENGTHS,minimum_lengths=MIN_LENGTHS,states_per_parity=STATES_PER_PARITY):
 data=static_series(lengths,states_per_parity=states_per_parity)
 rows=[]
 for name,powers in MODELS.items():
  intercepts=[]
  for lower in minimum_lengths:
   window=data.loc[data.L>=lower]
   inverse_L=1/window.L.to_numpy(dtype=float)
   design=np.column_stack([np.ones(len(window)),*(inverse_L**p for p in powers)])
   coefficients,_=solve_lstsq(design,window.C_direct)
   intercepts.append(coefficients[0])
  rows.append((name,*intercepts))
 return pd.DataFrame(rows,columns=['model',*[f'C_inf_L{L}'for L in minimum_lengths]])
def legacy_residual_bootstrap(samples_per_fit=3000,seed=20260710):
 data=static_series(range(6,19,2)).rename(columns={'C_direct':'C_direct_audit'})
 rng=np.random.default_rng(seed)
 fits,pooled=([],[])
 for powers in[(1,),(2,),(3,),(1,2),(2,4)]:
  for lower in[6,8,10,12]:
   selected=data.loc[data.L>=lower]
   L=selected.L.to_numpy(dtype=float)
   values=selected.C_direct_audit.to_numpy()
   design=np.column_stack([np.ones(len(L)),*(L**(-p)for p in powers)])
   coefficients,meta=solve_lstsq(design,values)
   residual=values-design@coefficients
   weights=checked_pinv(design)[0]
   draws=weights@(design@coefficients)+rng.choice(residual-residual.mean(),size=(samples_per_fit,len(L)))@weights
   pooled.extend(draws.tolist())
   fits.append((str(powers),lower,coefficients[0],np.sqrt(np.mean(residual**2)),meta['condition_number_scaled']))
 return(pd.DataFrame(fits,columns=['powers','L_min','C_infinity','fit_rms','condition_scaled']),dict(zip([0.025,0.5,0.975],np.quantile(pooled,[0.025,0.5,0.975]))))
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
