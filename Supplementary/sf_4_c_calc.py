from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sf_4_c.csv'
LENGTH=14
N_EVEN=N_ODD=24
ETA=0.005
BACKGROUND_DEGREE=2
GRID_POINTS=100
COORDINATES=np.linspace(-6.0,6.0,GRID_POINTS)
FIT_POINTS_PER_AXIS=11
DEFAULT_FIT_OFFSETS=np.array([-6,-4,-3,-2,-1,0,1,2,3,4,6],dtype=float)
from sf_4_a_calc import projected_operators
def projected_model(length=LENGTH,states=N_EVEN):
 e,o,s,z=projected_operators(length,states)
 ne=len(e)
 if not 0<o[0]<e[1]or not np.isfinite(s).all()or(not np.isfinite(z).all()):
  raise ValueError('Invalid fresh Ising spectrum or projected probes')
 for matrix in(s,z):
  np.testing.assert_allclose(matrix,matrix.T,atol=1e-12)
 numerators=(z[0,:ne,None]*s[:ne,ne:]*s[ne:,0][None,:],s[0,ne:,None]*z[ne:,ne:]*s[ne:,0][None,:],s[0,ne:,None]*z[ne:,ne:]*s[ne:,0][None,:],s[0,ne:,None]*s[ne:,:ne]*z[:ne,0][None,:])
 return(e,o,numerators,s,z)
def ordered_response(model,omega_sum,omega_first,eta):
 e,o,n=model[:3]
 transition=omega_sum+o[:,None]-o[None,:]+1j*eta
 t1=np.sum(n[0]/((omega_sum-e[:,None]+1j*eta)*(omega_first-o[None,:]+1j*eta)))
 t2=-np.sum(n[1]/(transition*(omega_first-o[None,:]+1j*eta)))
 t3=-np.sum(n[2]/(transition*(omega_first+o[:,None]+1j*eta)))
 t4=np.sum(n[3]/((omega_sum+e[None,:]+1j*eta)*(omega_first+o[:,None]+1j*eta)))
 return complex(t1+t2+t3+t4)
def response(model,delta1,delta_sum,eta):
 first=model[1][0]+delta1
 second=model[0][1]+delta_sum-first
 total=first+second
 return ordered_response(model,total,first,eta)+ordered_response(model,total,second,eta)
def laurent_row(delta1,delta_sum,eta,degree):
 d1,ds=(delta1+1j*eta,delta_sum+1j*eta)
 row=[1.0/(d1*ds)]
 row.extend((delta_sum**q/d1 for q in range(degree+1)))
 row.extend((delta1**p/ds for p in range(degree+1)))
 row.extend((delta1**p*delta_sum**(total-p)+0j for total in range(degree+1)for p in range(total+1)))
 return row
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
def fit_laurent(model,eta=ETA,width=6,degree=BACKGROUND_DEGREE,offsets=None):
 if offsets is None:
  offsets=np.linspace(-width,width,FIT_POINTS_PER_AXIS)
 design,values=([],[])
 for x in offsets:
  for y in offsets:
   d1,ds=(float(x)*eta,float(y)*eta)
   design.append(laurent_row(d1,ds,eta,degree))
   values.append(response(model,d1,ds,eta))
 design,values=(np.asarray(design),np.asarray(values))
 coeff,metadata=solve_lstsq(design,values)
 metadata.pop('residual_sum_squares')
 return(coeff,metadata)
def calculate_grid():
 model=projected_model()
 coefficients,meta=fit_laurent(model,offsets=DEFAULT_FIT_OFFSETS)
 full=np.empty((GRID_POINTS,GRID_POINTS),dtype=complex)
 fitted=np.empty_like(full)
 for i,x in enumerate(COORDINATES):
  for j,y in enumerate(COORDINATES):
   d1,ds=(x*ETA,y*ETA)
   full[i,j]=response(model,d1,ds,ETA)
   fitted[i,j]=np.asarray(laurent_row(d1,ds,ETA,BACKGROUND_DEGREE))@coefficients
 residual=full-fitted
 if not np.isfinite(residual).all():
  raise ValueError('Nonfinite frequency-grid residual')
 i,j=np.indices(residual.shape)
 ds,d1=np.meshgrid(COORDINATES,COORDINATES)
 result=pd.DataFrame(dict(image_row=i.ravel(),image_column=j.ravel(),delta1_eta=d1.ravel(),deltaSigma_eta=ds.ravel(),response=full.real.ravel(),fit=fitted.real.ravel(),residual=residual.real.ravel(),response_imag=full.imag.ravel(),fit_imag=fitted.imag.ravel(),residual_imag=residual.imag.ravel()))
 for key,value in dict(L=LENGTH,eta=ETA,background_degree=BACKGROUND_DEGREE,n_even=N_EVEN,n_odd=N_ODD,fit_points=meta['n_rows'],fit_rank=meta['rank']).items():
  result[key]=value
 return result
def calculate_scan():
 model=projected_model()
 direct=model[3][1,len(model[0])]/model[3][len(model[0]),0]
 normalization=model[4][1,0]*model[3][len(model[0]),0]**2
 rows=[]
 for degree in(1,2,3):
  configuration=0
  for eta in(0.0025,0.005,0.01):
   for width in(3,6,9):
    coeff,meta=fit_laurent(model,eta,width,degree)
    estimate=coeff[0]/normalization
    error=abs(estimate-direct)
    rows.append(dict(L=LENGTH,eta=eta,width_eta=width,degree=degree,C_real=estimate.real,C_imag=estimate.imag,complex_audit_error=error,**meta,configuration_index=configuration,complex_error_recomputed=error,C_direct_for_recheck=direct))
    configuration+=1
 return pd.DataFrame(rows)
def numerical_diagnostics():
 model=projected_model()
 design=np.array([laurent_row(x*ETA,y*ETA,ETA,2)for x in DEFAULT_FIT_OFFSETS for y in DEFAULT_FIT_OFFSETS])
 clean=np.array([response(model,x*ETA,y*ETA,ETA)for x in DEFAULT_FIT_OFFSETS for y in DEFAULT_FIT_OFFSETS])
 duplicate=design.copy()
 duplicate[:,1]=duplicate[:,0]
 zero=design.copy()
 zero[:,1]=0
 nonfinite=clean.copy()
 nonfinite[0]=np.nan
 invalid=((design[:9],clean[:9]),(duplicate,clean),(zero,clean),(design,nonfinite),(design,clean[:-1]))
 rejected=0
 for matrix,values in invalid:
  try:
   solve_lstsq(matrix,values)
  except(ValueError,np.linalg.LinAlgError):
   rejected+=1
  else:
   raise AssertionError('Production fit accepted invalid input')
 rng=np.random.default_rng(20274710)
 scale=0.0001*np.sqrt(np.mean(abs(clean)**2))
 normalization=model[4][1,0]*model[3][len(model[0]),0]**2
 estimates=[]
 for _ in range(200):
  noise=scale*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))/np.sqrt(2)
  coefficients,_=solve_lstsq(design,clean+noise)
  estimates.append(coefficients[0]/normalization)
 estimates=np.asarray(estimates)
 diagnostics=dict(rejected_fit_cases=rejected,noise_seed=20274710,noise_draws=200,noise_mean_real=np.mean(estimates.real),noise_std_real=np.std(estimates.real,ddof=1),noise_q025_real=np.quantile(estimates.real,0.025),noise_q975_real=np.quantile(estimates.real,0.975),noise_imag_rms=np.sqrt(np.mean(estimates.imag**2)))
 diagnostics.update(independent_liouville_check())
 return diagnostics
def independent_liouville_check():
 from scipy.linalg import solve_sylvester
 length,dim=(4,16)
 hamiltonian=np.zeros((dim,dim))
 spin,energy=(np.zeros_like(hamiltonian),np.zeros_like(hamiltonian))
 for state in range(dim):
  z=[1-2*(state>>site&1)for site in range(length)]
  zz=sum((z[site]*z[(site+1)%length]for site in range(length)))
  hamiltonian[state,state]=-zz
  spin[state,state]=sum(z)/length
  energy[state,state]=zz/length
  for site in range(length):
   target=state^1<<site
   hamiltonian[target,state]=-1
   energy[target,state]=-1/length
 energies,vectors=np.linalg.eigh(hamiltonian)
 energies-=energies[0]
 a=vectors.T@spin@vectors
 c=vectors.T@energy@vectors
 c-=c[0,0]*np.eye(dim)
 sigma_index=int(np.argmax(abs(a[1:,0])))+1
 epsilon_index=int(np.argmax(abs(c[1:,0])))+1
 rng=np.random.default_rng(20261016)
 q=np.linalg.qr(rng.normal(size=(dim,dim)))[0]
 h,ar,cr=(q@np.diag(energies)@q.T,q@a@q.T,q@c@q.T)
 rho,identity=(np.outer(q[:,0],q[:,0]),np.eye(dim))
 e_col,e_row=(energies[:,None],energies[None,:])
 n1=c[0,:,None]*a*a[:,0][None,:]
 n2=a[0,:,None]*c*a[:,0][None,:]
 n4=a[0,:,None]*a*c[:,0][None,:]
 def spectral(total,first):
  transition=total+e_col-e_row+1j*ETA
  return np.sum(n1/((total-e_col+1j*ETA)*(first-e_row+1j*ETA)))-np.sum(n2/(transition*(first-e_row+1j*ETA)))-np.sum(n2/(transition*(first+e_col+1j*ETA)))+np.sum(n4/((total+e_row+1j*ETA)*(first+e_col+1j*ETA)))
 def liouville(total,first):
  r1=solve_sylvester((first+1j*ETA)*identity-h,h,ar@rho-rho@ar)
  r2=solve_sylvester((total+1j*ETA)*identity-h,h,ar@r1-r1@ar)
  return np.trace(cr@r2)
 spectral_values,independent_values=([],[])
 for x in DEFAULT_FIT_OFFSETS:
  for y in DEFAULT_FIT_OFFSETS:
   first=energies[sigma_index]+x*ETA
   total=energies[epsilon_index]+y*ETA
   spectral_values.append(spectral(total,first)+spectral(total,total-first))
   independent_values.append(liouville(total,first)+liouville(total,total-first))
 spectral_values,independent_values=(np.array(spectral_values),np.array(independent_values))
 error=np.linalg.norm(spectral_values-independent_values)/np.linalg.norm(spectral_values)
 if error>1e-10:
  raise AssertionError(f'Independent Liouville validation failed: {error}')
 return dict(liouville_length=length,liouville_relative_error=error)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=OUTPUT,help='Destination CSV')
 parser.add_argument('--diagnostics',action='store_true',help='Include fresh fit/noise/Liouville diagnostics in the same CSV')
 args=parser.parse_args()
 result=calculate_scan()
 if args.diagnostics:
  for key,value in numerical_diagnostics().items():
   result[key]=value
 result.to_csv(args.output,index=False)
if __name__=='__main__':
 main()
