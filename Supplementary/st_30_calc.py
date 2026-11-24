from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from dataclasses import dataclass
from types import SimpleNamespace
from st_2_calc import fresh_models
OUTPUT=Path(__file__).with_name('st_30.csv')
LENGTHS=range(6,27,2)
STATES_PER_PARITY=64
@dataclass(frozen=True)
class KuboPrecomp:
 even_energies:np.ndarray
 odd_energies:np.ndarray
 num_1_even_odd:np.ndarray
 num_2_odd_odd:np.ndarray
 num_3_odd_odd:np.ndarray
 num_4_odd_even:np.ndarray
def precompute_kubo(model:LowEnergyModel,A:np.ndarray|None=None,B:np.ndarray|None=None,C:np.ndarray|None=None)->KuboPrecomp:
 A_mat=model.O_sigma if A is None else np.asarray(A,dtype=np.complex128)
 B_mat=model.O_sigma if B is None else np.asarray(B,dtype=np.complex128)
 C_mat=model.O_epsilon if C is None else np.asarray(C,dtype=np.complex128)
 Ne=model.n_even
 even_energies=np.asarray(model.energies[:Ne],dtype=np.float64)
 odd_energies=np.asarray(model.energies[Ne:],dtype=np.float64)
 num_1=C_mat[0,:Ne,None]*B_mat[:Ne,Ne:]*A_mat[Ne:,0][None,:]
 num_2=B_mat[0,Ne:,None]*C_mat[Ne:,Ne:]*A_mat[Ne:,0][None,:]
 num_3=A_mat[0,Ne:,None]*C_mat[Ne:,Ne:]*B_mat[Ne:,0][None,:]
 num_4=A_mat[0,Ne:,None]*B_mat[Ne:,:Ne]*C_mat[:Ne,0][None,:]
 return KuboPrecomp(even_energies=even_energies,odd_energies=odd_energies,num_1_even_odd=np.asarray(num_1,dtype=np.complex128),num_2_odd_odd=np.asarray(num_2,dtype=np.complex128),num_3_odd_odd=np.asarray(num_3,dtype=np.complex128),num_4_odd_even=np.asarray(num_4,dtype=np.complex128))
def chi_sequence_decomposed(pre:KuboPrecomp,omega_sum:float,omega_first:float,eta:float)->tuple[complex,complex,complex,complex]:
 Ee_col=pre.even_energies[:,None]
 Ee_row=pre.even_energies[None,:]
 Eo_col=pre.odd_energies[:,None]
 Eo_row=pre.odd_energies[None,:]
 term_1=np.sum(pre.num_1_even_odd/((omega_sum-Ee_col+1j*eta)*(omega_first-Eo_row+1j*eta)))
 transition_oo=omega_sum+Eo_col-Eo_row+1j*eta
 term_2=-np.sum(pre.num_2_odd_odd/(transition_oo*(omega_first-Eo_row+1j*eta)))
 term_3=-np.sum(pre.num_3_odd_odd/(transition_oo*(omega_first+Eo_col+1j*eta)))
 term_4=np.sum(pre.num_4_odd_even/((omega_sum+Ee_row+1j*eta)*(omega_first+Eo_col+1j*eta)))
 return(complex(term_1),complex(term_2),complex(term_3),complex(term_4))
def chi_sequence(pre:KuboPrecomp,omega_sum:float,omega_first:float,eta:float)->complex:
 return complex(sum(chi_sequence_decomposed(pre,omega_sum,omega_first,eta)))
def chi_two_tone(pre_ab:KuboPrecomp,pre_ba:KuboPrecomp,omega1:float,omega2:float,eta:float)->complex:
 omega_sum=omega1+omega2
 return complex(chi_sequence(pre_ab,omega_sum,omega1,eta)+chi_sequence(pre_ba,omega_sum,omega2,eta))
def laurent_design_row(delta1:float,delta_sum:float,eta:float,nuisance_degree:int)->list[complex]:
 D1=delta1+1j*eta
 Dsum=delta_sum+1j*eta
 row:list[complex]=[1.0/(D1*Dsum)]
 for q in range(nuisance_degree+1):
  row.append(delta_sum**q/D1)
 for p in range(nuisance_degree+1):
  row.append(delta1**p/Dsum)
 for total_degree in range(nuisance_degree+1):
  for p in range(total_degree+1):
   q=total_degree-p
   row.append(delta1**p*delta_sum**q+0j)
 return row
def finite_vector(values,name='values',*,complex_ok=True):
 a=np.asarray(values,dtype=np.complex128 if complex_ok else np.float64)
 if a.ndim!=1:
  raise ValueError(f'{name} must be a one-dimensional vector, got {a.shape}')
 if not np.all(np.isfinite(a)):
  raise ValueError(f'{name} contains nonfinite values')
 return a
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
def calculate(retained_models=None,lengths=LENGTHS,states_per_parity=STATES_PER_PARITY):
 retained_models=fresh_models(lengths,states_per_parity)if retained_models is None else retained_models
 rows=[]
 for length,data in sorted(retained_models.items()):
  model=SimpleNamespace(**data)
  model.gap_sigma=model.energies[model.sigma_index]
  model.gap_epsilon=model.energies[model.epsilon_index]
  reference=float((model.O_sigma[model.epsilon_index,model.sigma_index]/model.M_sigma).real)
  precomputed=precompute_kubo(model)
  normalization=model.M_epsilon*model.M_sigma**2
  errors,conditions=([],[])
  for eta in[0.0025,0.005,0.01]:
   for width in[3,6,9]:
    offsets=np.linspace(-width,width,11)
    coordinates=[(float(x*eta),float(y*eta))for x in offsets for y in offsets]
    values=np.array([chi_two_tone(precomputed,precomputed,model.gap_sigma+x,model.gap_epsilon+y-model.gap_sigma-x,eta)for x,y in coordinates])
    for degree in[1,2,3]:
     design=np.array([laurent_design_row(x,y,eta,degree)for x,y in coordinates])
     coefficients,meta=solve_lstsq(design,values)
     errors.append(abs(coefficients[0]/normalization-reference))
     conditions.append(meta['condition_number_scaled'])
  assert len(errors)==27
  rows.append((length,len(errors),max(errors),min(conditions),max(conditions)))
 return pd.DataFrame(rows,columns=['L','configurations','max_complex_error','kappa_min','kappa_max'])
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
