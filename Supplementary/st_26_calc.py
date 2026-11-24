from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import math
from dataclasses import asdict,dataclass
from time import perf_counter
from typing import Sequence
import scipy.sparse as sp
import scipy.sparse.linalg as spla
OUTPUT=Path(__file__).with_name('st_26.csv')
LENGTHS=range(7,27)
rydberg_ISING_C=0.5
rydberg_TWO_PI=2*math.pi
@dataclass(frozen=True)
class rydberg_DeviceParameters:
 Omega_over_2pi_MHz:float=6.0
 Delta_over_2pi_MHz:float=10.2
 V1_over_2pi_MHz:float=164.6
 V2_over_2pi_MHz:float=3.06
 T1_us:float=63.0
 @property
 def Delta_over_Omega(self)->float:
  return self.Delta_over_2pi_MHz/self.Omega_over_2pi_MHz
 @property
 def V1_over_Omega(self)->float:
  return self.V1_over_2pi_MHz/self.Omega_over_2pi_MHz
 @property
 def V2_over_Omega(self)->float:
  return self.V2_over_2pi_MHz/self.Omega_over_2pi_MHz
 @property
 def physical_scale_rad_per_us(self)->float:
  return rydberg_TWO_PI*self.Omega_over_2pi_MHz
@dataclass(frozen=True)
class rydberg_BoundaryRow:
 L:int
 parity:str
 blockade_dimension:int
 boundary_condition:str
 operator_window:str
 sigma_index:int
 epsilon_index:int
 gap_sigma_over_Omega:float
 gap_epsilon_over_Omega:float
 gap_ratio:float
 M_sigma:float
 M_epsilon:float
 B_epsilon_sigma:float
 C_direct:float
 C_minus_half:float
 sigma_weight:float
 epsilon_weight:float
 sigma_target_projector_fraction:float
 epsilon_target_projector_fraction:float
 solver:str
def rydberg_deterministic_vector(n:int,phase:float=0.173)->np.ndarray:
 x=np.arange(n,dtype=float)
 vec=np.cos(phase*(x+1.0))+0.37*np.sin(0.619*phase*(x+2.0))
 norm=np.linalg.norm(vec)
 if norm==0.0:
  vec[0]=1.0
  norm=1.0
 return vec/norm
def rydberg_target_projector_fraction(source:np.ndarray,selected_weight:float)->float:
 total_weight=float(np.vdot(source,source).real)
 return float(selected_weight/total_weight)if total_weight>0.0 else 0.0
def rydberg_generate_blockade_basis(L:int)->np.ndarray:
 if L<2:
  raise ValueError('L must be at least 2')
 states:list[int]=[]
 def recurse(site:int,previous_occupied:bool,state:int)->None:
  if site==L:
   states.append(state)
   return
  recurse(site+1,False,state)
  if not previous_occupied:
   recurse(site+1,True,state|1<<site)
 recurse(0,False,0)
 states.sort()
 return np.asarray(states,dtype=np.int64)
def rydberg_build_open_rydberg_hamiltonian(L:int,params:rydberg_DeviceParameters,boundary_condition:str='compensated',include_h2:bool=True)->tuple[np.ndarray,dict[int,int],sp.csr_matrix]:
 if boundary_condition not in{'native','compensated'}:
  raise ValueError("boundary_condition must be 'native' or 'compensated'")
 states=rydberg_generate_blockade_basis(L)
 index={int(state):i for i,state in enumerate(states)}
 dim=len(states)
 Delta=params.Delta_over_Omega
 V1=params.V1_over_Omega
 V2=params.V2_over_Omega
 h2_coefficient=-1.0/(4.0*V1)
 rows:list[int]=[]
 cols:list[int]=[]
 data:list[float]=[]
 for column,raw_state in enumerate(states):
  state=int(raw_state)
  occupations=[state>>i&1 for i in range(L)]
  diagonal=0.0
  for i,n_i in enumerate(occupations):
   local_detuning=Delta
   if boundary_condition=='compensated'and i in{0,L-1}:
    local_detuning=Delta-V2
   diagonal-=local_detuning*n_i
  for i in range(L):
   if not occupations[i]:
    continue
   for j in range(i+2,L):
    if occupations[j]:
     distance=j-i
     diagonal+=64.0*V2/float(distance**6)
  if include_h2:
   diagonal+=h2_coefficient*(2.0*sum(occupations)-1.5*sum((occupations[i-1]*occupations[i+1]for i in range(1,L-1))))
  rows.append(column)
  cols.append(column)
  data.append(diagonal)
  for i in range(L):
   left=0 if i==0 else occupations[i-1]
   right=0 if i==L-1 else occupations[i+1]
   if left==0 and right==0:
    target_state=state^1<<i
    target=index.get(target_state)
    if target is not None:
     rows.append(target)
     cols.append(column)
     data.append(0.5)
  if include_h2:
   for i in range(L-1):
    if occupations[i]==occupations[i+1]:
     continue
    left=0 if i==0 else occupations[i-1]
    right=0 if i+2>=L else occupations[i+2]
    if left==0 and right==0:
     target_state=state^1<<i^1<<i+1
     target=index.get(target_state)
     if target is not None:
      rows.append(target)
      cols.append(column)
      data.append(h2_coefficient)
 H=sp.coo_matrix((data,(rows,cols)),shape=(dim,dim),dtype=float).tocsr()
 H.sum_duplicates()
 asymmetry=H-H.T
 if asymmetry.nnz and np.max(np.abs(asymmetry.data))>1e-12:
  raise RuntimeError('Hamiltonian construction is not Hermitian')
 return(states,index,H)
def rydberg_bond_weights(L:int,window:str,bulk_fraction:float=0.56,edge_bonds:int=2)->np.ndarray:
 n_bonds=L-1
 if window=='global':
  weights=np.ones(n_bonds,dtype=float)
 elif window=='bulk_hann':
  x=np.arange(n_bonds,dtype=float)
  center=0.5*(n_bonds-1)
  half_width=max(1.0,0.5*bulk_fraction*n_bonds)
  u=np.abs(x-center)/half_width
  weights=np.where(u<1.0,0.5*(1.0+np.cos(math.pi*u)),0.0)
 elif window=='boundary':
  weights=np.zeros(n_bonds,dtype=float)
  count=min(edge_bonds,max(1,n_bonds//2))
  weights[:count]=1.0
  weights[-count:]=1.0
 else:
  raise ValueError(f'Unknown operator window: {window}')
 norm=float(np.sum(weights))
 if norm<=0.0:
  raise RuntimeError('Operator window has zero support')
 return weights/norm
def rydberg_bond_operator_diagonals(states:np.ndarray,L:int,window:str)->tuple[np.ndarray,np.ndarray]:
 weights=rydberg_bond_weights(L,window)
 sigma=np.zeros(len(states),dtype=float)
 epsilon=np.zeros(len(states),dtype=float)
 for basis_index,raw_state in enumerate(states):
  state=int(raw_state)
  for j,weight in enumerate(weights):
   n_j=state>>j&1
   n_next=state>>j+1&1
   sigma[basis_index]+=weight*(-1.0)**j*(n_j-n_next)
   epsilon[basis_index]+=weight*(n_j+n_next)
 return(sigma,epsilon)
def rydberg_tracked_target_indices(energies:np.ndarray,vectors:np.ndarray,sigma_diagonal:np.ndarray,epsilon_diagonal:np.ndarray,L:int)->tuple[int,int,np.ndarray]:
 ground=vectors[:,0]
 sigma_source=sigma_diagonal*ground
 sigma_source-=ground*float(np.dot(ground,sigma_source))
 epsilon_source=epsilon_diagonal*ground
 epsilon_source-=ground*float(np.dot(ground,epsilon_source))
 sigma_weights=np.abs(vectors.T@sigma_source)**2
 epsilon_weights=np.abs(vectors.T@epsilon_source)**2
 sigma_weights[0]=0.0
 epsilon_weights[0]=0.0
 sigma_index=int(np.argmax(sigma_weights))
 gap_sigma=float(energies[sigma_index]-energies[0])
 ratios=(energies-energies[0])/gap_sigma
 low,high=(3.75,5.05)if L%2==0 else(1.8,2.2)
 candidate_mask=(ratios>=low)&(ratios<=high)
 candidate_mask[0]=False
 candidate_mask[sigma_index]=False
 candidate_weights=np.where(candidate_mask,epsilon_weights,0.0)
 if float(np.max(candidate_weights))<=1e-18:
  candidate_mask=np.ones_like(epsilon_weights,dtype=bool)
  candidate_mask[0]=False
  candidate_mask[sigma_index]=False
  candidate_weights=np.where(candidate_mask,epsilon_weights,0.0)
 epsilon_index=int(np.argmax(candidate_weights))
 return(sigma_index,epsilon_index,candidate_mask)
def rydberg_assign_states_and_matrix_elements(energies:np.ndarray,vectors_input:np.ndarray,sigma_diagonal:np.ndarray,epsilon_diagonal:np.ndarray,fixed_sigma_index:int|None=None,fixed_epsilon_index:int|None=None,epsilon_candidate_mask:np.ndarray|None=None)->dict:
 vectors=np.asarray(vectors_input,dtype=float).copy()
 ground=vectors[:,0]
 sigma_source=sigma_diagonal*ground
 sigma_source-=ground*float(np.dot(ground,sigma_source))
 epsilon_source=epsilon_diagonal*ground
 epsilon_source-=ground*float(np.dot(ground,epsilon_source))
 sigma_amplitudes=vectors.T@sigma_source
 epsilon_amplitudes=vectors.T@epsilon_source
 sigma_weights=np.abs(sigma_amplitudes)**2
 epsilon_weights=np.abs(epsilon_amplitudes)**2
 sigma_weights[0]=0.0
 epsilon_weights[0]=0.0
 sigma_index=int(np.argmax(sigma_weights))if fixed_sigma_index is None else int(fixed_sigma_index)
 if fixed_epsilon_index is None:
  epsilon_candidates=epsilon_weights.copy()
  epsilon_candidates[sigma_index]=0.0
  epsilon_index=int(np.argmax(epsilon_candidates))
 else:
  epsilon_index=int(fixed_epsilon_index)
 M_sigma=float(np.dot(vectors[:,sigma_index],sigma_diagonal*ground))
 if M_sigma<0.0:
  vectors[:,sigma_index]*=-1.0
  M_sigma=-M_sigma
 M_epsilon=float(np.dot(vectors[:,epsilon_index],epsilon_diagonal*ground))
 if M_epsilon<0.0:
  vectors[:,epsilon_index]*=-1.0
  M_epsilon=-M_epsilon
 B=float(np.dot(vectors[:,epsilon_index],sigma_diagonal*vectors[:,sigma_index]))
 C_direct=B/M_sigma
 if epsilon_candidate_mask is None:
  epsilon_ratio_weights=epsilon_weights.copy()
  epsilon_ratio_weights[sigma_index]=0.0
 else:
  epsilon_ratio_weights=np.where(epsilon_candidate_mask,epsilon_weights,0.0)
 return{'vectors':vectors,'sigma_index':sigma_index,'epsilon_index':epsilon_index,'gap_sigma':float(energies[sigma_index]-energies[0]),'gap_epsilon':float(energies[epsilon_index]-energies[0]),'M_sigma':M_sigma,'M_epsilon':M_epsilon,'B':B,'C_direct':C_direct,'sigma_weight':float(sigma_weights[sigma_index]),'epsilon_weight':float(epsilon_weights[epsilon_index]),'sigma_target_fraction':rydberg_target_projector_fraction(sigma_source,sigma_weights[sigma_index]),'epsilon_target_fraction':rydberg_target_projector_fraction(epsilon_source,epsilon_weights[epsilon_index])}
def rydberg_lowest_eigensystem(H:sp.csr_matrix,n_low:int)->tuple[np.ndarray,np.ndarray,str]:
 dim=H.shape[0]
 if dim<=300 or n_low>=dim-1:
  energies,vectors=np.linalg.eigh(H.toarray())
  keep=min(n_low,dim)
  return(energies[:keep],vectors[:,:keep],'dense_eigh')
 k=min(n_low,dim-2)
 ncv=min(dim,max(4*k+12,64))
 energies,vectors=spla.eigsh(H,k=k,which='SA',tol=1e-11,maxiter=400000,ncv=ncv,v0=rydberg_deterministic_vector(dim))
 order=np.argsort(energies)
 return(energies[order],vectors[:,order],'sparse_eigsh')
def rydberg_analyze_boundary_model(L:int,params:rydberg_DeviceParameters,boundary_condition:str,n_low:int,windows:Sequence[str])->list[rydberg_BoundaryRow]:
 states,_,H=rydberg_build_open_rydberg_hamiltonian(L=L,params=params,boundary_condition=boundary_condition)
 energies,vectors,solver=rydberg_lowest_eigensystem(H,n_low=n_low)
 residual=float(np.max(np.linalg.norm(H@vectors-vectors*energies,axis=0)))
 if residual>1e-08:
  raise ValueError(f'Eigensolver residual exceeds tolerance: {residual}')
 print(f'L={L} eigensolver max residual={residual:.4g}',flush=True)
 global_sigma,global_epsilon=rydberg_bond_operator_diagonals(states,L,'global')
 sigma_index,epsilon_index,epsilon_candidate_mask=rydberg_tracked_target_indices(energies,vectors,global_sigma,global_epsilon,L)
 rows:list[rydberg_BoundaryRow]=[]
 for window in windows:
  sigma_diagonal,epsilon_diagonal=rydberg_bond_operator_diagonals(states,L,window)
  result=rydberg_assign_states_and_matrix_elements(energies,vectors,sigma_diagonal,epsilon_diagonal,fixed_sigma_index=sigma_index,fixed_epsilon_index=epsilon_index,epsilon_candidate_mask=epsilon_candidate_mask)
  gap_sigma=result['gap_sigma']
  gap_epsilon=result['gap_epsilon']
  rows.append(rydberg_BoundaryRow(L=L,parity='even'if L%2==0 else'odd',blockade_dimension=len(states),boundary_condition=boundary_condition,operator_window=window,sigma_index=result['sigma_index'],epsilon_index=result['epsilon_index'],gap_sigma_over_Omega=gap_sigma,gap_epsilon_over_Omega=gap_epsilon,gap_ratio=gap_epsilon/gap_sigma,M_sigma=result['M_sigma'],M_epsilon=result['M_epsilon'],B_epsilon_sigma=result['B'],C_direct=result['C_direct'],C_minus_half=result['C_direct']-rydberg_ISING_C,sigma_weight=result['sigma_weight'],epsilon_weight=result['epsilon_weight'],sigma_target_projector_fraction=result['sigma_target_fraction'],epsilon_target_projector_fraction=result['epsilon_target_fraction'],solver=solver))
 return rows
def recompute_scan(lengths=LENGTHS):
 rows=[]
 for length in lengths:
  start=perf_counter()
  windows=['global','bulk_hann','boundary']if length==16 else['global']
  records=rydberg_analyze_boundary_model(length,rydberg_DeviceParameters(),'compensated',28,windows)
  for record in records:
   rows.append(dict(asdict(record),elapsed_seconds=perf_counter()-start))
  print(f'L={length}: completed in {perf_counter() - start:.2f}s',flush=True)
 return pd.DataFrame(rows)
def calculate(scan=None,lengths=LENGTHS):
 lengths=list(lengths)
 window_length=16
 if scan is None:
  scan=recompute_scan(lengths)
 compensated=scan.loc[scan.boundary_condition=='compensated'].copy()
 compensated['C_strip_direct']=compensated.B_epsilon_sigma/compensated.M_sigma
 bulk=compensated.loc[compensated.operator_window.eq('global')&compensated.L.isin(lengths)].copy()
 if sorted(bulk.L.tolist())!=lengths:
  raise ValueError('Expected every requested Rydberg chain length')
 bulk['part']='even_odd_global'
 windows=compensated.loc[compensated.L==window_length].copy()
 windows['part']='window_comparison'
 result=pd.concat([bulk.sort_values(['parity','L']),windows],ignore_index=True)
 result=result[['part','L','parity','operator_window','C_strip_direct']]
 if not np.isfinite(result.C_strip_direct).all():
  raise ValueError('Nonfinite Rydberg matrix-element ratio')
 return result
def strip_intercept_diagnostics(table):
 even=table.loc[table.part.eq('even_odd_global')&table.parity.eq('even')]
 rows=[]
 for lower in(8,10,12):
  selected=even.loc[even.L>=lower]
  lengths=selected.L.to_numpy(dtype=float)
  values=selected.C_strip_direct.to_numpy()
  for powers in((1,),(2,),(1,2)):
   design=np.column_stack([np.ones(len(lengths))]+[lengths**(-p)for p in powers])
   scales=np.linalg.norm(design,axis=0)
   coefficients,_,rank,_=np.linalg.lstsq(design/scales,values,rcond=None)
   if rank!=design.shape[1]:
    raise np.linalg.LinAlgError('Rank-deficient strip correction fit')
   coefficients/=scales
   rows.append(dict(L_min=lower,L_max=int(lengths.max()),powers=powers,C_infinity=float(coefficients[0]),condition_scaled=float(np.linalg.cond(design/scales))))
 return pd.DataFrame(rows)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
