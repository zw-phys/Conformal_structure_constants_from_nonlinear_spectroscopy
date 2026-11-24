from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
import scipy.sparse as sp
import scipy.sparse.linalg as spla
OUTPUT=Path(__file__).with_name('st_29.csv')
LENGTHS=(6,14)
RETAINED_STATES=24
@dataclass(frozen=True)
class LowEnergyModel:
 L:int
 J:float
 h:float
 complete_spectrum:bool
 sector_dimension:int
 n_even:int
 n_odd:int
 solver_even:str
 solver_odd:str
 energies:np.ndarray
 O_sigma:np.ndarray
 O_epsilon:np.ndarray
 sigma_index:int
 epsilon_index:int
 gap_sigma:float
 gap_epsilon:float
 M_sigma:float
 M_epsilon:float
 B_epsilon_sigma:float
 R_direct:complex
 C_direct:float
 x_sigma:float
 x_epsilon:float
 sigma_target_projector_fraction:float
 epsilon_target_projector_fraction:float
def bit_z(state:int,site:int)->int:
 return 1 if state>>site&1==0 else-1
def parity_representatives(L:int)->tuple[np.ndarray,dict[int,int],int]:
 mask=(1<<L)-1
 reps:list[int]=[]
 index:dict[int,int]={}
 for state in range(1<<L):
  flipped=state^mask
  if state<flipped:
   idx=len(reps)
   reps.append(state)
   index[state]=idx
   index[flipped]=idx
 return(np.asarray(reps,dtype=np.int64),index,mask)
def z_average(state:int,L:int)->float:
 return sum((bit_z(state,r)for r in range(L)))/float(L)
def zz_average(state:int,L:int)->float:
 return sum((bit_z(state,r)*bit_z(state,(r+1)%L)for r in range(L)))/float(L)
def target_index_and_sign(state:int,parity:int,index:dict[int,int],mask:int)->tuple[int,float]:
 flipped=state^mask
 rep=state if state<flipped else flipped
 sign=1.0 if state==rep else float(parity)
 return(index[rep],sign)
def build_x_average_sector(L:int,parity:int,reps:np.ndarray,index:dict[int,int],mask:int)->sp.csr_matrix:
 rows:list[int]=[]
 cols:list[int]=[]
 data:list[float]=[]
 amplitude=1.0/float(L)
 for col,rep_state in enumerate(reps):
  state=int(rep_state)
  for r in range(L):
   row,sign=target_index_and_sign(state^1<<r,parity,index,mask)
   rows.append(row)
   cols.append(col)
   data.append(amplitude*sign)
 matrix=sp.coo_matrix((data,(rows,cols)),shape=(len(reps),len(reps)),dtype=np.float64).tocsr()
 matrix.sum_duplicates()
 return matrix
def build_tfim_sector(L:int,parity:int,reps:np.ndarray,index:dict[int,int],mask:int,J:float,h:float)->sp.csr_matrix:
 zz_diagonal=np.asarray([-J*L*zz_average(int(state),L)for state in reps],dtype=np.float64)
 x_sum=L*build_x_average_sector(L,parity,reps,index,mask)
 return(sp.diags(zz_diagonal,format='csr')-h*x_sum).tocsr()
def build_oepsilon_sector(L:int,parity:int,reps:np.ndarray,index:dict[int,int],mask:int)->sp.csr_matrix:
 zz_diagonal=np.asarray([zz_average(int(state),L)for state in reps],dtype=np.float64)
 x_average=build_x_average_sector(L,parity,reps,index,mask)
 return(sp.diags(zz_diagonal,format='csr')-x_average).tocsr()
def osigma_bridge_diagonal(L:int,reps:np.ndarray)->np.ndarray:
 return np.asarray([z_average(int(state),L)for state in reps],dtype=np.float64)
def deterministic_start_vector(dim:int)->np.ndarray:
 grid=np.arange(dim,dtype=np.float64)
 vector=np.cos(0.137*grid)+np.sin(0.071*grid)
 norm=np.linalg.norm(vector)
 return vector/norm if norm>0.0 else vector
def eigenpairs(H:sp.csr_matrix,n_low:int,complete:bool,dense_cutoff_dim:int,eigsh_tol:float)->tuple[np.ndarray,np.ndarray,str]:
 dim=H.shape[0]
 if complete:
  if dim>dense_cutoff_dim:
   raise ValueError(f'complete spectrum requested for sector dimension {dim}, above dense cutoff {dense_cutoff_dim}')
  evals,evecs=np.linalg.eigh(H.toarray())
  return(np.asarray(evals),np.asarray(evecs),'dense_complete')
 n=min(max(4,int(n_low)),dim)
 if dim<=dense_cutoff_dim and n>=dim-1:
  evals,evecs=np.linalg.eigh(H.toarray())
  return(np.asarray(evals[:n]),np.asarray(evecs[:,:n]),'dense_truncated')
 if dim<=min(dense_cutoff_dim,256)or n>=max(2,dim//2):
  evals,evecs=np.linalg.eigh(H.toarray())
  return(np.asarray(evals[:n]),np.asarray(evecs[:,:n]),'dense_truncated')
 k=min(n,dim-2)
 evals,evecs=spla.eigsh(H,k=k,which='SA',tol=eigsh_tol,maxiter=300000,ncv=min(dim,max(4*k+1,32)),v0=deterministic_start_vector(dim))
 order=np.argsort(evals)
 return(np.asarray(evals[order]),np.asarray(evecs[:,order]),'eigsh_truncated')
def _dominant_sigma_state(psi0:np.ndarray,evecs_odd:np.ndarray,osigma_diag:np.ndarray)->tuple[int,float,float]:
 source=osigma_diag*psi0
 amplitudes=np.asarray([np.vdot(evecs_odd[:,n],source)for n in range(evecs_odd.shape[1])],dtype=np.complex128)
 weights=np.abs(amplitudes)**2
 index=int(np.argmax(weights))
 amplitude=np.vdot(evecs_odd[:,index],source)
 if amplitude.real<0.0:
  evecs_odd[:,index]*=-1.0
  amplitude=-amplitude
 if abs(amplitude.imag)>1e-09:
  raise RuntimeError(f'sigma matrix element is unexpectedly complex: {amplitude}')
 total_connected_weight=float(np.vdot(source,source).real)
 target_projector_fraction=float(weights[index]/total_connected_weight)if total_connected_weight>0.0 else 0.0
 return(index,float(amplitude.real),target_projector_fraction)
def _dominant_epsilon_state(psi0:np.ndarray,evecs_even:np.ndarray,Oepsilon_even:sp.csr_matrix)->tuple[int,float,float]:
 raw_source=Oepsilon_even@psi0
 source=raw_source-np.vdot(psi0,raw_source)*psi0
 amplitudes=np.asarray([np.vdot(evecs_even[:,n],source)for n in range(evecs_even.shape[1])],dtype=np.complex128)
 weights=np.abs(amplitudes)**2
 candidates=range(1,evecs_even.shape[1])
 index=int(max(candidates,key=lambda n:weights[n]))
 amplitude=np.vdot(evecs_even[:,index],raw_source)
 if amplitude.real<0.0:
  evecs_even[:,index]*=-1.0
  amplitude=-amplitude
 if abs(amplitude.imag)>1e-09:
  raise RuntimeError(f'epsilon matrix element is unexpectedly complex: {amplitude}')
 total_connected_weight=float(np.vdot(source,source).real)
 target_projector_fraction=float(weights[index]/total_connected_weight)if total_connected_weight>0.0 else 0.0
 return(index,float(amplitude.real),target_projector_fraction)
def build_low_energy_model(L:int,J:float,h:float,n_low:int,full_spectrum_max_L:int,dense_cutoff_dim:int,eigsh_tol:float,*,audit=None)->LowEnergyModel:
 reps,orbit_index,mask=parity_representatives(L)
 H_even=build_tfim_sector(L,+1,reps,orbit_index,mask,J,h)
 H_odd=build_tfim_sector(L,-1,reps,orbit_index,mask,J,h)
 complete=L<=full_spectrum_max_L
 evals_even,evecs_even,solver_even=eigenpairs(H_even,n_low,complete,dense_cutoff_dim,eigsh_tol)
 evals_odd,evecs_odd,solver_odd=eigenpairs(H_odd,n_low,complete,dense_cutoff_dim,eigsh_tol)
 evecs_even=np.array(evecs_even,dtype=np.complex128,copy=True)
 evecs_odd=np.array(evecs_odd,dtype=np.complex128,copy=True)
 if audit is not None:
  audit.extend([(H_even,evals_even,evecs_even,solver_even),(H_odd,evals_odd,evecs_odd,solver_odd)])
 E0=float(evals_even[0])
 psi0=evecs_even[:,0]
 osigma_diag=osigma_bridge_diagonal(L,reps)
 Oepsilon_even=build_oepsilon_sector(L,+1,reps,orbit_index,mask)
 Oepsilon_odd=build_oepsilon_sector(L,-1,reps,orbit_index,mask)
 sigma_odd,_,sigma_target_fraction=_dominant_sigma_state(psi0,evecs_odd,osigma_diag)
 epsilon_even,_,epsilon_target_fraction=_dominant_epsilon_state(psi0,evecs_even,Oepsilon_even)
 n_even_states=evecs_even.shape[1]
 n_odd_states=evecs_odd.shape[1]
 sigma_block=np.empty((n_even_states,n_odd_states),dtype=np.complex128)
 for m in range(n_even_states):
  sigma_block[m,:]=np.asarray([np.vdot(evecs_even[:,m],osigma_diag*evecs_odd[:,n])for n in range(n_odd_states)],dtype=np.complex128)
 O_sigma=np.zeros((n_even_states+n_odd_states,n_even_states+n_odd_states),dtype=np.complex128)
 O_sigma[:n_even_states,n_even_states:]=sigma_block
 O_sigma[n_even_states:,:n_even_states]=sigma_block.conj().T
 Oeps_even_low=evecs_even.conj().T@(Oepsilon_even@evecs_even)
 Oeps_odd_low=evecs_odd.conj().T@(Oepsilon_odd@evecs_odd)
 O_epsilon=np.zeros_like(O_sigma)
 O_epsilon[:n_even_states,:n_even_states]=Oeps_even_low
 O_epsilon[n_even_states:,n_even_states:]=Oeps_odd_low
 O_epsilon=O_epsilon-O_epsilon[0,0]*np.eye(O_epsilon.shape[0],dtype=np.complex128)
 sigma_index=n_even_states+sigma_odd
 epsilon_index=epsilon_even
 energies=np.concatenate([evals_even-E0,evals_odd-E0]).astype(np.float64)
 M_sigma=complex(O_sigma[sigma_index,0])
 M_epsilon=complex(O_epsilon[epsilon_index,0])
 B_epsilon_sigma=complex(O_sigma[epsilon_index,sigma_index])
 for name,value in[('M_sigma',M_sigma),('M_epsilon',M_epsilon),('B_epsilon_sigma',B_epsilon_sigma)]:
  if abs(value.imag)>1e-08:
   raise RuntimeError(f'{name} is unexpectedly complex: {value}')
 if M_sigma.real<=0.0 or M_epsilon.real<=0.0:
  raise RuntimeError(f'phase convention failed: M_sigma={M_sigma}, M_epsilon={M_epsilon}')
 R_direct=O_epsilon[0,epsilon_index]*O_sigma[epsilon_index,sigma_index]*O_sigma[sigma_index,0]
 denominator=M_epsilon.real*M_sigma.real*M_sigma.real
 C_direct=float((R_direct/denominator).real)
 gap_sigma=float(energies[sigma_index])
 gap_epsilon=float(energies[epsilon_index])
 velocity=2.0*J
 x_sigma=L*gap_sigma/(2.0*math.pi*velocity)
 x_epsilon=L*gap_epsilon/(2.0*math.pi*velocity)
 return LowEnergyModel(L=L,J=J,h=h,complete_spectrum=complete,sector_dimension=H_even.shape[0],n_even=n_even_states,n_odd=n_odd_states,solver_even=solver_even,solver_odd=solver_odd,energies=energies,O_sigma=O_sigma,O_epsilon=O_epsilon,sigma_index=sigma_index,epsilon_index=epsilon_index,gap_sigma=gap_sigma,gap_epsilon=gap_epsilon,M_sigma=float(M_sigma.real),M_epsilon=float(M_epsilon.real),B_epsilon_sigma=float(B_epsilon_sigma.real),R_direct=complex(R_direct),C_direct=C_direct,x_sigma=float(x_sigma),x_epsilon=float(x_epsilon),sigma_target_projector_fraction=float(sigma_target_fraction),epsilon_target_projector_fraction=float(epsilon_target_fraction))
def recalculate_state_diagnostics(lengths=(6,14),retained_states=24):
 rows=[]
 for length in lengths:
  captured=[]
  model=build_low_energy_model(length,1.0,1.0,retained_states,8,1024,1e-10,audit=captured)
  for label,sector,target,source in[('sigma',1,model.sigma_index-model.n_even,model.O_sigma[model.n_even:,0]),('epsilon',0,model.epsilon_index,model.O_epsilon[:model.n_even,0])]:
   H,energies,vectors,solver=captured[sector]
   weights=np.abs(source)**2
   if sector==0:
    weights[0]=0
   runner=next((int(i)for i in np.argsort(weights)[::-1]if i!=target))
   representatives,lookup,mask=parity_representatives(length)
   parity=1 if sector==0 else-1
   translated=np.zeros(vectors.shape[0],dtype=complex)
   for i,representative in enumerate(representatives):
    shifted=int(representative)<<1&mask|int(representative)>>length-1
    phase=1
    if shifted!=int(representatives[lookup[shifted]]):
     shifted=mask^shifted
     phase=parity
    translated[lookup[shifted]]+=phase*vectors[i,target]
   near=np.min(np.abs(energies[np.arange(len(energies))!=target]-energies[target]))
   residual=np.linalg.norm(H@vectors[:,target]-energies[target]*vectors[:,target])
   rows.append((length,label,parity,np.vdot(vectors[:,target],translated).real,weights[target],weights[runner],near,residual))
 return pd.DataFrame(rows,columns=['L','field','parity','translation_real','w1','w2','g_near','residual_norm']).sort_values(['L','parity'])
def calculate(lengths=LENGTHS,retained_states=RETAINED_STATES):
 return recalculate_state_diagnostics(lengths,retained_states)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
