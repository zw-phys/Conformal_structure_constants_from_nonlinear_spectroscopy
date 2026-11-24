from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.linalg as sla
import scipy.linalg as la
from st_33_calc import PHASE_SHIFTS,PHASE_SIGNS,as_complex,complex_json,unit_kernel,extract,fit_linear_peak
OUTPUT=Path(__file__).with_name('st_32.csv')
@dataclass(frozen=True)
class ProjectedModel:
 L:int
 J1:float
 J2:float
 h:float
 n_even:int
 n_odd:int
 dim_even_sector:int
 dim_odd_sector:int
 complete_spectrum:bool
 solver_even:str
 solver_odd:str
 E0:float
 gap_sigma:float
 gap_epsilon:float
 gap_ratio:float
 sigma_index_odd:int
 epsilon_index_even:int
 M_sigma:float
 M_epsilon:float
 B_epsilon_sigma_audit_only:float
 C_direct_audit_only:float
 sigma_target_projector_fraction:float
 epsilon_target_projector_fraction:float
 energies:np.ndarray
 O_sigma:np.ndarray
 O_epsilon_connected:np.ndarray
def bit_z(state:int,site:int)->int:
 return 1 if state>>site&1==0 else-1
def parity_representatives(L:int)->tuple[np.ndarray,dict[int,int],int]:
 if L<2:
  raise ValueError('L must be at least 2')
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
def zz_average(state:int,L:int,distance:int)->float:
 return sum((bit_z(state,r)*bit_z(state,(r+distance)%L)for r in range(L)))/float(L)
def target_index_and_sign(state:int,parity:int,index:dict[int,int],mask:int)->tuple[int,float]:
 flipped=state^mask
 rep=state if state<flipped else flipped
 sign=1.0 if state==rep else float(parity)
 return(index[rep],sign)
def build_x_average_sector(L:int,parity:int,reps:np.ndarray,index:dict[int,int],mask:int)->sp.csr_matrix:
 rows:list[int]=[]
 cols:list[int]=[]
 data:list[float]=[]
 amp=1.0/float(L)
 for col,state0 in enumerate(reps):
  state=int(state0)
  for r in range(L):
   row,sign=target_index_and_sign(state^1<<r,parity,index,mask)
   rows.append(row)
   cols.append(col)
   data.append(amp*sign)
 out=sp.coo_matrix((data,(rows,cols)),shape=(len(reps),len(reps)),dtype=np.float64).tocsr()
 out.sum_duplicates()
 return out
def build_hamiltonian_sector(L:int,parity:int,reps:np.ndarray,index:dict[int,int],mask:int,J1:float,J2:float,h:float)->sp.csr_matrix:
 diag=np.asarray([-J1*L*zz_average(int(state),L,1)-J2*L*zz_average(int(state),L,2)for state in reps],dtype=np.float64)
 x_sum=L*build_x_average_sector(L,parity,reps,index,mask)
 return(sp.diags(diag,format='csr')-h*x_sum).tocsr()
def build_oepsilon_standard_sector(L:int,parity:int,reps:np.ndarray,index:dict[int,int],mask:int)->sp.csr_matrix:
 zz_diag=np.asarray([zz_average(int(state),L,1)for state in reps],dtype=np.float64)
 x_avg=build_x_average_sector(L,parity,reps,index,mask)
 return(sp.diags(zz_diag,format='csr')-x_avg).tocsr()
def osigma_bridge_diag(L:int,reps:np.ndarray)->np.ndarray:
 return np.asarray([z_average(int(state),L)for state in reps],dtype=np.float64)
def low_eigenpairs(H:sp.csr_matrix,n_low:int,dense_cutoff_dim:int,eigsh_tol:float)->tuple[np.ndarray,np.ndarray,str,bool]:
 dim=H.shape[0]
 if n_low<2:
  raise ValueError('n_low must be at least 2')
 n=min(n_low,dim)
 if dim<=dense_cutoff_dim or n>=dim-1:
  evals,evecs=np.linalg.eigh(H.toarray())
  return(np.asarray(evals[:n],dtype=np.float64),np.asarray(evecs[:,:n],dtype=np.float64),'dense',n==dim)
 k=min(n,dim-2)
 grid=np.arange(dim,dtype=np.float64)
 v0=np.cos(0.137*grid)+np.sin(0.071*grid)
 v0/=np.linalg.norm(v0)
 evals,evecs=spla.eigsh(H,k=k,which='SA',tol=eigsh_tol,maxiter=300000,ncv=min(dim,max(4*k+1,24)),v0=v0)
 order=np.argsort(evals)
 return(np.asarray(evals[order],dtype=np.float64),np.asarray(evecs[:,order],dtype=np.float64),'eigsh',False)
def choose_sigma_and_fix_phase(psi0:np.ndarray,evecs_odd:np.ndarray,osigma_diag:np.ndarray)->tuple[int,float,float]:
 source=osigma_diag*psi0
 amps=np.asarray([np.vdot(evecs_odd[:,n],source)for n in range(evecs_odd.shape[1])],dtype=np.complex128)
 weights=np.abs(amps)**2
 idx=int(np.argmax(weights))
 amp=np.vdot(evecs_odd[:,idx],source)
 if amp.real<0.0:
  evecs_odd[:,idx]*=-1.0
  amp=-amp
 if abs(amp.imag)>1e-08:
  raise RuntimeError(f'unexpected complex sigma amplitude: {amp}')
 total_connected_weight=float(np.vdot(source,source).real)
 target_projector_fraction=float(weights[idx]/total_connected_weight)if total_connected_weight>0.0 else 0.0
 return(idx,float(amp.real),target_projector_fraction)
def choose_epsilon_and_fix_phase(psi0:np.ndarray,evecs_even:np.ndarray,Oepsilon_even:sp.csr_matrix)->tuple[int,float,float]:
 raw_source=Oepsilon_even@psi0
 connected_source=raw_source-np.vdot(psi0,raw_source)*psi0
 amps=np.asarray([np.vdot(evecs_even[:,n],connected_source)for n in range(evecs_even.shape[1])],dtype=np.complex128)
 weights=np.abs(amps)**2
 if evecs_even.shape[1]<=1:
  raise RuntimeError('at least one excited even state is required')
 idx=int(max(range(1,evecs_even.shape[1]),key=lambda n:weights[n]))
 amp=np.vdot(evecs_even[:,idx],raw_source)
 if amp.real<0.0:
  evecs_even[:,idx]*=-1.0
  amp=-amp
 if abs(amp.imag)>1e-08:
  raise RuntimeError(f'unexpected complex epsilon amplitude: {amp}')
 total_connected_weight=float(np.vdot(connected_source,connected_source).real)
 target_projector_fraction=float(weights[idx]/total_connected_weight)if total_connected_weight>0.0 else 0.0
 return(idx,float(amp.real),target_projector_fraction)
def build_projected_model(L:int,J1:float,J2:float,h:float,n_low:int,dense_cutoff_dim:int,eigsh_tol:float)->ProjectedModel:
 reps,index,mask=parity_representatives(L)
 H_even=build_hamiltonian_sector(L,+1,reps,index,mask,J1,J2,h)
 H_odd=build_hamiltonian_sector(L,-1,reps,index,mask,J1,J2,h)
 evals_even,evecs_even,solver_even,complete_even=low_eigenpairs(H_even,n_low,dense_cutoff_dim,eigsh_tol)
 evals_odd,evecs_odd,solver_odd,complete_odd=low_eigenpairs(H_odd,n_low,dense_cutoff_dim,eigsh_tol)
 evecs_even=np.array(evecs_even,copy=True)
 evecs_odd=np.array(evecs_odd,copy=True)
 E0=float(evals_even[0])
 psi0=evecs_even[:,0]
 osigma_diag=osigma_bridge_diag(L,reps)
 Oepsilon_even=build_oepsilon_standard_sector(L,+1,reps,index,mask)
 Oepsilon_odd=build_oepsilon_standard_sector(L,-1,reps,index,mask)
 sigma_idx,_,sigma_target_fraction=choose_sigma_and_fix_phase(psi0,evecs_odd,osigma_diag)
 epsilon_idx,_,epsilon_target_fraction=choose_epsilon_and_fix_phase(psi0,evecs_even,Oepsilon_even)
 psi0=evecs_even[:,0]
 M_sigma=float(np.vdot(evecs_odd[:,sigma_idx],osigma_diag*psi0).real)
 M_epsilon=float(np.vdot(evecs_even[:,epsilon_idx],Oepsilon_even@psi0).real)
 if M_sigma<=0.0 or M_epsilon<=0.0:
  raise RuntimeError('phase fixing failed to make linear amplitudes positive')
 Ne=evecs_even.shape[1]
 No=evecs_odd.shape[1]
 bridge=np.empty((Ne,No),dtype=np.complex128)
 for m in range(Ne):
  left=evecs_even[:,m]
  for n in range(No):
   bridge[m,n]=np.vdot(left,osigma_diag*evecs_odd[:,n])
 O_sigma=np.zeros((Ne+No,Ne+No),dtype=np.complex128)
 O_sigma[:Ne,Ne:]=bridge
 O_sigma[Ne:,:Ne]=bridge.conj().T
 epsilon_even_low=evecs_even.conj().T@(Oepsilon_even@evecs_even)
 epsilon_odd_low=evecs_odd.conj().T@(Oepsilon_odd@evecs_odd)
 O_epsilon=np.zeros((Ne+No,Ne+No),dtype=np.complex128)
 O_epsilon[:Ne,:Ne]=epsilon_even_low
 O_epsilon[Ne:,Ne:]=epsilon_odd_low
 O_epsilon_connected=O_epsilon-epsilon_even_low[0,0]*np.eye(Ne+No,dtype=np.complex128)
 middle=bridge[epsilon_idx,sigma_idx]
 if abs(middle.imag)>1e-08:
  raise RuntimeError(f'unexpected complex target middle matrix element: {middle}')
 B=float(middle.real)
 C_direct=float(B/M_sigma)
 gap_sigma=float(evals_odd[sigma_idx]-E0)
 gap_epsilon=float(evals_even[epsilon_idx]-E0)
 if gap_sigma<=0.0 or gap_epsilon<=gap_sigma:
  raise RuntimeError(f'invalid target gaps for L={L}: E_sigma={gap_sigma}, E_epsilon={gap_epsilon}')
 return ProjectedModel(L=L,J1=J1,J2=J2,h=h,n_even=Ne,n_odd=No,dim_even_sector=H_even.shape[0],dim_odd_sector=H_odd.shape[0],complete_spectrum=bool(complete_even and complete_odd),solver_even=solver_even,solver_odd=solver_odd,E0=E0,gap_sigma=gap_sigma,gap_epsilon=gap_epsilon,gap_ratio=float(gap_epsilon/gap_sigma),sigma_index_odd=sigma_idx,epsilon_index_even=epsilon_idx,M_sigma=M_sigma,M_epsilon=M_epsilon,B_epsilon_sigma_audit_only=B,C_direct_audit_only=C_direct,sigma_target_projector_fraction=sigma_target_fraction,epsilon_target_projector_fraction=epsilon_target_fraction,energies=np.concatenate([evals_even-E0,evals_odd-E0]).astype(np.float64),O_sigma=O_sigma,O_epsilon_connected=O_epsilon_connected)
@dataclass
class Q0Model:
 L:int
 J1:float
 J2:float
 h:float
 n_even:int
 n_odd:int
 dim_even_sector:int
 dim_odd_sector:int
 complete_spectrum:bool
 E0:float
 gap_sigma:float
 gap_epsilon:float
 M_sigma:float
 M_epsilon:float
 sigma_index_odd:int
 epsilon_index_even:int
 sigma_index:int
 epsilon_index:int
 energies:np.ndarray
 O_sigma:np.ndarray
 O_epsilon_connected:np.ndarray
 diagnostics:dict
def translation_projector(L:int,parity:int,reps=None,index=None,mask=None):
 if reps is None:
  reps,index,mask=parity_representatives(L)
 dim=len(reps)
 destinations=np.empty(dim,dtype=int)
 signs=np.empty(dim,dtype=float)
 for j,rep in enumerate(reps):
  r=int(rep)
  shifted=r<<1&mask|r>>L-1
  dest,sign=target_index_and_sign(shifted,parity,index,mask)
  destinations[j]=dest
  signs[j]=sign
 T=sp.csr_matrix((signs,(destinations,np.arange(dim))),shape=(dim,dim))
 visited=np.zeros(dim,dtype=bool)
 rows=[]
 cols=[]
 data=[]
 col=0
 for start in range(dim):
  if visited[start]:
   continue
  j=start
  sgn=1.0
  orbit=[]
  amps=[]
  while not visited[j]:
   visited[j]=True
   orbit.append(j)
   amps.append(sgn)
   sgn*=signs[j]
   j=destinations[j]
  if j!=start:
   raise RuntimeError('Translation cycle did not close')
  if sgn<0:
   continue
  norm=np.sqrt(len(orbit))
  rows.extend(orbit)
  cols.extend([col]*len(orbit))
  data.extend((a/norm for a in amps))
  col+=1
 Q=sp.csr_matrix((data,(rows,cols)),shape=(dim,col))
 return(Q,T)
def _eigenpairs(H,n_low):
 dim=H.shape[0]
 n=dim if n_low is None else min(int(n_low),dim)
 if dim<=256 or n>=dim-1:
  e,v=la.eigh(H.toarray(),check_finite=False)
  return(e[:n],v[:,:n])
 start=np.cos(np.arange(dim,dtype=float)*0.371)+np.sin(np.arange(dim,dtype=float)*0.193)
 e,v=sla.eigsh(H,k=n,which='SA',tol=1e-11,maxiter=300000,ncv=min(dim,max(4*n+1,32)),v0=start)
 order=np.argsort(e)
 return(e[order],v[:,order])
def build_q0_model(L:int,J1:float=1.0,J2:float=0.05,h:float=1.08385,n_low:int|None=None)->Q0Model:
 if L<4 or L%2:
  raise ValueError('Use an even periodic length L>=4')
 reps,index,mask=parity_representatives(L)
 Qe,Te=translation_projector(L,+1,reps,index,mask)
 Qo,To=translation_projector(L,-1,reps,index,mask)
 He=build_hamiltonian_sector(L,+1,reps,index,mask,J1,J2,h)
 Ho=build_hamiltonian_sector(L,-1,reps,index,mask,J1,J2,h)
 Oee=build_oepsilon_standard_sector(L,+1,reps,index,mask)
 Oeo=build_oepsilon_standard_sector(L,-1,reps,index,mask)
 Hz_e=(Qe.T@He@Qe).tocsr()
 Hz_o=(Qo.T@Ho@Qo).tocsr()
 eps_e=(Qe.T@Oee@Qe).tocsr()
 eps_o=(Qo.T@Oeo@Qo).tocsr()
 sigma_eo=(Qe.T@sp.diags(osigma_bridge_diag(L,reps))@Qo).tocsr()
 Ee,Ve=_eigenpairs(Hz_e,n_low)
 Eo,Vo=_eigenpairs(Hz_o,n_low)
 if len(Ee)<2:
  raise ValueError('Need at least two even states')
 E0=float(Ee[0])
 ground=Ve[:,0]
 if Eo[0]<E0-1e-09:
  raise RuntimeError('Odd-sector energy below assumed even vacuum')
 sigma_source=sigma_eo.T@ground
 sigma_amplitudes=Vo.T@sigma_source
 si=int(np.argmax(abs(sigma_amplitudes)))
 if sigma_amplitudes[si]<0:
  Vo[:,si]*=-1
 epsilon_raw=eps_e@ground
 epsilon_source=epsilon_raw-ground*np.dot(ground,epsilon_raw)
 epsilon_amplitudes=Ve.T@epsilon_source
 epsilon_amplitudes[0]=0
 ei=int(np.argmax(abs(epsilon_amplitudes)))
 if epsilon_amplitudes[ei]<0:
  Ve[:,ei]*=-1
 sigma_proj=Ve.T@(sigma_eo@Vo)
 eps_even=Ve.T@(eps_e@Ve)
 eps_odd=Vo.T@(eps_o@Vo)
 Ms=float(sigma_proj[0,si])
 Me=float(eps_even[ei,0])
 ne,no=(len(Ee),len(Eo))
 Os=np.zeros((ne+no,ne+no),dtype=complex)
 Os[:ne,ne:]=sigma_proj
 Os[ne:,:ne]=sigma_proj.T
 Oe=la.block_diag(eps_even,eps_odd).astype(complex)
 Oe-=eps_even[0,0]*np.eye(ne+no)
 er_e=np.linalg.norm(Hz_e@Ve-Ve*Ee,axis=0)
 er_o=np.linalg.norm(Hz_o@Vo-Vo*Eo,axis=0)
 snorm=float(np.vdot(sigma_source,sigma_source).real)
 enorm=float(np.vdot(epsilon_source,epsilon_source).real)
 diag={'q0_orthogonality_even':float(sla.norm(Qe.T@Qe-sp.eye(Qe.shape[1]))),'q0_orthogonality_odd':float(sla.norm(Qo.T@Qo-sp.eye(Qo.shape[1]))),'translation_residual_even':float(sla.norm(Te@Qe-Qe)),'translation_residual_odd':float(sla.norm(To@Qo-Qo)),'maximum_eigen_residual_even':float(max(er_e)),'maximum_eigen_residual_odd':float(max(er_o)),'target_eigen_residuals':[float(er_e[0]),float(er_o[si]),float(er_e[ei])],'eigen_orthogonality_even':float(np.linalg.norm(Ve.T@Ve-np.eye(ne))),'eigen_orthogonality_odd':float(np.linalg.norm(Vo.T@Vo-np.eye(no))),'source_coverage_sigma':float(np.sum(abs(Vo.T@sigma_source)**2)/snorm),'source_coverage_epsilon':float(np.sum(abs(Ve.T@epsilon_source)**2)/enorm),'target_fraction_sigma':Ms**2/snorm,'target_fraction_epsilon':Me**2/enorm,'cutoff_even':float(Ee[-1]-E0),'cutoff_odd':float(Eo[-1]-E0)}
 return Q0Model(L,J1,J2,h,ne,no,Qe.shape[1],Qo.shape[1],ne==Qe.shape[1]and no==Qo.shape[1],E0,float(Eo[si]-E0),float(Ee[ei]-E0),Ms,Me,si,ei,ne+si,ei,np.concatenate([Ee-E0,Eo-E0]),Os,Oe,diag)
def make_protocol(cal,cycles,area,steps):
 period=2*np.pi/cal['gap_epsilon']
 Tr=8*period
 Tm=cycles*period
 total=Tr+Tm
 return{'omega1':cal['gap_sigma'],'omega2':cal['gap_epsilon']-cal['gap_sigma'],'omega_lock':cal['gap_epsilon'],'lambda1':area/total,'lambda2':area/total,'phi1':0.0,'phi2':0.0,'T_ramp':Tr,'T_measure':Tm,'T_total':total,'n_steps':int(round((cycles+8)*steps)),'readout_gamma':0.0}
def propagate(model,p,drift=(0.0,0.0)):
 E=model.energies
 Os=model.O_sigma
 Oe=model.O_epsilon_connected
 n=int(p['n_steps'])
 t=np.linspace(0.0,p['T_total'],n+1)
 dt=t[1]
 states=np.zeros((len(E),4),dtype=complex)
 states[0]=1.0
 observations=np.empty((n+1,4),dtype=float)
 shifts=np.array(PHASE_SHIFTS)
 omega1=p['omega1']+drift[0]
 omega2=p['omega2']+drift[1]
 def rhs(time,state):
  env=1.0 if time>=p['T_ramp']else np.sin(np.pi*time/(2*p['T_ramp']))**2
  f=env*(p['lambda1']*np.cos(omega1*time+p['phi1']+shifts[:,0])+p['lambda2']*np.cos(omega2*time+p['phi2']+shifts[:,1]))
  phase=np.exp(-1j*E*time)
  return-1j*phase.conj()[:,None]*(Os@(phase[:,None]*state))*f[None,:]
 max_norm=0.0
 for k,time_now in enumerate(t):
  phase=np.exp(-1j*E*time_now)
  stateS=phase[:,None]*states
  observations[k]=np.sum(stateS.conj()*(Oe@stateS),axis=0).real
  if k==n:
   break
  k1=rhs(time_now,states)
  k2=rhs(time_now+dt/2,states+dt*k1/2)
  k3=rhs(time_now+dt/2,states+dt*k2/2)
  k4=rhs(time_now+dt,states+dt*k3)
  states+=dt*(k1+2*k2+2*k3+k4)/6
  norms=np.sum(abs(states)**2,axis=0)
  if not np.all(np.isfinite(norms))or np.min(norms)<=0:
   raise FloatingPointError('Invalid state norm')
  max_norm=max(max_norm,float(max(abs(norms-1.0))))
  states/=np.sqrt(norms)[None,:]
 use=t>=p['T_ramp']-1e-12
 weights=np.exp(-p['readout_gamma']*(t[use]-p['T_ramp']))
 locks=2*np.trapezoid(weights[:,None]*observations[use]*np.exp(-1j*p['omega_lock']*t[use])[:,None],x=t[use],axis=0)/np.trapezoid(weights,x=t[use])
 return(t,observations,locks,max_norm)
def calibrate_linear_spectrum(model):
 calibration={}
 for channel,operator,target in(('sigma',model.O_sigma,model.gap_sigma),('epsilon',model.O_epsilon_connected,model.gap_epsilon)):
  weights=abs(operator[:,0])**2
  keep=(model.energies>1e-10)&(weights>1e-12*max(weights))
  energies,weights=(model.energies[keep],weights[keep])
  separation=abs(energies-target)
  distinct=separation[separation>1e-07]
  separation=min(target,float(min(distinct))if len(distinct)else target)
  window,eta=(0.1*separation,0.002*separation)
  omega=np.linspace(target-window,target+window,4001)
  density=np.sum(weights[:,None]*eta/np.pi/((omega[None,:]-energies[:,None])**2+eta**2),axis=0)
  fitted=fit_linear_peak(omega,density,eta)
  calibration['gap_'+channel]=fitted['gap']
  calibration['M_'+channel]=fitted['M']
 return calibration
def compare_complete_and_zero_momentum():
 full=build_projected_model(6,1.0,0.05,1.08385,32,256,1e-10)
 zero=build_q0_model(6,1.0,0.05,1.08385,None)
 if len(full.energies)!=64 or len(zero.energies)!=14:
  raise AssertionError('The comparison must retain both complete spaces')
 for key in('q0_orthogonality_even','q0_orthogonality_odd','translation_residual_even','translation_residual_odd'):
  if zero.diagnostics[key]>1e-12:
   raise AssertionError('Invalid momentum-zero projector: '+key)
 calibration=calibrate_linear_spectrum(zero)
 protocol=make_protocol(calibration,96,0.02,128)
 _,_,full_locks,full_norm=propagate(full,protocol)
 _,_,zero_locks,zero_norm=propagate(zero,protocol)
 maximum=float(np.max(abs(full_locks-zero_locks)))
 relative=float(np.linalg.norm(full_locks-zero_locks)/np.linalg.norm(zero_locks))
 if not np.isfinite(maximum)or relative>1e-09:
  raise AssertionError('Full-space and zero-momentum dynamics disagree')
 return dict(max_complex_lock_difference=maximum,relative_four_lock_norm_difference=relative,full_locks=full_locks,q0_locks=zero_locks,full_parity_dimension=len(full.energies),q0_dimension=len(zero.energies),full_max_norm_error=full_norm,q0_max_norm_error=zero_norm,calibration=calibration,protocol=protocol)
def independent_kernel_check()->dict:
 calibration={'gap_sigma':0.73,'gap_epsilon':2.4,'M_sigma':0.6,'M_epsilon':0.35}
 protocol={'omega1':0.73,'omega2':1.67,'omega_lock':2.4,'lambda1':0.007,'lambda2':0.011,'phi1':0.31,'phi2':-0.43,'T_ramp':1.7,'T_measure':17.3,'T_total':19.0,'n_steps':4096,'readout_gamma':0.013}
 times=np.linspace(0,protocol['T_total'],protocol['n_steps']+1)
 phase_shifts=np.array(PHASE_SHIFTS)
 def rhs(time:float,amplitudes:np.ndarray)->np.ndarray:
  envelope=1.0 if time>=protocol['T_ramp']else np.sin(np.pi*time/(2*protocol['T_ramp']))**2
  drives=envelope*(protocol['lambda1']*np.cos(protocol['omega1']*time+protocol['phi1']+phase_shifts[:,0])+protocol['lambda2']*np.cos(protocol['omega2']*time+protocol['phi2']+phase_shifts[:,1]))
  first_amplitudes=amplitudes[:4]
  first_derivative=-1j*calibration['M_sigma']*drives*np.exp(1j*calibration['gap_sigma']*time)
  second_derivative=-1j*calibration['M_sigma']*drives*np.exp(1j*(calibration['gap_epsilon']-calibration['gap_sigma'])*time)*first_amplitudes
  return np.concatenate([first_derivative,second_derivative])
 solution=solve_ivp(rhs,[0,protocol['T_total']],np.zeros(8,complex),t_eval=times,method='DOP853',rtol=1e-12,atol=1e-14,max_step=0.01)
 if not solution.success:
  raise RuntimeError(solution.message)
 signal=2*calibration['M_epsilon']*np.real(solution.y[4:]*np.exp(-1j*calibration['gap_epsilon']*times)[None,:])
 keep=times>=protocol['T_ramp']-1e-12
 weights=np.exp(-protocol['readout_gamma']*(times[keep]-protocol['T_ramp']))
 locks=2*np.trapezoid(weights[None,:]*signal[:,keep]*np.exp(-1j*protocol['omega_lock']*times[keep])[None,:],x=times[keep],axis=1)/np.trapezoid(weights,x=times[keep])
 phase_cycled=PHASE_SIGNS@locks
 kernel=unit_kernel(calibration,protocol)
 return{'calibration':calibration,'protocol':protocol,'independent_phase_cycled_second_order_lock':{'real':float(phase_cycled.real),'imag':float(phase_cycled.imag)},'unit_kernel':{'real':float(kernel.real),'imag':float(kernel.imag)},'absolute_difference':float(abs(phase_cycled-kernel)),'relative_difference':float(abs(phase_cycled-kernel)/abs(kernel)),'method':'DOP853 independently integrates first- and second-order amplitudes for all four phases; same native lock quadrature; arbitrary nonzero carrier phases and weighted readout'}
def calculate():
 symmetry_check=compare_complete_and_zero_momentum()
 rows=[('full64_vs_q0_14_four_lock_change','NI L6',symmetry_check['max_complex_lock_difference'])]
 internal=[]
 model=build_q0_model(8,1.0,0.05,1.08385,None)
 calibration=calibrate_linear_spectrum(model)
 def acquire(cycles,area,steps):
  protocol=make_protocol(calibration,cycles,area,steps)
  _,_,locks,norm_error=propagate(model,protocol)
  if norm_error>1e-07:
   raise ValueError('Pulse norm defect exceeds tolerance')
  return dict(schema_version='1.0',case_id=f'NI_L8_N{cycles}_A{area}_s{steps}',calibration=calibration,protocol=protocol,measured_locks=[complex_json(z)for z in locks])
 for cycles,area in[(96,0.02),(384,0.005),(96,0.6)]:
  base=f'NI_L8_N{cycles}_A{area:.3f}_s128_'+('amplitude_scan'if area==0.6 else'nominal')
  refined=f'NI_L8_N{cycles}_A{area:.3f}_s256_'+('amplitude_refinement'if area==0.6 else'step_refinement')
  nominal=acquire(cycles,area,128)
  fine=acquire(cycles,area,256)
  nominal_value=as_complex(extract(nominal)['estimate'])
  refined_value=as_complex(extract(fine)['estimate'])
  rows.append(('native_grid_128_to_256_C_change',base,abs(nominal_value-refined_value)))
  if area!=0.6:
   higher=as_complex(extract(nominal,internal_refinement=4)['estimate'])
   internal.append(('internal_kernel_grid_2_to_4_C_change',base,abs(higher-nominal_value)))
 rows.extend(internal)
 checked=independent_kernel_check()
 rows.append(('independent_DOP853_kernel_absolute_change','fixed nonzero phases and raised ramp',checked['absolute_difference']))
 result=pd.DataFrame(rows,columns=['quantity','setting','max_complex_change'])
 result.loc[0,'full_dimension']=symmetry_check['full_parity_dimension']
 result.loc[0,'q0_dimension']=symmetry_check['q0_dimension']
 result.loc[0,'relative_lock_difference']=symmetry_check['relative_four_lock_norm_difference']
 for space in('full','q0'):
  for phase,lock in zip(('00','pi0','0pi','pipi'),symmetry_check[space+'_locks']):
   result.loc[0,f'{space}_lock_{phase}_real']=lock.real
   result.loc[0,f'{space}_lock_{phase}_imag']=lock.imag
 return result
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
