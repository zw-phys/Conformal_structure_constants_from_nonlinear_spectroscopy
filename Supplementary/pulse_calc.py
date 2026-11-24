import math
from itertools import groupby
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from st_1_calc import make_orbits,projected_model
def build_model(length,j2=0.05,field=1.08385,states_per_parity=64):
 orbits=make_orbits(int(length))
 energy,spin,readout,static=projected_model(int(length),j2,field,orbits,np.random.default_rng(2718),states_per_parity=states_per_parity)
 return SimpleNamespace(energies=energy,O_sigma=spin,O_epsilon_connected=readout,gap_sigma=static['gap_sigma'],gap_epsilon=static['gap_epsilon'],M_sigma=static['M_sigma'],M_epsilon=static['M_epsilon'],C_direct=static['C_direct'],static=static,length=int(length),j2=j2,field=field)
def forward_pulse(model,*,cycles=32,area=0.02,steps=128,ramp=8,phi1=0.0,phi2=0.0,di=0.0,do=0.0,gamma=0.0,quadrature_stride=1,envelope_scale=1.0,phase_offset=0.0,drift1=0.0,drift2=0.0):
 omega1=model.gap_sigma+di
 omega_lock=model.gap_epsilon+do
 omega2=omega_lock-omega1
 if min(omega1,omega2,omega_lock)<=0:
  raise ValueError('Drive and readout frequencies must be positive')
 if cycles<=0 or ramp<0 or steps<16 or(quadrature_stride<1):
  raise ValueError('Invalid pulse duration or integration settings')
 period=2*np.pi/omega_lock
 t_ramp,t_measure=(ramp*period,cycles*period)
 t_total=t_ramp+t_measure
 n_steps=max(4,round((cycles+ramp)*steps))
 dt,amplitude=(t_total/n_steps,area/t_total)
 energies=np.asarray(model.energies)
 state=np.zeros((len(energies),4),dtype=complex)
 state[0]=1
 sigma=np.zeros(4,dtype=complex)
 epsilon=np.zeros(4,dtype=complex)
 phases1=np.array([0,np.pi,0,np.pi])+phi1
 phases2=np.array([0,0,np.pi,np.pi])+phi2
 signs=np.array([1,-1,-1,1])/4
 times,signals,kernels=([],[],[])
 max_norm_error=0.0
 def rhs(t,psi,cs,ce):
  phase=np.exp(-1j*energies*t)
  envelope=1.0 if t_ramp<=0 or t>=t_ramp else np.sin(np.pi*t/(2*t_ramp))**2
  field=amplitude*envelope*(np.cos(omega1*t+phases1)+np.cos(omega2*t+phases2))
  actual_field=amplitude*envelope*envelope_scale*(np.cos((omega1+drift1)*t+phases1+phase_offset)+np.cos((omega2+drift2)*t+phases2))
  dpsi=-1j*phase.conj()[:,None]*(model.O_sigma@(phase[:,None]*psi))*actual_field
  dcs=-1j*field*model.M_sigma*np.exp(1j*model.gap_sigma*t)
  dce=-1j*field*model.M_sigma*np.exp(1j*(model.gap_epsilon-model.gap_sigma)*t)*cs
  return(dpsi,dcs,dce)
 for step in range(n_steps+1):
  t=step*dt
  if step>=round(ramp*steps):
   psi=np.exp(-1j*energies*t)[:,None]*state
   signals.append(np.sum(psi.conj()*(model.O_epsilon_connected@psi),axis=0).real@signs)
   kernels.append(2*model.M_epsilon*(np.exp(-1j*model.gap_epsilon*t)*epsilon).real@signs)
   times.append(t)
  if step==n_steps:
   break
  k1=rhs(t,state,sigma,epsilon)
  k2=rhs(t+dt/2,state+dt*k1[0]/2,sigma+dt*k1[1]/2,epsilon+dt*k1[2]/2)
  k3=rhs(t+dt/2,state+dt*k2[0]/2,sigma+dt*k2[1]/2,epsilon+dt*k2[2]/2)
  k4=rhs(t+dt,state+dt*k3[0],sigma+dt*k3[1],epsilon+dt*k3[2])
  state+=dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
  sigma+=dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
  epsilon+=dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
  norms=np.sum(abs(state)**2,axis=0)
  if not np.isfinite(norms).all()or np.any(norms<=0):
   raise FloatingPointError('Invalid propagated state norm')
  max_norm_error=max(max_norm_error,float(np.max(abs(norms-1))))
  state/=np.sqrt(norms)
 times=np.asarray(times)[::quadrature_stride]
 signal=np.asarray(signals)[::quadrature_stride]
 kernel=np.asarray(kernels)[::quadrature_stride]
 weights=np.exp(-gamma*(times-t_ramp))
 lockin=weights*np.exp(-1j*omega_lock*times)
 normalization=np.trapezoid(weights,times)
 numerator=2*np.trapezoid(signal*lockin,times)/normalization
 denominator=2*np.trapezoid(kernel*lockin,times)/normalization
 if abs(denominator)==0:
  raise ValueError('The unit-coefficient reference vanishes')
 return{'numerator':numerator,'denominator':denominator,'estimate':numerator/denominator,'max_norm_error':max_norm_error}
def pulse_scan(settings):
 def model_key(item):
  _,p=item
  return(int(p['L']),float(p['J2']),float(p['h']),int(p['n']),p.get('projection','q0'))
 rows=[]
 for key,group in groupby(sorted(enumerate(settings),key=model_key),key=model_key):
  length,j2,field,states,projection=key
  model=build_full_parity_model(length,j2,field,states)if projection=='full_parity'else build_model(length,j2,field,states)
  for index,settings_row in group:
   p=dict(settings_row)
   cycles=p.get('cycles',32)
   ramp=p.get('ramp',8)
   area=p.get('area',0.02)
   resolution=model.gap_epsilon/cycles
   di=p.get('intermediate_detuning_units',0.0)*resolution
   do=p.get('output_detuning_units',0.0)*resolution
   gamma=p.get('gamma_units',0.0)*resolution
   result=forward_pulse(model,cycles=cycles,area=area,ramp=ramp,steps=p.get('steps',128),phi1=p.get('phi1',0.0),phi2=p.get('phi2',0.0),di=di,do=do,gamma=gamma,quadrature_stride=p.get('quadrature_stride',1),envelope_scale=p.get('envelope_scale',1.0),phase_offset=p.get('phase_offset',0.0))
   period=2*np.pi/(model.gap_epsilon+do)
   duration=(ramp+cycles)*period
   numerator,denominator=(result['numerator'],result['denominator'])
   estimate=result['estimate']
   p.update(intermediate_detuning=di,output_detuning=do,gamma=gamma,T_measure=cycles*period,resolution=2*np.pi/(cycles*period),lambda_value=area/duration,actual_envelope_area=area*(cycles+ramp/2)/(cycles+ramp),C_direct=model.C_direct,C_real=estimate.real,C_imag=estimate.imag,error_real=estimate.real-model.C_direct,error_complex=abs(estimate-model.C_direct),numerator_real=numerator.real,numerator_imag=numerator.imag,denominator_real=denominator.real,denominator_imag=denominator.imag,max_norm_error=result['max_norm_error'],spectral_cutoff_even=model.static['spectral_cutoff_even'],spectral_cutoff_odd=model.static['spectral_cutoff_odd'],target_gap=model.gap_epsilon)
   rows.append((index,p))
 return pd.DataFrame((row for _,row in sorted(rows)))
def ising_bit_z(state,site):
 return 1 if state>>site&1==0 else-1
def ising_parity_representatives(L):
 mask=(1<<L)-1
 reps=[]
 index={}
 for s in range(1<<L):
  fs=s^mask
  if s<fs:
   index[s]=len(reps)
   index[fs]=len(reps)
   reps.append(s)
 return(np.asarray(reps,dtype=np.int64),index,mask)
def ising_z_avg(state,L):
 return sum((ising_bit_z(state,r)for r in range(L)))/float(L)
def ising_zz_avg(state,L):
 return sum((ising_bit_z(state,r)*ising_bit_z(state,(r+1)%L)for r in range(L)))/float(L)
def ising_target_index_and_sign(t,parity,index,mask):
 ft=t^mask
 rep=t if t<ft else ft
 sign=1.0 if t==rep else float(parity)
 return(index[rep],sign)
def ising_build_x_average_sector(L,parity,reps,index,mask):
 rows=[]
 cols=[]
 data=[]
 amp=1.0/float(L)
 for col,s0 in enumerate(reps):
  s=int(s0)
  for r in range(L):
   row,sign=ising_target_index_and_sign(s^1<<r,parity,index,mask)
   rows.append(row)
   cols.append(col)
   data.append(amp*sign)
 out=sp.coo_matrix((data,(rows,cols)),shape=(len(reps),len(reps)),dtype=np.float64).tocsr()
 out.sum_duplicates()
 return out
def ising_build_tfim_sector(L,parity,reps,index,mask,J,h):
 zz_diag=np.asarray([-J*L*ising_zz_avg(int(s),L)for s in reps],dtype=np.float64)
 x_sum=L*ising_build_x_average_sector(L,parity,reps,index,mask)
 return(sp.diags(zz_diag,format='csr')-h*x_sum).tocsr()
def ising_build_oepsilon_standard_sector(L,parity,reps,index,mask):
 zz_diag=np.asarray([ising_zz_avg(int(s),L)for s in reps],dtype=np.float64)
 x_avg=ising_build_x_average_sector(L,parity,reps,index,mask)
 return(sp.diags(zz_diag,format='csr')-x_avg).tocsr()
def ising_osigma_bridge_diag(L,reps):
 return np.asarray([ising_z_avg(int(s),L)for s in reps],dtype=np.float64)
def ising_low_eigenpairs(H,n_low,dense_cutoff_dim,eigsh_tol):
 dim=H.shape[0]
 n=min(max(2,n_low),dim)
 if dim<=dense_cutoff_dim or n>=dim-1:
  evals,evecs=np.linalg.eigh(H.toarray())
  return(np.asarray(evals[:n]),np.asarray(evecs[:,:n]),'dense')
 k=min(n,dim-2)
 grid=np.arange(dim,dtype=np.float64)
 v0=np.cos(0.137*grid)+np.sin(0.071*grid)
 v0_norm=np.linalg.norm(v0)
 if v0_norm>0:
  v0=v0/v0_norm
 evals,evecs=spla.eigsh(H,k=k,which='SA',tol=eigsh_tol,maxiter=300000,ncv=min(dim,max(4*k+1,24)),v0=v0)
 order=np.argsort(evals)
 return(np.asarray(evals[order]),np.asarray(evecs[:,order]),'eigsh')
def ising_choose_sigma_index_and_fix_phase(psi0,evecs_odd,osig_diag):
 source=osig_diag*psi0
 amps=np.asarray([np.vdot(evecs_odd[:,n],source)for n in range(evecs_odd.shape[1])])
 weights=np.abs(amps)**2
 idx=int(np.argmax(weights))
 amp=np.vdot(evecs_odd[:,idx],source)
 if amp.real<0.0:
  evecs_odd[:,idx]*=-1.0
  amp=-amp
 if abs(amp.imag)>1e-08:
  raise RuntimeError(f'unexpected complex sigma matrix element {amp}')
 total_connected_weight=float(np.vdot(source,source).real)
 target_projector_fraction=float(weights[idx]/total_connected_weight)if total_connected_weight>0.0 else 0.0
 return(idx,float(amp.real),target_projector_fraction)
def ising_choose_epsilon_index_and_fix_phase(psi0,evecs_even,Oeps_even):
 raw_source=Oeps_even@psi0
 ground_component=np.vdot(psi0,raw_source)
 source=raw_source-ground_component*psi0
 amps=np.asarray([np.vdot(evecs_even[:,n],source)for n in range(evecs_even.shape[1])])
 weights=np.abs(amps)**2
 idx=int(max(range(1,evecs_even.shape[1]),key=lambda n:weights[n]))
 amp=np.vdot(evecs_even[:,idx],raw_source)
 if amp.real<0.0:
  evecs_even[:,idx]*=-1.0
  amp=-amp
 if abs(amp.imag)>1e-08:
  raise RuntimeError(f'unexpected complex epsilon matrix element {amp}')
 total_connected_weight=float(np.vdot(source,source).real)
 target_projector_fraction=float(weights[idx]/total_connected_weight)if total_connected_weight>0.0 else 0.0
 return(idx,float(amp.real),target_projector_fraction)
def build_ising_model(L,J,h,n_low,dense_cutoff_dim,eigsh_tol):
 reps,index,mask=ising_parity_representatives(L)
 H_even=ising_build_tfim_sector(L,+1,reps,index,mask,J,h)
 H_odd=ising_build_tfim_sector(L,-1,reps,index,mask,J,h)
 evals_even,evecs_even,solver_even=ising_low_eigenpairs(H_even,n_low,dense_cutoff_dim,eigsh_tol)
 evals_odd,evecs_odd,solver_odd=ising_low_eigenpairs(H_odd,n_low,dense_cutoff_dim,eigsh_tol)
 evecs_even=np.array(evecs_even,copy=True)
 evecs_odd=np.array(evecs_odd,copy=True)
 E0=float(evals_even[0])
 psi0=evecs_even[:,0]
 osig_diag=ising_osigma_bridge_diag(L,reps)
 Oeps_even=ising_build_oepsilon_standard_sector(L,+1,reps,index,mask)
 Oeps_odd=ising_build_oepsilon_standard_sector(L,-1,reps,index,mask)
 sigma_idx,M_sigma,sigma_target_fraction=ising_choose_sigma_index_and_fix_phase(psi0,evecs_odd,osig_diag)
 epsilon_idx,M_epsilon,epsilon_target_fraction=ising_choose_epsilon_index_and_fix_phase(psi0,evecs_even,Oeps_even)
 psi0=evecs_even[:,0]
 M_sigma=float(np.vdot(evecs_odd[:,sigma_idx],osig_diag*psi0).real)
 M_epsilon=float(np.vdot(evecs_even[:,epsilon_idx],Oeps_even@psi0).real)
 Ne=evecs_even.shape[1]
 No=evecs_odd.shape[1]
 S_even_odd=np.empty((Ne,No),dtype=np.complex128)
 for m in range(Ne):
  left=evecs_even[:,m]
  for n in range(No):
   S_even_odd[m,n]=np.vdot(left,osig_diag*evecs_odd[:,n])
 O_sigma=np.zeros((Ne+No,Ne+No),dtype=np.complex128)
 O_sigma[:Ne,Ne:]=S_even_odd
 O_sigma[Ne:,:Ne]=S_even_odd.conj().T
 Eps_even_low=evecs_even.conj().T@(Oeps_even@evecs_even)
 Eps_odd_low=evecs_odd.conj().T@(Oeps_odd@evecs_odd)
 O_epsilon=np.zeros((Ne+No,Ne+No),dtype=np.complex128)
 O_epsilon[:Ne,:Ne]=Eps_even_low
 O_epsilon[Ne:,Ne:]=Eps_odd_low
 O_epsilon_connected=O_epsilon-Eps_even_low[0,0]*np.eye(Ne+No,dtype=np.complex128)
 B=float(S_even_odd[epsilon_idx,sigma_idx].real)
 if abs(S_even_odd[epsilon_idx,sigma_idx].imag)>1e-08:
  raise RuntimeError(f'unexpected complex target middle matrix element {S_even_odd[epsilon_idx, sigma_idx]}')
 gap_sigma=float(evals_odd[sigma_idx]-E0)
 gap_epsilon=float(evals_even[epsilon_idx]-E0)
 omega2=gap_epsilon-gap_sigma
 v=2.0*J
 x_sigma=L*gap_sigma/(2.0*math.pi*v)
 x_epsilon=L*gap_epsilon/(2.0*math.pi*v)
 return SimpleNamespace(L=L,J=J,h=h,n_low_even=Ne,n_low_odd=No,dim_even_sector=H_even.shape[0],dim_odd_sector=H_odd.shape[0],solver_even=solver_even,solver_odd=solver_odd,E0=E0,gap_sigma=gap_sigma,gap_epsilon=gap_epsilon,omega1=gap_sigma,omega2=omega2,omega_sum=gap_epsilon,x_sigma=float(x_sigma),x_epsilon=float(x_epsilon),sigma_index_odd=sigma_idx,epsilon_index_even=epsilon_idx,M_sigma=float(M_sigma),M_epsilon=float(M_epsilon),B_epsilon_sigma=B,C_hat_static=float(B/M_sigma),sigma_target_projector_fraction=float(sigma_target_fraction),epsilon_target_projector_fraction=float(epsilon_target_fraction),energies=np.concatenate([evals_even-E0,evals_odd-E0]).astype(np.float64),O_sigma=O_sigma,O_epsilon_connected=O_epsilon_connected)
def ising_make_time_grid(omega_sum,measurement_cycles,ramp_cycles,steps_per_cycle):
 period_sum=2.0*math.pi/omega_sum
 T_ramp=float(ramp_cycles)*period_sum
 T_measure=float(measurement_cycles)*period_sum
 T_total=T_ramp+T_measure
 dt_initial=period_sum/float(steps_per_cycle)
 n_steps=int(round(T_total/dt_initial))
 n_steps=max(n_steps,4)
 dt=T_total/n_steps
 return SimpleNamespace(omega_sum=float(omega_sum),period_sum=float(period_sum),ramp_cycles=float(ramp_cycles),measurement_cycles=float(measurement_cycles),steps_per_cycle=int(steps_per_cycle),T_ramp=float(T_ramp),T_measure=float(T_measure),T_total=float(T_total),dt=float(dt),n_steps=int(n_steps))
def ising_ramp_envelope(t,T_ramp):
 if T_ramp<=0.0 or t>=T_ramp:
  return 1.0
 return math.sin(0.5*math.pi*t/T_ramp)**2
def ising_drive_value(model,grid,t,lambda1,lambda2,phi1,phi2):
 env=ising_ramp_envelope(t,grid.T_ramp)
 return env*(lambda1*math.cos(model.omega1*t+phi1)+lambda2*math.cos(model.omega2*t+phi2))
def ising_lockin_full(model,grid,lambda1,lambda2,phi1,phi2,return_norm_error=False):
 energies=model.energies
 O_sigma=model.O_sigma
 O_epsilon=model.O_epsilon_connected
 dim=energies.shape[0]
 psi=np.zeros(dim,dtype=np.complex128)
 psi[0]=1.0+0j
 acc=0.0+0j
 prev_val=None
 prev_t=None
 def rhs(t,y):
  f=ising_drive_value(model,grid,t,lambda1,lambda2,phi1,phi2)
  return-1j*(energies*y+f*(O_sigma@y))
 for k in range(grid.n_steps+1):
  t=k*grid.dt
  signal=float(np.vdot(psi,O_epsilon@psi).real)
  if t>=grid.T_ramp:
   val=signal*np.exp(-1j*model.omega_sum*t)
   if prev_val is not None and prev_t is not None:
    acc+=0.5*(val+prev_val)*(t-prev_t)
   prev_val=complex(val)
   prev_t=t
  if k==grid.n_steps:
   break
  k1=rhs(t,psi)
  k2=rhs(t+0.5*grid.dt,psi+0.5*grid.dt*k1)
  k3=rhs(t+0.5*grid.dt,psi+0.5*grid.dt*k2)
  k4=rhs(t+grid.dt,psi+grid.dt*k3)
  psi=psi+grid.dt/6.0*(k1+2.0*k2+2.0*k3+k4)
  norm=np.linalg.norm(psi)
  if norm==0.0 or not np.isfinite(norm):
   raise FloatingPointError('time evolution produced a non-finite state')
  psi=psi/norm
 lock=2.0*acc/grid.T_measure
 norm_error=abs(float(np.vdot(psi,psi).real)-1.0)
 if return_norm_error:
  return(complex(lock),float(norm_error))
 return complex(lock)
def ising_phase_cycled_cross(lockin,model,grid,lambda1,lambda2,phi1,phi2):
 return(lockin(model,grid,lambda1,lambda2,phi1,phi2)-lockin(model,grid,lambda1,lambda2,phi1+math.pi,phi2)-lockin(model,grid,lambda1,lambda2,phi1,phi2+math.pi)+lockin(model,grid,lambda1,lambda2,phi1+math.pi,phi2+math.pi))/4.0
def parity_z_average(state,L):
 return sum((ising_bit_z(state,r)for r in range(L)))/float(L)
def parity_zz_average(state,L,distance):
 return sum((ising_bit_z(state,r)*ising_bit_z(state,(r+distance)%L)for r in range(L)))/float(L)
def parity_build_x_average_sector(L,parity,reps,index,mask):
 rows=[]
 cols=[]
 data=[]
 amp=1.0/float(L)
 for col,state0 in enumerate(reps):
  state=int(state0)
  for r in range(L):
   row,sign=ising_target_index_and_sign(state^1<<r,parity,index,mask)
   rows.append(row)
   cols.append(col)
   data.append(amp*sign)
 out=sp.coo_matrix((data,(rows,cols)),shape=(len(reps),len(reps)),dtype=np.float64).tocsr()
 out.sum_duplicates()
 return out
def parity_build_hamiltonian_sector(L,parity,reps,index,mask,J1,J2,h):
 diag=np.asarray([-J1*L*parity_zz_average(int(state),L,1)-J2*L*parity_zz_average(int(state),L,2)for state in reps],dtype=np.float64)
 x_sum=L*parity_build_x_average_sector(L,parity,reps,index,mask)
 return(sp.diags(diag,format='csr')-h*x_sum).tocsr()
def parity_build_oepsilon_standard_sector(L,parity,reps,index,mask):
 zz_diag=np.asarray([parity_zz_average(int(state),L,1)for state in reps],dtype=np.float64)
 x_avg=parity_build_x_average_sector(L,parity,reps,index,mask)
 return(sp.diags(zz_diag,format='csr')-x_avg).tocsr()
def parity_osigma_bridge_diag(L,reps):
 return np.asarray([parity_z_average(int(state),L)for state in reps],dtype=np.float64)
def parity_low_eigenpairs(H,n_low,dense_cutoff_dim,eigsh_tol):
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
def parity_choose_sigma_and_fix_phase(psi0,evecs_odd,osigma_diag):
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
def parity_choose_epsilon_and_fix_phase(psi0,evecs_even,Oepsilon_even):
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
def parity_build_projected_model(L,J1,J2,h,n_low,dense_cutoff_dim,eigsh_tol):
 reps,index,mask=ising_parity_representatives(L)
 H_even=parity_build_hamiltonian_sector(L,+1,reps,index,mask,J1,J2,h)
 H_odd=parity_build_hamiltonian_sector(L,-1,reps,index,mask,J1,J2,h)
 evals_even,evecs_even,solver_even,complete_even=parity_low_eigenpairs(H_even,n_low,dense_cutoff_dim,eigsh_tol)
 evals_odd,evecs_odd,solver_odd,complete_odd=parity_low_eigenpairs(H_odd,n_low,dense_cutoff_dim,eigsh_tol)
 evecs_even=np.array(evecs_even,copy=True)
 evecs_odd=np.array(evecs_odd,copy=True)
 E0=float(evals_even[0])
 psi0=evecs_even[:,0]
 osigma_diag=parity_osigma_bridge_diag(L,reps)
 Oepsilon_even=parity_build_oepsilon_standard_sector(L,+1,reps,index,mask)
 Oepsilon_odd=parity_build_oepsilon_standard_sector(L,-1,reps,index,mask)
 sigma_idx,_,sigma_target_fraction=parity_choose_sigma_and_fix_phase(psi0,evecs_odd,osigma_diag)
 epsilon_idx,_,epsilon_target_fraction=parity_choose_epsilon_and_fix_phase(psi0,evecs_even,Oepsilon_even)
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
 return SimpleNamespace(L=L,J1=J1,J2=J2,h=h,n_even=Ne,n_odd=No,dim_even_sector=H_even.shape[0],dim_odd_sector=H_odd.shape[0],complete_spectrum=bool(complete_even and complete_odd),solver_even=solver_even,solver_odd=solver_odd,E0=E0,gap_sigma=gap_sigma,gap_epsilon=gap_epsilon,gap_ratio=float(gap_epsilon/gap_sigma),sigma_index_odd=sigma_idx,epsilon_index_even=epsilon_idx,M_sigma=M_sigma,M_epsilon=M_epsilon,B_epsilon_sigma_audit_only=B,C_direct_audit_only=C_direct,sigma_target_projector_fraction=sigma_target_fraction,epsilon_target_projector_fraction=epsilon_target_fraction,energies=np.concatenate([evals_even-E0,evals_odd-E0]).astype(np.float64),O_sigma=O_sigma,O_epsilon_connected=O_epsilon_connected)
def build_full_parity_model(length,j2=0.05,field=1.0825,states_per_parity=20):
 model=parity_build_projected_model(length,1.0,j2,field,states_per_parity,256,1e-11)
 model.C_direct=model.C_direct_audit_only
 model.static=dict(spectral_cutoff_even=model.energies[model.n_even-1],spectral_cutoff_odd=model.energies[-1])
 model.length,model.j2,model.field=(length,j2,field)
 return model
