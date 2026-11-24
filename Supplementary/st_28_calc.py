from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spla
OUTPUT=Path(__file__).with_name('st_28.csv')
Omega_over_2pi_MHz=6.0
Delta_over_2pi_MHz=10.2
V1_over_2pi_MHz=164.6
V2_over_2pi_MHz=3.06
T1_us=63.0
TWO_PI=2*np.pi
PARAMS={'Delta_over_Omega':Delta_over_2pi_MHz/Omega_over_2pi_MHz,'V1_over_Omega':V1_over_2pi_MHz/Omega_over_2pi_MHz,'V2_over_Omega':V2_over_2pi_MHz/Omega_over_2pi_MHz,'physical_scale_rad_per_us':TWO_PI*Omega_over_2pi_MHz}
L=14
N_LOW=14
AMPLITUDE_MHZ=0.5
PULSE_WIDTH_US=0.1
STEPS=360
N_SAMPLES=20000
SIGNS=np.array([1.0,-1.0,-1.0,1.0])
THETA=np.array([0.0,np.pi,np.pi/2,3*np.pi/2])
PROFILE_DATA=[dict(name='batch_static_conservative',T2=0.75,N=20000,amplitude_sigma=0.03,frequency_sigma=0.05,T2_sigma=0.15,T1_sigma=0.2,prep=0.1,prep_sigma=0.03,contrast=0.94,contrast_sigma=0.02,M_sigma_sigma=0.05),dict(name='batch_static_improved',T2=3.0,N=200000,amplitude_sigma=0.01,frequency_sigma=0.015,T2_sigma=0.08,T1_sigma=0.1,prep=0.04,prep_sigma=0.01,contrast=0.98,contrast_sigma=0.005,M_sigma_sigma=0.02)]
SAMPLES=N_SAMPLES
def deterministic_vector(n,phase=0.173):
 x=np.arange(n,dtype=float)
 vec=np.cos(phase*(x+1.0))+0.37*np.sin(0.619*phase*(x+2.0))
 norm=np.linalg.norm(vec)
 if norm==0.0:
  vec[0]=1.0
  norm=1.0
 return vec/norm
def target_projector_fraction(source,selected_weight):
 total_weight=float(np.vdot(source,source).real)
 return float(selected_weight/total_weight)if total_weight>0.0 else 0.0
def generate_blockade_basis(L):
 if L<2:
  raise ValueError('L must be at least 2')
 states=[]
 def recurse(site,previous_occupied,state):
  if site==L:
   states.append(state)
   return
  recurse(site+1,False,state)
  if not previous_occupied:
   recurse(site+1,True,state|1<<site)
 recurse(0,False,0)
 states.sort()
 return np.asarray(states,dtype=np.int64)
def build_open_rydberg_hamiltonian(L,params,boundary_condition='compensated',include_h2=True):
 if boundary_condition not in{'native','compensated'}:
  raise ValueError("boundary_condition must be 'native' or 'compensated'")
 states=generate_blockade_basis(L)
 index={int(state):i for i,state in enumerate(states)}
 dim=len(states)
 Delta=params['Delta_over_Omega']
 V1=params['V1_over_Omega']
 V2=params['V2_over_Omega']
 h2_coefficient=-1.0/(4.0*V1)
 rows=[]
 cols=[]
 data=[]
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
def lowest_eigensystem(H,n_low):
 dim=H.shape[0]
 if dim<=300 or n_low>=dim-1:
  energies,vectors=np.linalg.eigh(H.toarray())
  keep=min(n_low,dim)
  return(energies[:keep],vectors[:,:keep],'dense_eigh')
 k=min(n_low,dim-2)
 ncv=min(dim,max(4*k+12,64))
 energies,vectors=spla.eigsh(H,k=k,which='SA',tol=1e-11,maxiter=400000,ncv=ncv,v0=deterministic_vector(dim))
 order=np.argsort(energies)
 return(energies[order],vectors[:,order],'sparse_eigsh')
def bond_weights(L,window,bulk_fraction=0.56,edge_bonds=2):
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
def bond_operator_diagonals(states,L,window):
 weights=bond_weights(L,window)
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
def tracked_target_indices(energies,vectors,sigma_diagonal,epsilon_diagonal,L):
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
def assign_states_and_matrix_elements(energies,vectors_input,sigma_diagonal,epsilon_diagonal,fixed_sigma_index=None,fixed_epsilon_index=None,epsilon_candidate_mask=None):
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
 return{'vectors':vectors,'sigma_index':sigma_index,'epsilon_index':epsilon_index,'gap_sigma':float(energies[sigma_index]-energies[0]),'gap_epsilon':float(energies[epsilon_index]-energies[0]),'M_sigma':M_sigma,'M_epsilon':M_epsilon,'B':B,'C_direct':C_direct,'sigma_weight':float(sigma_weights[sigma_index]),'epsilon_weight':float(epsilon_weights[epsilon_index]),'sigma_target_fraction':target_projector_fraction(sigma_source,sigma_weights[sigma_index]),'epsilon_target_fraction':target_projector_fraction(epsilon_source,epsilon_weights[epsilon_index])}
def sparse_lowering_operator(states,index,site):
 rows=[]
 cols=[]
 data=[]
 for column,raw_state in enumerate(states):
  state=int(raw_state)
  if state>>site&1:
   target_state=state^1<<site
   rows.append(index[target_state])
   cols.append(column)
   data.append(1.0)
 return sp.coo_matrix((data,(rows,cols)),shape=(len(states),len(states)),dtype=float).tocsr()
def build_projected_device_model(L,params,n_low,boundary_condition='compensated',window='global'):
 states,index,H=build_open_rydberg_hamiltonian(L=L,params=params,boundary_condition=boundary_condition)
 energies,vectors,_=lowest_eigensystem(H,n_low=n_low)
 sigma_diagonal,epsilon_diagonal=bond_operator_diagonals(states,L,window)
 sigma_index,epsilon_index,epsilon_candidate_mask=tracked_target_indices(energies,vectors,sigma_diagonal,epsilon_diagonal,L)
 assigned=assign_states_and_matrix_elements(energies,vectors,sigma_diagonal,epsilon_diagonal,fixed_sigma_index=sigma_index,fixed_epsilon_index=epsilon_index,epsilon_candidate_mask=epsilon_candidate_mask)
 vectors=assigned['vectors']
 O_sigma=vectors.T@(sigma_diagonal[:,None]*vectors)
 O_epsilon=vectors.T@(epsilon_diagonal[:,None]*vectors)
 O_epsilon_connected=O_epsilon-np.eye(len(energies))*O_epsilon[0,0]
 local_n=[]
 local_b=[]
 for site in range(L):
  n_diagonal=np.asarray([float(int(state)>>site&1)for state in states],dtype=float)
  local_n.append(vectors.T@(n_diagonal[:,None]*vectors))
  lowering=sparse_lowering_operator(states,index,site)
  local_b.append(vectors.T@(lowering@vectors))
 return dict(L=L,blockade_dimension=len(states),n_low=len(energies),boundary_condition=boundary_condition,energies_over_Omega=energies-energies[0],O_sigma=np.asarray(O_sigma,dtype=complex),O_epsilon_connected=np.asarray(O_epsilon_connected,dtype=complex),local_n=tuple((np.asarray(op,dtype=complex)for op in local_n)),local_b=tuple((np.asarray(op,dtype=complex)for op in local_b)),sigma_index=assigned['sigma_index'],epsilon_index=assigned['epsilon_index'],gap_sigma_over_Omega=assigned['gap_sigma'],gap_epsilon_over_Omega=assigned['gap_epsilon'],M_sigma=assigned['M_sigma'],M_epsilon=assigned['M_epsilon'],B_epsilon_sigma_audit_only=assigned['B'],C_direct_audit_only=assigned['C_direct'],sigma_target_projector_fraction=assigned['sigma_target_fraction'],epsilon_target_projector_fraction=assigned['epsilon_target_fraction'])
def dissipator_apply(jump,rho):
 adjoint_product=jump.conj().T@jump
 return jump@rho@jump.conj().T-0.5*adjoint_product@rho-0.5*rho@adjoint_product
def coherence_decay_rate(jumps,left_index,right_index):
 if not jumps:
  return 0.0
 n=jumps[0].shape[0]
 rho=np.zeros((n,n),dtype=complex)
 rho[left_index,right_index]=1.0
 derivative=np.zeros_like(rho)
 for jump in jumps:
  derivative+=dissipator_apply(jump,rho)
 return-float(np.real(derivative[left_index,right_index]))
def build_jump_operators(model,T2_us,T1_us,dephasing_model='energy_pure'):
 jumps=[]
 if np.isfinite(T1_us)and T1_us>0.0:
  decay_scale=math.sqrt(1.0/T1_us)
  jumps.extend((decay_scale*op for op in model['local_b']))
 decay_rate=coherence_decay_rate(jumps,0,model['sigma_index'])
 if dephasing_model=='energy_pure':
  dephasing_operator=np.eye(model['n_low'],dtype=complex)
  dephasing_operator[0,0]=0.0
 elif dephasing_model=='number_white':
  dephasing_operator=np.sum(np.stack(model['local_n'],axis=0),axis=0)
 else:
  raise ValueError("dephasing_model must be 'energy_pure' or 'number_white'")
 unit_rate=coherence_decay_rate([dephasing_operator],0,model['sigma_index'])
 target_rate=0.0 if not np.isfinite(T2_us)else 1.0/T2_us
 dephasing_strength=0.0
 if target_rate>decay_rate and unit_rate>1e-14:
  dephasing_strength=(target_rate-decay_rate)/unit_rate
  jumps.append(math.sqrt(dephasing_strength)*dephasing_operator)
 achieved_rate=coherence_decay_rate(jumps,0,model['sigma_index'])
 return(jumps,{'decay_coherence_rate_per_us':decay_rate,'dephasing_unit_rate':unit_rate,'dephasing_strength':dephasing_strength,'achieved_coherence_rate_per_us':achieved_rate,'dephasing_model':dephasing_model})
def unit_ope_calibration_operator(model):
 operator=np.zeros_like(model['O_sigma'],dtype=complex)
 sigma=model['sigma_index']
 epsilon=model['epsilon_index']
 M=model['M_sigma']
 operator[0,sigma]=M
 operator[sigma,0]=M
 operator[sigma,epsilon]=M
 operator[epsilon,sigma]=M
 return operator
def sequential_phase_cycle(energies_rad_per_us,drive_operator,jumps,sigma_index,epsilon_index,amplitude_MHz,pulse_width_us,steps,delta_f1_MHz=0.0,delta_f2_MHz=0.0,extra_hamiltonian=None,amplitude1_relative=0.0,amplitude2_relative=0.0):
 n=len(energies_rad_per_us)
 if steps<40:
  raise ValueError('At least 40 time steps are required')
 t1=2.5*pulse_width_us
 t2=6.5*pulse_width_us
 total_time=9.0*pulse_width_us
 dt=total_time/steps
 omega1=energies_rad_per_us[sigma_index]+TWO_PI*delta_f1_MHz
 omega2=energies_rad_per_us[epsilon_index]-energies_rad_per_us[sigma_index]+TWO_PI*delta_f2_MHz
 omega_out=energies_rad_per_us[epsilon_index]
 amplitude=TWO_PI*amplitude_MHz
 H0=np.diag(energies_rad_per_us)
 if extra_hamiltonian is not None:
  H0=H0+extra_hamiltonian
 jump_products=[(jj,jj.conj().T@jj)for jj in jumps]
 phase_settings=((0.0,0.0,+1.0),(math.pi,0.0,-1.0),(0.0,math.pi,-1.0),(math.pi,math.pi,+1.0))
 states=np.zeros((n*n,4),dtype=complex)
 rho0=np.zeros((n,n),dtype=complex)
 rho0[0,0]=1.0
 rho0_vector=rho0.reshape(-1,order='F')
 for shot in range(4):
  states[:,shot]=rho0_vector
 max_trace_error=0.0
 max_excited_population=0.0
 def derivative(t,matrix):
  envelope1=math.exp(-((t-t1)/pulse_width_us)**2)
  envelope2=math.exp(-((t-t2)/pulse_width_us)**2)
  coefficients=np.asarray([amplitude*((1.0+amplitude1_relative)*envelope1*math.cos(omega1*t+phi1)+(1.0+amplitude2_relative)*envelope2*math.cos(omega2*t+phi2))for phi1,phi2,_ in phase_settings],dtype=float)
  pp=matrix.T.reshape((4,n,n),order='F')
  HH=H0[None,:,:]+coefficients[:,None,None]*drive_operator[None,:,:]
  out=-1j*(HH@pp-pp@HH)
  for jj,aa in jump_products:
   out+=jj@pp@jj.conj().T-0.5*(aa@pp+pp@aa)
  return out.reshape((4,n*n),order='F').T
 for step in range(steps):
  t=step*dt
  k1=derivative(t,states)
  k2=derivative(t+0.5*dt,states+0.5*dt*k1)
  k3=derivative(t+0.5*dt,states+0.5*dt*k2)
  k4=derivative(t+dt,states+dt*k3)
  states+=dt/6.0*(k1+2.0*k2+2.0*k3+k4)
  if step%max(1,steps//24)==0 or step==steps-1:
   for shot in range(4):
    rho=states[:,shot].reshape((n,n),order='F')
    trace_error=abs(np.trace(rho)-1.0)
    max_trace_error=max(max_trace_error,float(trace_error))
    excited=max(0.0,1.0-float(np.real(rho[0,0])))
    max_excited_population=max(max_excited_population,excited)
 phase_cycled_density=np.zeros((n,n),dtype=complex)
 min_final_eigenvalue=float('inf')
 for shot,(_,_,sign)in enumerate(phase_settings):
  rho=states[:,shot].reshape((n,n),order='F')
  rho_hermitian=0.5*(rho+rho.conj().T)
  min_final_eigenvalue=min(min_final_eigenvalue,float(np.min(np.linalg.eigvalsh(rho_hermitian))))
  phase_cycled_density+=sign*rho/4.0
 signal=2.0*phase_cycled_density[epsilon_index,0]
 return{'signal':complex(signal),'phase_cycle_population_background':float(np.real(phase_cycled_density[0,0]+phase_cycled_density[epsilon_index,epsilon_index])/2.0),'final_density_matrices':[states[:,i].reshape((n,n),order='F')for i in range(4)],'omega1':float(omega1),'omega2':float(omega2),'omega_out':float(omega_out),'total_time':float(total_time),'max_trace_error':max_trace_error,'max_excited_population':max_excited_population,'min_final_density_eigenvalue':min_final_eigenvalue}
def vec_coherence(mats,model):
 b=np.array([(x[0,0]+x[model['epsilon_index'],model['epsilon_index']]).real/2 for x in mats])
 s=np.array([2*x[model['epsilon_index'],0]for x in mats])
 return(b,s)
def exact_phase(model,profile,z=None,amplitude=AMPLITUDE_MHZ,model_name='energy_pure',calibration=False,msz=0.0,steps=STEPS,T1_override=None):
 m=model
 zz=np.zeros(6)if z is None else np.asarray(z)
 en=m['energies_over_Omega']*PARAMS['physical_scale_rad_per_us']
 TT=profile['T2']*max(0.05,1+profile['T2_sigma']*zz[4])
 T1=T1_us*max(0.05,1+profile['T1_sigma']*zz[5])if T1_override is None else T1_override
 jumps,rates=build_jump_operators(m,TT,T1,model_name)
 op=m['O_sigma']
 if calibration:
  op=unit_ope_calibration_operator(dict(m,M_sigma=m['M_sigma']*(1+profile['M_sigma_sigma']*msz)))
 result=sequential_phase_cycle(en,op,jumps,m['sigma_index'],m['epsilon_index'],amplitude,PULSE_WIDTH_US,steps,profile['frequency_sigma']*zz[2],profile['frequency_sigma']*zz[3],amplitude1_relative=profile['amplitude_sigma']*zz[0],amplitude2_relative=profile['amplitude_sigma']*zz[1])
 return result
def phase_signal(b,s):
 return np.sum(s*SIGNS,axis=-1)/4
def encode_phase(b,s):
 tt=np.clip(2*b,1e-12,1-1e-12)
 beta=np.log(tt/(1-tt))
 rr=s/(2*b)
 rad=np.abs(rr)
 uu=np.where(rad>1e-14,np.arctanh(np.clip(rad,0,1-1e-12))*rr/np.maximum(rad,1e-30),rr)
 return np.r_[beta,uu.real,uu.imag]
def decode_phase(coords):
 beta=coords[...,0:4]
 uu=coords[...,4:8]+1j*coords[...,8:12]
 rad=np.abs(uu)
 rr=np.tanh(rad)/np.maximum(rad,1e-30)*uu
 rr=np.where(rad>1e-14,rr,uu)
 b=0.5/(1+np.exp(-np.clip(beta,-700,700)))
 ss=2*b*rr
 probs=b[...,None]+0.5*np.real(ss[...,None]*np.exp(-1j*THETA))
 if np.min(probs)<-1e-12 or np.max(probs)>1+1e-12:
  raise RuntimeError('invalid analyzer probabilities')
 return(b,ss,np.clip(probs,0,1))
def surrogate_coords(z,data):
 z=np.asarray(z)
 a=np.asarray(data['linear'])
 b=np.asarray(data['quadratic'])
 co=np.asarray(data['base'])+z@a+z*z@b
 for key,v in data['mixed'].items():
  i,j=map(int,key.split(','))
  co+=z[...,i,None]*z[...,j,None]*np.asarray(v)
 return co
def build_surrogate(model,profile):
 nominal=exact_phase(model,profile)
 b,s=vec_coherence(nominal['final_density_matrices'],model)
 base=encode_phase(b,s)
 linear,quadratic=([],[])
 data={'profile':profile,'base':base,'mixed':{}}
 def evaluate(z):
  response=exact_phase(model,profile,z)
  background,coherence=vec_coherence(response['final_density_matrices'],model)
  return encode_phase(background,coherence)
 for i in range(6):
  z=np.zeros(6)
  z[i]=1
  plus,minus=(evaluate(z),evaluate(-z))
  linear.append((plus-minus)/2)
  quadratic.append((plus+minus-2*base)/2)
 data['linear'],data['quadratic']=(np.array(linear),np.array(quadratic))
 for i in range(6):
  for j in range(i+1,6):
   mixed=np.zeros(12)
   for a in(-1,1):
    for b in(-1,1):
     z=np.zeros(6)
     z[i],z[j]=(a,b)
     mixed+=a*b*evaluate(z)/4
   data['mixed'][f'{i},{j}']=mixed
 K0=exact_phase(model,profile,calibration=True)['signal']
 minus=exact_phase(model,profile,calibration=True,msz=-1)['signal']
 plus=exact_phase(model,profile,calibration=True,msz=1)['signal']
 lm,lp=(np.log(minus/K0),np.log(plus/K0))
 data['K_nominal']=[K0.real,K0.imag]
 linear_K,quadratic_K=((lp-lm)/2,(lp+lm)/2)
 data['K_m_linear']=[linear_K.real,linear_K.imag]
 data['K_m_quadratic']=[quadratic_K.real,quadratic_K.imag]
 center=nominal['signal']/K0
 data['C_nominal']=[center.real,center.imag]
 data['C_direct_audit_only']=model['C_direct_audit_only']
 return data
def effective_latents(profile,latent):
 z=np.array(latent,dtype=float,copy=True)
 for index,key in((4,'T2_sigma'),(5,'T1_sigma'),(8,'M_sigma_sigma')):
  z[...,index]=np.maximum(z[...,index],-0.95/profile[key])
 return z
def batch_quantities(data,latent,counts_N=None,rng=None):
 p=data['profile']
 z=effective_latents(p,latent)
 _,ss,ideal=decode_phase(surrogate_coords(z[...,:6],data))
 prep=np.clip(p['prep']+p['prep_sigma']*z[...,6],0,0.8)
 contrast=np.clip(p['contrast']+p['contrast_sigma']*z[...,7],0.05,1)
 population=(1-contrast[...,None,None])/2+contrast[...,None,None]*(1-prep[...,None,None])*ideal
 gain0=p['contrast']*(1-p['prep'])
 km=np.asarray(data['K_nominal'])
 K0=complex(*km)
 mm=z[...,8]
 K=K0*np.exp(complex(*data['K_m_linear'])*mm+complex(*data['K_m_quadratic'])*mm*mm)
 def reconstruct(v):
  raw=(np.sum(SIGNS*(v[...,0]-v[...,1]),axis=-1)+1j*np.sum(SIGNS*(v[...,2]-v[...,3]),axis=-1))/4
  return raw/(gain0*K)
 conditional=reconstruct(population)
 counts=None
 estimate=conditional
 if counts_N is not None:
  counts=rng.binomial(counts_N,population)
  estimate=reconstruct(counts/counts_N)
 vr=np.sum(population[...,0]*(1-population[...,0])+population[...,1]*(1-population[...,1]),axis=-1)/16
 vi=np.sum(population[...,2]*(1-population[...,2])+population[...,3]*(1-population[...,3]),axis=-1)/16
 if counts_N is not None:
  vr/=counts_N
  vi/=counts_N
 hh=1/(gain0*K)
 cvr=hh.real**2*vr+hh.imag**2*vi
 cvi=hh.imag**2*vr+hh.real**2*vi
 return(estimate,conditional,population,counts,cvr,cvi)
def calculate(length=L,retained_states=N_LOW,samples=SAMPLES):
 model=build_projected_device_model(length,PARAMS,retained_states)
 surrogates=[build_surrogate(model,profile)for profile in PROFILE_DATA]
 sources={'amplitude_offsets':[0,1],'frequency_offsets':[2,3],'T2_T1_uncertainty':[4,5],'preparation_probability':[6],'readout_contrast':[7],'linear_M_sigma_calibration':[8],'finite_binomial_counts':[]}
 joint,isolated=([],{})
 for profile_index,data in enumerate(surrogates):
  p=data['profile']
  rng=np.random.default_rng(2026101701+profile_index)
  latent=rng.normal(size=(samples,9))
  draws,_,probabilities,counts,_,_=batch_quantities(data,latent,p['N'],rng)
  assert np.all((probabilities>=0)&(probabilities<=1))
  assert np.all((counts>=0)&(counts<=p['N']))
  q025,median,q975=np.quantile(draws.real,[0.025,0.5,0.975])
  joint.append(dict(part='joint',setting=p['name'].removeprefix('batch_static_'),median_real=median,q025_real=q025,q975_real=q975,rms_imag=np.sqrt(np.mean(draws.imag**2)),half_width_95_real=(q975-q025)/2))
  if profile_index!=1:
   continue
  for index,(source,active)in enumerate(sources.items()):
   rng=np.random.default_rng(2026101801+100*profile_index+index)
   latent=np.zeros((samples,9))
   latent[:,active]=rng.normal(size=(samples,len(active)))
   draws,*_=batch_quantities(data,latent,p['N']if source=='finite_binomial_counts'else None,rng)
   q025,q975=np.quantile(draws.real,[0.025,0.975])
   isolated[source]=dict(part='isolated_improved',setting=source,half_width_95_real=(q975-q025)/2)
 order=['finite_binomial_counts','linear_M_sigma_calibration','frequency_offsets','T2_T1_uncertainty','amplitude_offsets','preparation_probability','readout_contrast']
 columns=['part','setting','median_real','q025_real','q975_real','rms_imag','half_width_95_real']
 return pd.DataFrame(joint+[isolated[source]for source in order]).reindex(columns=columns)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
