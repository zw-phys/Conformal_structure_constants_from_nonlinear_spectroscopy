from math import gamma
import sys
from types import ModuleType
import numpy as np
import pandas as pd
try:
 from scipy.sparse import csr_matrix
 from scipy.sparse.linalg import eigsh
except ImportError as error:
 if'_spropack'not in str(error)or'__thread_bss'not in str(error):
  raise
 def unavailable_propack(*args,**kwargs):
  raise ImportError('The installed SciPy PROPACK extension could not load')
 svd_module=ModuleType('scipy.sparse.linalg._svdp')
 svd_module._svdp=unavailable_propack
 sys.modules[svd_module.__name__]=svd_module
 from scipy.sparse import csr_matrix
 from scipy.sparse.linalg import eigsh
LAMBDA_I=1.0
G_TCI=0.856
STATES_PER_PARITY=96
EIGEN_TOLERANCE=1e-11
RNG_SEED=2718
MEASURE_CYCLES=96
RAMP_CYCLES=6
STEPS_PER_CYCLE=256
NOMINAL_AREA=0.005
ETA=0.002
LAURENT_OFFSETS=range(-4,5)
LAURENT_DEGREE=4
C_CFT=np.sqrt(gamma(4/5)*gamma(2/5)**3/(gamma(1/5)*gamma(3/5)**3))
ORBIT_BATCH=1<<20
ROW_BATCH=1<<15
PROJECTION_COLUMNS=16
def rotate(words,shift,length,mask):
 return words<<shift&mask|words>>length-shift
def make_orbits(length):
 half=1<<length-1
 mask=2*half-1
 batches=[]
 for start in range(0,half,ORBIT_BATCH):
  words=np.arange(start,min(start+ORBIT_BATCH,half),dtype=np.uint32)
  for shift in range(1,length):
   shifted=rotate(words,shift,length,mask)
   words=words[(words<=shifted)&(words<=mask^shifted)]
  batches.append(words)
 words=np.concatenate(batches)
 labels=np.arange(1,len(words)+1,dtype=np.int32)
 lookup=np.empty(half,dtype=np.int32)
 period=np.full(len(words),length,dtype=np.int32)
 self_complement=np.zeros(len(words),dtype=bool)
 lookup[words]=labels
 shifted=words
 for shift in range(1,length):
  shifted=rotate(shifted,1,length,mask)
  period[(period==length)&(shifted==words)]=shift
  self_complement|=shifted==mask^words
  complemented=shifted>=half
  key=np.where(complemented,mask^shifted,shifted)
  lookup[key]=np.where(complemented,-labels,labels)
 orbit_size=np.where(self_complement,period,2*period)
 odd_allowed=~self_complement
 d1=length-2*np.bitwise_count(words^rotate(words,1,length,mask)).astype(int)
 magnetization=(length-2*np.bitwise_count(words).astype(int))/length
 return(words,lookup,orbit_size,odd_allowed,d1,magnetization)
def symmetry_block(length,orbits,parity,g=G_TCI):
 words,lookup,orbit_size,odd_allowed,d1,_=orbits
 half=1<<length-1
 mask=2*half-1
 active=np.arange(len(words),dtype=np.int32)
 if parity==-1:
  active=active[odd_allowed]
 dimension=len(active)
 sector_index=np.full(len(words),-1,dtype=np.int32)
 sector_index[active]=np.arange(dimension,dtype=np.int32)
 sites=np.arange(length,dtype=np.int32)
 flips=np.uint32(1)<<sites.astype(np.uint32)
 columns=np.empty((dimension,length+1),dtype=np.int32)
 h_values=np.empty(columns.shape,dtype=np.float64)
 q_values=np.empty(columns.shape,dtype=np.float64)
 for start in range(0,dimension,ROW_BATCH):
  stop=min(start+ROW_BATCH,dimension)
  source=active[start:stop]
  states=words[source,None]
  flipped=states^flips
  complemented=flipped>=half
  labels=lookup[np.where(complemented,mask^flipped,flipped)]
  target=np.abs(labels)-1
  col=sector_index[target]
  normalization=np.sqrt(orbit_size[source,None]/orbit_size[target])
  if parity==-1:
   normalization[(labels<0)!=complemented]*=-1
   normalization[col<0]=0
   col[col<0]=0
  right_pair=(states>>(sites+1)%length^states>>(sites+2)%length)&1
  left_pair=(states>>(sites-2)%length^states>>(sites-1)%length)&1
  dressing=2-2*(right_pair.astype(int)+left_pair.astype(int))
  columns[start:stop,:-1]=col
  columns[start:stop,-1]=np.arange(start,stop)
  h_values[start:stop,:-1]=LAMBDA_I*(-2+g*dressing)*normalization
  h_values[start:stop,-1]=-2*LAMBDA_I*d1[source]
  q_values[start:stop,:-1]=-normalization/length
  q_values[start:stop,-1]=d1[source]/length
 offsets=(length+1)*np.arange(dimension+1,dtype=np.int32)
 shape=(dimension,dimension)
 hamiltonian=csr_matrix((h_values.ravel(),columns.ravel(),offsets),shape=shape)
 readout=csr_matrix((q_values.ravel(),columns.ravel(),offsets),shape=shape)
 return(hamiltonian,readout)
def low_states(matrix,rng,states=STATES_PER_PARITY):
 dimension=matrix.shape[0]
 if dimension<=states:
  energy,vectors=np.linalg.eigh(matrix.toarray())
 else:
  energy,vectors=eigsh(matrix,k=states,which='SA',tol=EIGEN_TOLERANCE,ncv=min(dimension,max(32,2*states+1)),v0=rng.standard_normal(dimension))
  order=np.argsort(energy)
  energy,vectors=(energy[order],vectors[:,order])
 residual=max((np.linalg.norm(matrix@vectors[:,i]-energy[i]*vectors[:,i])for i in range(len(energy))))
 if not np.isfinite(residual)or residual>1e-07:
  raise ValueError(f'Invalid eigensolution: maximum residual {residual}')
 return(energy,vectors,residual)
def project_readout(operator,vectors):
 retained=vectors.shape[1]
 projected=np.empty((retained,retained))
 for start in range(0,retained,PROJECTION_COLUMNS):
  stop=min(start+PROJECTION_COLUMNS,retained)
  projected[:,start:stop]=vectors.T@(operator@vectors[:,start:stop])
 return projected
def projected_model(length,orbits,rng,*,g=G_TCI,states=STATES_PER_PARITY,project_extra=None):
 even_h,even_q=symmetry_block(length,orbits,1,g)
 dimension_even=even_h.shape[0]
 even_e,even_v,residual_even=low_states(even_h,rng,states)
 del even_h
 qe=project_readout(even_q,even_v)
 del even_q
 odd_h,odd_q=symmetry_block(length,orbits,-1,g)
 dimension_odd=odd_h.shape[0]
 odd_e,odd_v,residual_odd=low_states(odd_h,rng,states)
 del odd_h
 qo=project_readout(odd_q,odd_v)
 del odd_q
 _,_,_,odd_allowed,_,magnetization=orbits
 odd_rows=np.flatnonzero(odd_allowed)
 bridge=np.zeros((len(even_e),len(odd_e)))
 for start in range(0,len(odd_rows),ROW_BATCH):
  stop=min(start+ROW_BATCH,len(odd_rows))
  rows=odd_rows[start:stop]
  bridge+=even_v[rows].T@(magnetization[rows,None]*odd_v[start:stop])
 extra=None if project_extra is None else project_extra(length,orbits,even_v,odd_v)
 del even_v,odd_v
 sigma_index=int(np.argmax(abs(bridge[0])**2))
 epsilon_index=1+int(np.argmax(abs(qe[1:,0])**2))
 sign_even=np.ones(len(even_e))
 sign_odd=np.ones(len(odd_e))
 sign_even[epsilon_index]=np.sign(qe[epsilon_index,0])
 sign_odd[sigma_index]=np.sign(bridge[0,sigma_index])
 bridge*=sign_even[:,None]*sign_odd[None,:]
 qe*=sign_even[:,None]*sign_even[None,:]
 qo*=sign_odd[:,None]*sign_odd[None,:]
 ne,no=(len(even_e),len(odd_e))
 energy=np.concatenate((even_e,odd_e))-even_e[0]
 spin=np.zeros((ne+no,ne+no),dtype=np.complex128)
 spin[:ne,ne:]=bridge
 spin[ne:,:ne]=bridge.T
 readout=np.zeros_like(spin)
 readout[:ne,:ne]=qe
 readout[ne:,ne:]=qo
 readout-=qe[0,0]*np.eye(ne+no)
 gap_sigma=energy[ne+sigma_index]
 gap_epsilon=energy[epsilon_index]
 m_sigma=bridge[0,sigma_index]
 m_epsilon=qe[epsilon_index,0]
 if not(0<gap_sigma<gap_epsilon and m_sigma>0 and(m_epsilon>0)):
  raise ValueError(f'Invalid target transitions at L={length}')
 static=dict(ground_energy=even_e[0],gap_sigma=gap_sigma,gap_epsilon=gap_epsilon,gap_ratio=gap_epsilon/gap_sigma,M_sigma=m_sigma,M_epsilon=m_epsilon,sigma_index_odd=sigma_index,epsilon_index_even=epsilon_index,B_epsilon_sigma=bridge[epsilon_index,sigma_index],C_direct=bridge[epsilon_index,sigma_index]/m_sigma,dimension_even=dimension_even,dimension_odd=dimension_odd,retained_even=ne,retained_odd=no,full_q0=int(ne==dimension_even and no==dimension_odd),spectral_cutoff_even=energy[ne-1],spectral_cutoff_odd=energy[-1],eigen_residual=max(residual_even,residual_odd))
 if extra is not None:
  return(energy,spin,readout,static,extra*sign_even[:,None]*sign_odd[None,:])
 return(energy,spin,readout,static)
def pulse_derivative(state,reference,phase,field,first_phase,second_phase,spin,m_sigma):
 derivative=-1j*field*phase.conj()[:,None]*(spin@(phase[:,None]*state))
 reference_derivative=np.empty_like(reference)
 reference_derivative[0]=-1j*m_sigma*field*first_phase
 reference_derivative[1]=-1j*m_sigma*field*second_phase*reference[0]
 return(derivative,reference_derivative)
def finite_pulse(energy,spin,readout,gap_sigma,gap_epsilon,m_sigma,m_epsilon,*,steps_per_cycle=STEPS_PER_CYCLE,cycles=MEASURE_CYCLES,ramp_cycles=RAMP_CYCLES,area=NOMINAL_AREA):
 period=2*np.pi/gap_epsilon
 dt=period/steps_per_cycle
 ramp_time=ramp_cycles*period
 measure_time=cycles*period
 total_time=ramp_time+measure_time
 drive_amplitude=area/total_time
 nsteps=(ramp_cycles+cycles)*steps_per_cycle
 first_sample=ramp_cycles*steps_per_cycle
 times=np.arange(2*nsteps+1)*(dt/2)
 envelope=np.ones_like(times)
 ramp=times<ramp_time
 envelope[ramp]=np.sin(np.pi*times[ramp]/(2*ramp_time))**2
 sign1=np.array([1.0,-1.0,1.0,-1.0])
 sign2=np.array([1.0,1.0,-1.0,-1.0])
 fields=drive_amplitude*envelope[:,None]*(np.cos(gap_sigma*times)[:,None]*sign1+np.cos((gap_epsilon-gap_sigma)*times)[:,None]*sign2)
 phases=np.exp(-1j*times[:,None]*energy)
 first_phases=np.exp(1j*gap_sigma*times)
 second_phases=np.exp(1j*(gap_epsilon-gap_sigma)*times)
 lock_phases=np.exp(-1j*gap_epsilon*times[::2])
 state=np.zeros((len(energy),4),dtype=np.complex128)
 state[0]=1
 reference=np.zeros((2,4),dtype=np.complex128)
 signal_sum=np.zeros(4,dtype=np.complex128)
 reference_sum=np.zeros(4,dtype=np.complex128)
 max_norm_error=0.0
 for step in range(nsteps+1):
  index=2*step
  if step>=first_sample:
   psi=phases[index,:,None]*state
   measured=np.sum(psi.conj()*(readout@psi),axis=0).real
   reference_readout=2*m_epsilon*(reference[1]*lock_phases[step]).real
   weight=0.5 if step in(first_sample,nsteps)else 1.0
   signal_sum+=weight*measured*lock_phases[step]
   reference_sum+=weight*reference_readout*lock_phases[step]
  if step==nsteps:
   break
  k1,r1=pulse_derivative(state,reference,phases[index],fields[index],first_phases[index],second_phases[index],spin,m_sigma)
  k2,r2=pulse_derivative(state+dt*k1/2,reference+dt*r1/2,phases[index+1],fields[index+1],first_phases[index+1],second_phases[index+1],spin,m_sigma)
  k3,r3=pulse_derivative(state+dt*k2/2,reference+dt*r2/2,phases[index+1],fields[index+1],first_phases[index+1],second_phases[index+1],spin,m_sigma)
  k4,r4=pulse_derivative(state+dt*k3,reference+dt*r3,phases[index+2],fields[index+2],first_phases[index+2],second_phases[index+2],spin,m_sigma)
  state+=dt*(k1+2*k2+2*k3+k4)/6
  reference+=dt*(r1+2*r2+2*r3+r4)/6
  norms=np.sum(abs(state)**2,axis=0)
  if not np.all(np.isfinite(norms))or np.any(norms<=0):
   raise FloatingPointError('Invalid pulse-state norm')
  max_norm_error=max(max_norm_error,float(np.max(abs(norms-1))))
  state/=np.sqrt(norms)
 weights=np.array([1.0,-1.0,-1.0,1.0])/4
 signal=2*dt/measure_time*(signal_sum@weights)
 kernel=2*dt/measure_time*(reference_sum@weights)
 if abs(kernel)<1e-18 or max_norm_error>1e-07:
  raise ValueError(f'Invalid pulse: |kernel|={abs(kernel)}, norm error={max_norm_error}')
 coefficient=signal/kernel
 if not np.all(np.isfinite([signal,kernel,coefficient])):
  raise ValueError('Nonfinite finite-pulse result')
 return dict(C_real=coefficient.real,C_imag=coefficient.imag,S_real=signal.real,S_imag=signal.imag,K_real=kernel.real,K_imag=kernel.imag,max_norm_error=max_norm_error,lambda_value=drive_amplitude,dt=dt,T_ramp=ramp_time,T_measure=measure_time,T_total=total_time,resolution=2*np.pi/measure_time,actual_envelope_area=drive_amplitude*(measure_time+ramp_time/2))
def scaled_lstsq(design,values):
 norms=np.linalg.norm(design,axis=0)
 coefficients,_,rank,_=np.linalg.lstsq(design/norms,values,rcond=None)
 if rank!=design.shape[1]:
  raise np.linalg.LinAlgError('Rank-deficient fit design')
 coefficients/=norms
 if not np.all(np.isfinite(coefficients)):
  raise FloatingPointError('Nonfinite fit coefficients')
 return coefficients
def frequency_residue(energy,spin,readout,ne,gap_sigma,gap_epsilon,m_sigma,m_epsilon):
 ee=energy[:ne]
 eo=energy[ne:]
 cross=spin[:ne,ne:]
 qe=readout[:ne,:ne]
 qo=readout[ne:,ne:]
 numerator1=qe[0,:,None]*cross*cross[0,None,:]
 numerator2=cross[0,:,None]*qo*cross[0,None,:]
 numerator4=cross[0,:,None]*cross.T*qe[None,:,0]
 rows,response=([],[])
 for x in LAURENT_OFFSETS:
  for y in LAURENT_OFFSETS:
   delta1,delta_sum=(x*ETA,y*ETA)
   omega1,omega_sum=(gap_sigma+delta1,gap_epsilon+delta_sum)
   omega2=omega_sum-omega1
   value=0j
   for omega_first in(omega1,omega2):
    odd_transition=omega_sum+eo[:,None]-eo[None,:]+1j*ETA
    value+=np.sum(numerator1/((omega_sum-ee[:,None]+1j*ETA)*(omega_first-eo[None,:]+1j*ETA)))
    value-=np.sum(numerator2/(odd_transition*(omega_first-eo[None,:]+1j*ETA)))
    value-=np.sum(numerator2/(odd_transition*(omega_first+eo[:,None]+1j*ETA)))
    value+=np.sum(numerator4/((omega_sum+ee[None,:]+1j*ETA)*(omega_first+eo[:,None]+1j*ETA)))
   d1,ds=(delta1+1j*ETA,delta_sum+1j*ETA)
   row=[1/(d1*ds)]
   row+=[delta_sum**q/d1 for q in range(LAURENT_DEGREE+1)]
   row+=[delta1**p/ds for p in range(LAURENT_DEGREE+1)]
   row+=[delta1**p*delta_sum**(degree-p)for degree in range(LAURENT_DEGREE+1)for p in range(degree+1)]
   rows.append(row)
   response.append(value)
 design,response=(np.asarray(rows),np.asarray(response))
 coefficients=scaled_lstsq(design,response)
 coefficient=coefficients[0]/(m_epsilon*m_sigma**2)
 relative_rms=np.linalg.norm(design@coefficients-response)/np.linalg.norm(response)
 return dict(C_frequency_real=coefficient.real,C_frequency_imag=coefficient.imag,residue_real=coefficients[0].real,residue_imag=coefficients[0].imag,frequency_fit_relative_rms=relative_rms)
def project_xz(length,orbits,even_v,odd_v):
 words,lookup,orbit_size,odd_allowed,_,_=orbits
 half=1<<length-1
 mask=2*half-1
 odd_index=np.full(len(words),-1,dtype=np.int32)
 odd_index[odd_allowed]=np.arange(odd_v.shape[0],dtype=np.int32)
 sites=np.arange(length,dtype=np.uint32)
 flips=np.uint32(1)<<sites
 left=(sites+length-1)%length
 right=(sites+1)%length
 links=np.zeros((even_v.shape[1],odd_v.shape[1]))
 for start in range(0,len(words),ROW_BATCH):
  stop=min(start+ROW_BATCH,len(words))
  source=words[start:stop,None]
  flipped=source^flips
  complemented=flipped>=half
  labels=lookup[np.where(complemented,mask^flipped,flipped)]
  target=abs(labels)-1
  col=odd_index[target]
  norm=np.sqrt(orbit_size[start:stop,None]/orbit_size[target])
  norm[(labels<0)!=complemented]*=-1
  norm[col<0]=0
  col[col<0]=0
  neighbors=(source>>left&1)+(source>>right&1)
  weights=(2.0-2.0*neighbors)*norm/length
  offsets=length*np.arange(stop-start+1,dtype=np.int32)
  block=csr_matrix((weights.ravel(),col.ravel(),offsets),shape=(stop-start,odd_v.shape[0]))
  links+=even_v[start:stop].T@(block@odd_v)
 return links
def project_zzz(length,orbits,even_v,odd_v):
 words,_,_,odd_allowed,_,_=orbits
 mask=(1<<length)-1
 products=words^rotate(words,1,length,mask)^rotate(words,2,length,mask)
 diagonal=(length-2*np.bitwise_count(products).astype(int))/length
 return even_v[odd_allowed].T@(diagonal[odd_allowed,None]*odd_v)
def build_model(length,g=G_TCI,states=STATES_PER_PARITY,auxiliary=None):
 project={None:None,'XZ':project_xz,'ZZZ':project_zzz}[auxiliary]
 return projected_model(length,make_orbits(length),np.random.default_rng(RNG_SEED),g=g,states=states,project_extra=project)
def probe_transitions(length,operator='XZ',g=G_TCI,states=8):
 energy,spin,readout,static,aux=build_model(length,g,states,operator)
 ne=static['retained_even']
 sigma,epsilon=(static['sigma_index_odd'],static['epsilon_index_even'])
 secondary=1
 if sigma!=0 or static['retained_odd']<=secondary:
  raise ValueError('The two lowest odd branches are required for probe calibration')
 secondary_sign=np.sign(spin[0,ne+secondary].real)
 return dict(L=length,g=g,gap_sigma=static['gap_sigma'],gap_epsilon=static['gap_epsilon'],sigma_prime_gap=energy[ne+secondary],m_z_sigma=spin[0,ne+sigma].real,m_z_sigma_prime=secondary_sign*spin[0,ne+secondary].real,m_xz_sigma=aux[0,sigma],m_xz_sigma_prime=secondary_sign*aux[0,secondary],middle_z=spin[epsilon,ne+sigma].real,middle_aux=aux[epsilon,sigma])
def probe_calibration(length=18,operator='XZ',g=G_TCI,states=8):
 row=probe_transitions(length,operator,g,states)
 if abs(row['m_xz_sigma_prime'])<1e-14:
  raise ValueError('The auxiliary probe cannot cancel the secondary transition')
 return-row['m_z_sigma_prime']/row['m_xz_sigma_prime']
def reference_series(lengths=range(6,27,2),g=G_TCI,states=96,cycles=96,ramp=6,area=0.005,steps=256):
 rows=[]
 for length in lengths:
  energy,spin,readout,static=build_model(length,g,states)
  pulse=finite_pulse(energy,spin,readout,static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'],steps_per_cycle=steps,cycles=cycles,ramp_cycles=ramp,area=area)
  frequency=frequency_residue(energy,spin,readout,static['retained_even'],static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'])
  rows.append(dict(L=length,g=g,**static,**pulse,**frequency))
 return pd.DataFrame(rows)
def derivative(state,reference,phase,field,first_phase,second_phase,spin,m_sigma):
 multiplier=-1j*phase.conj()[:,None]*field
 derivative_state=np.empty_like(state)
 derivative_state[0]=multiplier*(spin@(phase[:,None]*state[0]))
 derivative_state[1]=multiplier*spin[:,0,None]
 derivative_state[2]=multiplier*(spin@(phase[:,None]*state[1]))
 derivative_reference=np.empty_like(reference)
 derivative_reference[0]=-1j*m_sigma*field*first_phase
 derivative_reference[1]=-1j*m_sigma*field*second_phase*reference[0]
 return(derivative_state,derivative_reference)
def propagate(energy,spin,readout,gap_sigma,gap_epsilon,m_sigma,m_epsilon,cycles,ramp_cycles,area,steps):
 if cycles<=0 or ramp_cycles<0 or steps<16 or(area<=0):
  raise ValueError('Invalid pulse duration, grid or area')
 period=2*np.pi/gap_epsilon
 dt=period/steps
 ramp_time=ramp_cycles*period
 measure_time=cycles*period
 total_time=ramp_time+measure_time
 amplitude=area/total_time
 nsteps=(ramp_cycles+cycles)*steps
 first_sample=ramp_cycles*steps
 times=np.arange(2*nsteps+1)*(dt/2)
 envelope=np.ones_like(times)
 ramp=times<ramp_time
 envelope[ramp]=np.sin(np.pi*times[ramp]/(2*ramp_time))**2
 sign1=np.array([1.0,-1.0,1.0,-1.0])
 sign2=np.array([1.0,1.0,-1.0,-1.0])
 fields=amplitude*envelope[:,None]*(np.cos(gap_sigma*times)[:,None]*sign1+np.cos((gap_epsilon-gap_sigma)*times)[:,None]*sign2)
 phases=np.exp(-1j*times[:,None]*energy)
 first_phases=np.exp(1j*gap_sigma*times)
 second_phases=np.exp(1j*(gap_epsilon-gap_sigma)*times)
 lock_phases=np.exp(-1j*gap_epsilon*times[::2])
 state=np.zeros((3,len(energy),4),dtype=np.complex128)
 state[0,0]=1
 reference=np.zeros((2,4),dtype=np.complex128)
 locks=np.zeros((4,4),dtype=np.complex128)
 max_norm_error=0.0
 max_second_norm_error=0.0
 for step in range(nsteps+1):
  index=2*step
  if step>=first_sample:
   full,first,second=phases[index,:,None]*state
   full_readout=np.sum(full.conj()*(readout@full),axis=0).real
   second_interference=2*(readout[0]@second).real
   second_first_squared=np.sum(first.conj()*(readout@first),axis=0).real
   reference_readout=2*m_epsilon*(reference[1]*lock_phases[step]).real
   weight=0.5 if step in(first_sample,nsteps)else 1.0
   locks+=weight*lock_phases[step]*np.stack((full_readout,second_interference,second_first_squared,reference_readout))
  if step==nsteps:
   break
  k1,r1=derivative(state,reference,phases[index],fields[index],first_phases[index],second_phases[index],spin,m_sigma)
  k2,r2=derivative(state+dt*k1/2,reference+dt*r1/2,phases[index+1],fields[index+1],first_phases[index+1],second_phases[index+1],spin,m_sigma)
  k3,r3=derivative(state+dt*k2/2,reference+dt*r2/2,phases[index+1],fields[index+1],first_phases[index+1],second_phases[index+1],spin,m_sigma)
  k4,r4=derivative(state+dt*k3,reference+dt*r3,phases[index+2],fields[index+2],first_phases[index+2],second_phases[index+2],spin,m_sigma)
  state+=dt*(k1+2*k2+2*k3+k4)/6
  reference+=dt*(r1+2*r2+2*r3+r4)/6
  norms=np.sum(abs(state[0])**2,axis=0)
  if not np.all(np.isfinite(norms))or np.any(norms<=0):
   raise FloatingPointError('Invalid full evolution norm')
  max_norm_error=max(max_norm_error,float(max(abs(norms-1))))
  state[0]/=np.sqrt(norms)
  second_norm=2*state[2,0].real+np.sum(abs(state[1])**2,axis=0)
  max_second_norm_error=max(max_second_norm_error,float(max(abs(second_norm))))
 weights=np.array([1.0,-1.0,-1.0,1.0])/4
 signals=2*dt/measure_time*(locks@weights)
 if not np.all(np.isfinite(signals))or abs(signals[-1])<1e-18:
  raise ValueError('Invalid phase-cycled signals or reference kernel')
 if max_norm_error>1e-07 or max_second_norm_error>1e-10:
  raise ValueError(f'Invalid norms: max_norm_error={max_norm_error!r}, max_second_norm_error={max_second_norm_error!r}')
 return(signals,dict(dt=dt,T_ramp=ramp_time,T_measure=measure_time,T_total=total_time,lambda_value=amplitude,actual_envelope_area=amplitude*(measure_time+ramp_time/2),resolution=2*np.pi/measure_time,max_norm_error=max_norm_error,max_second_order_norm_error=max_second_norm_error))
def paired_series(lengths=range(6,27,2),g=G_TCI,states=96,cycles=384,ramp=6,area=0.0025,steps=256,calibration_length=18):
 alpha=probe_calibration(calibration_length,g=g)
 rows=[]
 for length in lengths:
  energy,bare,readout,static,aux=build_model(length,g,states,'XZ')
  ne=static['retained_even']
  sigma=ne+static['sigma_index_odd']
  epsilon=static['epsilon_index_even']
  improved=bare.copy()
  improved[:ne,ne:]+=alpha*aux
  improved[ne:,:ne]=improved[:ne,ne:].T
  for name,spin in(('bare',bare),('improved',improved)):
   m_sigma=spin[0,sigma].real
   middle=spin[epsilon,sigma].real
   if m_sigma<=0:
    raise ValueError('Nonpositive phase-fixed target calibration')
   signals,metadata=propagate(energy,spin,readout,static['gap_sigma'],static['gap_epsilon'],m_sigma,static['M_epsilon'],cycles,ramp,area,steps)
   full,interference,first_squared,kernel=signals
   second=interference+first_squared
   direct=middle/m_sigma
   coefficient=full/kernel
   background=second/kernel-direct
   higher=(full-second)/kernel
   if abs(coefficient-direct-background-higher)>1e-12:
    raise ValueError('Full response and perturbative decomposition disagree')
   row=dict(L=length,g=g,probe=name,cycles=cycles,ramp_cycles=ramp,nominal_area=area,steps_per_cycle=steps,alpha=alpha,retained_even=ne,retained_odd=static['retained_odd'],gap_sigma=static['gap_sigma'],gap_epsilon=static['gap_epsilon'],M_sigma=m_sigma,M_epsilon=static['M_epsilon'],B_epsilon_sigma=middle,C_direct=direct,**metadata)
   for key,value in(('S_full',full),('S_second',second),('K',kernel),('full',coefficient),('background',background),('higher',higher)):
    row[key+'_real'],row[key+'_imag']=(value.real,value.imag)
   rows.append(row)
 return pd.DataFrame(rows)
