import csv
from math import gamma
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
import numpy as np
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
LENGTHS=range(6,27,2)
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
SIZE_MODELS=(('probe',(0.8,)),('probe_stress',(0.8,2.0)),('empirical',(1.0,)),('empirical_stress',(1.0,2.0)),('probe_descendant',(0.8,1.2)))
C_CFT=np.sqrt(gamma(4/5)*gamma(2/5)**3/(gamma(1/5)*gamma(3/5)**3))
ORBIT_BATCH=1<<20
ROW_BATCH=1<<15
PROJECTION_COLUMNS=16
OUTPUT=Path(__file__).with_name('Fig_3_b_result.csv')
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
def symmetry_block(length,orbits,parity):
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
  h_values[start:stop,:-1]=LAMBDA_I*(-2+G_TCI*dressing)*normalization
  h_values[start:stop,-1]=-2*LAMBDA_I*d1[source]
  q_values[start:stop,:-1]=-normalization/length
  q_values[start:stop,-1]=d1[source]/length
 offsets=(length+1)*np.arange(dimension+1,dtype=np.int32)
 shape=(dimension,dimension)
 hamiltonian=csr_matrix((h_values.ravel(),columns.ravel(),offsets),shape=shape)
 readout=csr_matrix((q_values.ravel(),columns.ravel(),offsets),shape=shape)
 return(hamiltonian,readout)
def low_states(matrix,rng):
 dimension=matrix.shape[0]
 if dimension<=STATES_PER_PARITY:
  energy,vectors=np.linalg.eigh(matrix.toarray())
 else:
  energy,vectors=eigsh(matrix,k=STATES_PER_PARITY,which='SA',tol=EIGEN_TOLERANCE,ncv=min(dimension,2*STATES_PER_PARITY+1),v0=rng.standard_normal(dimension))
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
def projected_model(length,orbits,rng):
 even_h,even_q=symmetry_block(length,orbits,1)
 dimension_even=even_h.shape[0]
 even_e,even_v,residual_even=low_states(even_h,rng)
 del even_h
 qe=project_readout(even_q,even_v)
 del even_q
 odd_h,odd_q=symmetry_block(length,orbits,-1)
 dimension_odd=odd_h.shape[0]
 odd_e,odd_v,residual_odd=low_states(odd_h,rng)
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
 return(energy,spin,readout,static)
def pulse_derivative(state,reference,phase,field,first_phase,second_phase,spin,m_sigma):
 derivative=-1j*field*phase.conj()[:,None]*(spin@(phase[:,None]*state))
 reference_derivative=np.empty_like(reference)
 reference_derivative[0]=-1j*m_sigma*field*first_phase
 reference_derivative[1]=-1j*m_sigma*field*second_phase*reference[0]
 return(derivative,reference_derivative)
def finite_pulse(energy,spin,readout,gap_sigma,gap_epsilon,m_sigma,m_epsilon):
 period=2*np.pi/gap_epsilon
 dt=period/STEPS_PER_CYCLE
 ramp_time=RAMP_CYCLES*period
 measure_time=MEASURE_CYCLES*period
 total_time=ramp_time+measure_time
 drive_amplitude=NOMINAL_AREA/total_time
 nsteps=(RAMP_CYCLES+MEASURE_CYCLES)*STEPS_PER_CYCLE
 first_sample=RAMP_CYCLES*STEPS_PER_CYCLE
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
def size_fits(rows):
 lengths=np.array([row['L']for row in rows],dtype=float)
 values=np.array([row['C_frequency_real']for row in rows])
 parameters={}
 predictions,intercepts=([],[])
 for name,powers in SIZE_MODELS:
  design=np.column_stack([np.ones(len(lengths))]+[lengths**(-p)for p in powers])
  coefficients=scaled_lstsq(design,values)
  parameters[f'{name}_C_inf']=coefficients[0]
  for index,(power,coefficient)in enumerate(zip(powers,coefficients[1:]),start=1):
   parameters[f'{name}_p{index}']=power
   parameters[f'{name}_a{index}']=coefficient
  predictions.append(design@coefficients)
  intercepts.append(coefficients[0])
 parameters['C_inf_lower']=min(intercepts)
 parameters['C_inf_upper']=max(intercepts)
 return(parameters,np.min(predictions,axis=0),np.max(predictions,axis=0))
def main():
 rng=np.random.default_rng(RNG_SEED)
 rows=[]
 writer=None
 with OUTPUT.open('w',newline='',encoding='utf-8')as output:
  for length in LENGTHS:
   start=perf_counter()
   print(f'L={length}: constructing q=0 orbits and sparse TCI blocks',flush=True)
   orbits=make_orbits(length)
   orbit_seconds=perf_counter()-start
   ed_start=perf_counter()
   energy,spin,readout,static=projected_model(length,orbits,rng)
   del orbits
   ed_seconds=perf_counter()-ed_start
   print(f"L={length}: {static['retained_even']}+{static['retained_odd']} states, ED {ed_seconds:.1f} s; finite pulse and frequency extraction",flush=True)
   pulse_start=perf_counter()
   pulse=finite_pulse(energy,spin,readout,static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'])
   pulse_seconds=perf_counter()-pulse_start
   frequency=frequency_residue(energy,spin,readout,static['retained_even'],static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'])
   row=dict(L=length,lambda_I=LAMBDA_I,g=G_TCI,cycles=MEASURE_CYCLES,ramp_cycles=RAMP_CYCLES,steps_per_cycle=STEPS_PER_CYCLE,area=NOMINAL_AREA,phi1=0.0,phi2=0.0,states_per_parity=STATES_PER_PARITY,eta=ETA,laurent_degree=LAURENT_DEGREE,C_cft=C_CFT,**static,**pulse,**frequency)
   row['error_real']=pulse['C_real']-static['C_direct']
   row['error_complex']=abs(complex(pulse['C_real'],pulse['C_imag'])-static['C_direct'])
   row['orbit_seconds']=orbit_seconds
   row['ed_seconds']=ed_seconds
   row['pulse_seconds']=pulse_seconds
   row['elapsed_seconds']=perf_counter()-start
   rows.append(row)
   if writer is None:
    writer=csv.DictWriter(output,fieldnames=list(row))
    writer.writeheader()
   writer.writerow(row)
   output.flush()
   print(f"L={length}: C_pulse={pulse['C_real']:.10f}{pulse['C_imag']:+.10f}j, C_direct={static['C_direct']:.10f}, elapsed {row['elapsed_seconds']:.1f} s",flush=True)
 parameters,lower,upper=size_fits(rows)
 for index,row in enumerate(rows):
  row.update(parameters)
  row['envelope_lower']=lower[index]
  row['envelope_upper']=upper[index]
 with OUTPUT.open('w',newline='',encoding='utf-8')as output:
  writer=csv.DictWriter(output,fieldnames=list(rows[0]))
  writer.writeheader()
  writer.writerows(rows)
 print(f"Completed {len(rows)} lengths; five-model C_inf range [{parameters['C_inf_lower']:.10f}, {parameters['C_inf_upper']:.10f}]",flush=True)
if __name__=='__main__':
 main()
