import csv
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
MODELS=(('Ising',0.0,1.0),('Retuned J2=0.05',0.05,1.08387382))
J1=1.0
STATES_PER_PARITY=64
EIGEN_TOLERANCE=1e-11
RNG_SEED=2718
MEASURE_CYCLES=768
RAMP_CYCLES=8
STEPS_PER_CYCLE=128
NOMINAL_AREA=0.01
ORBIT_BATCH=1<<20
ROW_BATCH=1<<15
OUTPUT=Path(__file__).with_name('Fig_3_a_result.csv')
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
 shifted=words
 lookup[words]=labels
 for shift in range(1,length):
  shifted=rotate(shifted,1,length,mask)
  period[(period==length)&(shifted==words)]=shift
  self_complement|=shifted==mask^words
  flipped=shifted>=half
  key=np.where(flipped,mask^shifted,shifted)
  lookup[key]=np.where(flipped,-labels,labels)
 orbit_size=np.where(self_complement,period,2*period)
 odd_allowed=~self_complement
 d1=length-2*np.bitwise_count(words^rotate(words,1,length,mask)).astype(int)
 d2=length-2*np.bitwise_count(words^rotate(words,2,length,mask)).astype(int)
 magnetization=(length-2*np.bitwise_count(words).astype(int))/length
 return(words,lookup,orbit_size,odd_allowed,d1,d2,magnetization)
def hamiltonian(length,j2,field,orbits,parity):
 words,lookup,orbit_size,odd_allowed,d1,d2,_=orbits
 half=1<<length-1
 mask=2*half-1
 active=np.arange(len(words),dtype=np.int32)
 if parity==-1:
  active=active[odd_allowed]
 dimension=len(active)
 sector_index=np.full(len(words),-1,dtype=np.int32)
 sector_index[active]=np.arange(dimension,dtype=np.int32)
 flips=np.uint32(1)<<np.arange(length,dtype=np.uint32)
 columns=np.empty((dimension,length+1),dtype=np.int32)
 values=np.empty(columns.shape,dtype=np.float64)
 for start in range(0,dimension,ROW_BATCH):
  stop=min(start+ROW_BATCH,dimension)
  source=active[start:stop]
  flipped_words=words[source,None]^flips
  complemented=flipped_words>=half
  key=np.where(complemented,mask^flipped_words,flipped_words)
  labels=lookup[key]
  target=np.abs(labels)-1
  col=sector_index[target]
  amplitude=-field*np.sqrt(orbit_size[source,None]/orbit_size[target])
  if parity==-1:
   amplitude[(labels<0)!=complemented]*=-1
   amplitude[col<0]=0
   col[col<0]=0
  columns[start:stop,:-1]=col
  values[start:stop,:-1]=amplitude
  columns[start:stop,-1]=np.arange(start,stop)
  values[start:stop,-1]=-J1*d1[source]-j2*d2[source]
 offsets=(length+1)*np.arange(dimension+1,dtype=np.int32)
 return csr_matrix((values.ravel(),columns.ravel(),offsets),shape=(dimension,dimension))
def low_states(matrix,rng,states_per_parity=STATES_PER_PARITY):
 dimension=matrix.shape[0]
 if dimension<=states_per_parity:
  energy,vectors=np.linalg.eigh(matrix.toarray())
 else:
  energy,vectors=eigsh(matrix,k=states_per_parity,which='SA',tol=EIGEN_TOLERANCE,ncv=min(dimension,2*states_per_parity+1),v0=rng.standard_normal(dimension))
  order=np.argsort(energy)
  energy,vectors=(energy[order],vectors[:,order])
 residual=max((np.linalg.norm(matrix@vectors[:,i]-energy[i]*vectors[:,i])for i in range(len(energy))))
 if not np.isfinite(residual)or residual>1e-07:
  raise ValueError(f'Invalid eigensolution: maximum residual {residual}')
 return(energy,vectors,residual)
def projected_model(length,j2,field,orbits,rng,*,states_per_parity=STATES_PER_PARITY):
 even_h=hamiltonian(length,j2,field,orbits,1)
 dimension_even=even_h.shape[0]
 even_e,even_v,residual_even=low_states(even_h,rng,states_per_parity)
 del even_h
 odd_h=hamiltonian(length,j2,field,orbits,-1)
 dimension_odd=odd_h.shape[0]
 odd_e,odd_v,residual_odd=low_states(odd_h,rng,states_per_parity)
 del odd_h
 _,_,_,odd_allowed,d1,d2,magnetization=orbits
 bridge=even_v[odd_allowed].T@(magnetization[odd_allowed,None]*odd_v)
 diagonal=(1+J1/field)*d1+j2/field*d2
 qe=(even_v.T@(diagonal[:,None]*even_v)+np.diag(even_e/field))/length
 qo=(odd_v.T@(diagonal[odd_allowed,None]*odd_v)+np.diag(odd_e/field))/length
 del even_v,odd_v
 sign_even=np.ones(len(even_e))
 sign_odd=np.ones(len(odd_e))
 sign_even[1]=np.sign(qe[1,0])
 sign_odd[0]=np.sign(bridge[0,0])
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
 gap_sigma,gap_epsilon=(energy[ne],energy[1])
 m_sigma,m_epsilon=(bridge[0,0],qe[1,0])
 if not(0<gap_sigma<gap_epsilon and m_sigma>0 and(m_epsilon>0)):
  raise ValueError(f'Invalid target transitions at L={length}, J2={j2}')
 static=dict(ground_energy=even_e[0],gap_sigma=gap_sigma,gap_epsilon=gap_epsilon,M_sigma=m_sigma,M_epsilon=m_epsilon,B_epsilon_sigma=bridge[1,0],C_direct=bridge[1,0]/m_sigma,dimension_even=dimension_even,dimension_odd=dimension_odd,retained_even=ne,retained_odd=no,spectral_cutoff_even=energy[ne-1],spectral_cutoff_odd=energy[-1],eigen_residual=max(residual_even,residual_odd))
 return(energy,spin,readout,static)
def pulse_derivative(state,reference,phase,field,first_phase,second_phase,spin,m_sigma):
 derivative=-1j*field*phase.conj()[:,None]*(spin@(phase[:,None]*state))
 reference_derivative=np.empty_like(reference)
 reference_derivative[0]=-1j*m_sigma*field*first_phase
 reference_derivative[1]=-1j*m_sigma*field*second_phase*reference[0]
 return(derivative,reference_derivative)
def finite_pulse(energy,spin,readout,gap_sigma,gap_epsilon,m_sigma,m_epsilon,*,steps_per_cycle=STEPS_PER_CYCLE,measure_cycles=MEASURE_CYCLES,nominal_area=NOMINAL_AREA):
 period=2*np.pi/gap_epsilon
 dt=period/steps_per_cycle
 ramp_time=RAMP_CYCLES*period
 measure_time=measure_cycles*period
 total_time=ramp_time+measure_time
 drive_amplitude=nominal_area/total_time
 nsteps=(RAMP_CYCLES+measure_cycles)*steps_per_cycle
 first_sample=RAMP_CYCLES*steps_per_cycle
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
  max_norm_error=max(max_norm_error,float(np.max(abs(np.sum(abs(state)**2,axis=0)-1))))
  if step==nsteps:
   break
  k1,r1=pulse_derivative(state,reference,phases[index],fields[index],first_phases[index],second_phases[index],spin,m_sigma)
  k2,r2=pulse_derivative(state+dt*k1/2,reference+dt*r1/2,phases[index+1],fields[index+1],first_phases[index+1],second_phases[index+1],spin,m_sigma)
  k3,r3=pulse_derivative(state+dt*k2/2,reference+dt*r2/2,phases[index+1],fields[index+1],first_phases[index+1],second_phases[index+1],spin,m_sigma)
  k4,r4=pulse_derivative(state+dt*k3,reference+dt*r3,phases[index+2],fields[index+2],first_phases[index+2],second_phases[index+2],spin,m_sigma)
  state+=dt*(k1+2*k2+2*k3+k4)/6
  reference+=dt*(r1+2*r2+2*r3+r4)/6
 weights=np.array([1.0,-1.0,-1.0,1.0])/4
 signal=2*dt/measure_time*(signal_sum@weights)
 kernel=2*dt/measure_time*(reference_sum@weights)
 if abs(kernel)==0 or max_norm_error>1e-07:
  raise ValueError(f'Invalid pulse: |kernel|={abs(kernel)}, norm error={max_norm_error}')
 coefficient=signal/kernel
 if not np.all(np.isfinite([signal,kernel,coefficient])):
  raise ValueError('Nonfinite finite-pulse result')
 return dict(C_real=coefficient.real,C_imag=coefficient.imag,S_real=signal.real,S_imag=signal.imag,K_real=kernel.real,K_imag=kernel.imag,max_norm_error=max_norm_error,lambda_value=drive_amplitude,dt=dt,T_ramp=ramp_time,T_measure=measure_time,resolution=2*np.pi/measure_time,actual_envelope_area=drive_amplitude*(measure_time+ramp_time/2))
def main():
 rng=np.random.default_rng(RNG_SEED)
 writer=None
 with OUTPUT.open('w',newline='',encoding='utf-8')as output:
  for length in LENGTHS:
   orbit_start=perf_counter()
   orbits=make_orbits(length)
   orbit_seconds=perf_counter()-orbit_start
   for name,j2,field in MODELS:
    start=perf_counter()
    print(f'L={length}, {name}: sparse diagonalization',flush=True)
    energy,spin,readout,static=projected_model(length,j2,field,orbits,rng)
    ed_seconds=perf_counter()-start
    print(f"L={length}, {name}: {static['retained_even']}+{static['retained_odd']} states, ED {ed_seconds:.1f} s; finite pulse",flush=True)
    pulse_start=perf_counter()
    pulse=finite_pulse(energy,spin,readout,static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'])
    row=dict(series=name,L=length,J1=J1,J2=j2,h=field,cycles=MEASURE_CYCLES,ramp_cycles=RAMP_CYCLES,steps_per_cycle=STEPS_PER_CYCLE,area=NOMINAL_AREA,phi1=0.0,phi2=0.0,states_per_parity=STATES_PER_PARITY,**static,**pulse)
    row['error_real']=pulse['C_real']-static['C_direct']
    row['error_complex']=abs(complex(pulse['C_real'],pulse['C_imag'])-static['C_direct'])
    row['orbit_seconds']=orbit_seconds
    row['ed_seconds']=ed_seconds
    row['pulse_seconds']=perf_counter()-pulse_start
    row['elapsed_seconds']=perf_counter()-start
    if writer is None:
     writer=csv.DictWriter(output,fieldnames=list(row))
     writer.writeheader()
    writer.writerow(row)
    output.flush()
    print(f"L={length}, {name}: C_pulse={pulse['C_real']:.10f}{pulse['C_imag']:+.10f}j, C_direct={static['C_direct']:.10f}, elapsed {row['elapsed_seconds']:.1f} s",flush=True)
if __name__=='__main__':
 main()
