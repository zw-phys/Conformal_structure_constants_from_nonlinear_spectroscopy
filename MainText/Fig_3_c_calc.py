import os
for thread_variable in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[thread_variable]='1'
os.environ['NUMBA_THREADING_LAYER']='workqueue'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import csv
import ctypes
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import socket
import subprocess
from time import perf_counter
import numpy as np
from numba import njit,prange,set_num_threads
from scipy.integrate import cumulative_simpson,trapezoid
from scipy.integrate._ivp import dop853_coefficients as dop
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import least_squares
from scipy.sparse import block_diag,csr_matrix
from scipy.sparse.linalg import LinearOperator,eigsh
from threadpoolctl import threadpool_limits
LENGTH=26
J1=1.0
J2=0.05
FIELD=1.08387382
LOW_STATES=16
EIGEN_TOLERANCE=1e-12
SPECTRAL_KRYLOV_STEPS=384
SPECTRAL_POINTS=4001
RNG_SEED=2718
ORBIT_BATCH=1<<20
ROW_BATCH=1<<15
MEASURE_CYCLES=384
RAMP_CYCLES=8
ACQUISITION_STEPS=128
NOMINAL_AREA=0.005
KERNEL_REFINEMENT=2
TRAJECTORY_POINTS=4097
PREPARATION_RATIO=0.02
DRIVE_RATIOS=(0.02,0.01)
FOLLOWUP_PHASE=0.37
PHASE_SHIFTS=((0.0,0.0),(np.pi,0.0),(0.0,np.pi),(np.pi,np.pi))
RTOL=2e-10
ATOL=2e-12
CARRIER_STEPS=32
MAX_NORM_DRIFT=1e-07
OUTPUT=Path(__file__).with_name('Fig_3_c_result.csv')
@njit(parallel=True,cache=False)
def real_matvec(data,columns,offsets,vector):
 result=np.empty(vector.size,dtype=np.float64)
 for row in prange(vector.size):
  value=0.0
  for k in range(offsets[row],offsets[row+1]):
   value+=data[k]*vector[columns[k]]
  result[row]=value
 return result
def orbit_basis(length):
 half=1<<length-1
 mask=2*half-1
 representatives=[]
 for start in range(0,half,ORBIT_BATCH):
  words=np.arange(start,min(start+ORBIT_BATCH,half),dtype=np.uint32)
  for shift in range(1,length):
   translated=words<<shift&mask|words>>length-shift
   words=words[(words<=translated)&(words<=mask^translated)]
  representatives.append(words)
 words=np.concatenate(representatives)
 labels=np.arange(1,words.size+1,dtype=np.int32)
 lookup=np.empty(half,dtype=np.int32)
 period=np.full(words.size,length,dtype=np.int32)
 self_complement=np.zeros(words.size,dtype=bool)
 lookup[words]=labels
 for shift in range(1,length):
  translated=words<<shift&mask|words>>length-shift
  period[(period==length)&(translated==words)]=shift
  self_complement|=translated==mask^words
  complement=translated>=half
  lookup[np.where(complement,mask^translated,translated)]=np.where(complement,-labels,labels)
 size=np.where(self_complement,period,2*period)
 odd=np.flatnonzero(~self_complement).astype(np.int32)
 z=(length-2*np.bitwise_count(words).astype(int))/length
 translated1=words<<1&mask|words>>length-1
 translated2=words<<2&mask|words>>length-2
 zz1=length-2*np.bitwise_count(words^translated1).astype(int)
 zz2=length-2*np.bitwise_count(words^translated2).astype(int)
 return(words,lookup,size,odd,z,zz1,zz2)
def symmetry_hamiltonian(length,j1,j2,field,basis,parity):
 words,lookup,size,odd,_,zz1,zz2=basis
 half=1<<length-1
 mask=2*half-1
 active=np.arange(words.size,dtype=np.int32)if parity==1 else odd
 inverse=np.full(words.size,-1,dtype=np.int32)
 inverse[active]=np.arange(active.size,dtype=np.int32)
 flips=np.uint32(1)<<np.arange(length,dtype=np.uint32)
 columns=np.empty((active.size,length+1),dtype=np.int32)
 values=np.empty(columns.shape,dtype=np.float64)
 for start in range(0,active.size,ROW_BATCH):
  stop=min(start+ROW_BATCH,active.size)
  source=active[start:stop]
  flipped=words[source,None]^flips
  complement=flipped>=half
  signed_label=lookup[np.where(complement,mask^flipped,flipped)]
  target=np.abs(signed_label)-1
  destination=inverse[target]
  amplitude=-field*np.sqrt(size[source,None]/size[target])
  if parity==-1:
   amplitude[(signed_label<0)!=complement]*=-1
   amplitude[destination<0]=0
   destination[destination<0]=0
  columns[start:stop,:-1]=destination
  columns[start:stop,-1]=np.arange(start,stop)
  values[start:stop,:-1]=amplitude
  values[start:stop,-1]=-j1*zz1[source]-j2*zz2[source]
 offsets=(length+1)*np.arange(active.size+1,dtype=np.int32)
 matrix=csr_matrix((values.ravel(),columns.ravel(),offsets),shape=(active.size,active.size))
 matrix.sum_duplicates()
 matrix.eliminate_zeros()
 matrix.sort_indices()
 difference=matrix-matrix.T
 if difference.nnz and np.max(np.abs(difference.data))>2e-12:
  raise ValueError('The symmetry Hamiltonian is not Hermitian')
 return matrix
def sparse_low_states(matrix,seed):
 dimension=matrix.shape[0]
 if dimension<=LOW_STATES:
  energy,vectors=np.linalg.eigh(matrix.toarray())
 else:
  operator=LinearOperator(matrix.shape,dtype=np.float64,matvec=lambda v:real_matvec(matrix.data,matrix.indices,matrix.indptr,v))
  rng=np.random.default_rng(seed)
  energy,vectors=eigsh(operator,k=LOW_STATES,which='SA',tol=EIGEN_TOLERANCE,ncv=2*LOW_STATES+1,v0=rng.standard_normal(dimension))
  order=np.argsort(energy)
  energy,vectors=(energy[order],vectors[:,order])
 residual=max((np.linalg.norm(matrix@vectors[:,k]-energy[k]*vectors[:,k])for k in range(energy.size)))
 if not np.isfinite(residual)or residual>1e-08:
  raise ValueError(f'Invalid target eigensolution: residual={residual}')
 return(energy,vectors,residual)
def vacuum_spectral_measure(matrix,source,energy,vectors,ground_energy):
 amplitude=vectors.T@source
 residual_source=source-vectors@amplitude
 residual_source-=vectors@(vectors.T@residual_source)
 residual_weight=float(residual_source@residual_source)
 poles=energy-ground_energy
 weights=amplitude**2
 dimension=matrix.shape[0]
 count=min(SPECTRAL_KRYLOV_STEPS,dimension-vectors.shape[1])
 if count==0 or residual_weight<1e-28:
  return(poles,weights,0,residual_weight)
 krylov=np.empty((dimension,count),dtype=np.float64,order='F')
 diagonal=np.empty(count)
 off_diagonal=np.empty(max(count-1,0))
 krylov[:,0]=residual_source/np.sqrt(residual_weight)
 actual=count
 for step in range(count):
  vector=krylov[:,step]
  work=real_matvec(matrix.data,matrix.indices,matrix.indptr,vector)
  work-=ground_energy*vector
  diagonal[step]=vector@work
  for _ in range(2):
   work-=vectors@(vectors.T@work)
   previous=krylov[:,:step+1]
   work-=previous@(previous.T@work)
  beta=np.linalg.norm(work)
  if beta<1e-12:
   actual=step+1
   break
  if step+1<count:
   off_diagonal[step]=beta
   krylov[:,step+1]=work/beta
 extra_poles,extra_vectors=eigh_tridiagonal(diagonal[:actual],off_diagonal[:actual-1])
 extra_weights=residual_weight*extra_vectors[0]**2
 poles=np.concatenate((poles,extra_poles))
 weights=np.concatenate((weights,extra_weights))
 if abs(weights.sum()-source@source)>1e-10:
  raise ValueError('Linear spectral quadrature lost vacuum-source weight')
 return(poles,weights,actual,residual_weight)
def fit_spectral_channel(matrix,source,energy,vectors,ground_energy,target):
 poles,weights,count,residual_weight=vacuum_spectral_measure(matrix,source,energy,vectors,ground_energy)
 use=(poles>1e-10)&(weights>1e-12*weights.max())
 poles,weights=(poles[use],weights[use])
 separation=np.abs(poles-target)
 distinct=separation[separation>1e-07]
 distance=min(target,distinct.min()if distinct.size else target)
 half_width,linewidth=(0.1*distance,0.002*distance)
 frequency=np.linspace(target-half_width,target+half_width,SPECTRAL_POINTS)
 density=np.sum(weights[:,None]*linewidth/np.pi/((frequency[None,:]-poles[:,None])**2+linewidth**2),axis=0)
 fit=fit_linear_peak(frequency,density,linewidth)
 fit['krylov_steps']=count
 fit['remaining_source_weight']=residual_weight
 return fit
def build_model(length,j1,j2,field):
 started=perf_counter()
 basis=orbit_basis(length)
 even=symmetry_hamiltonian(length,j1,j2,field,basis,1)
 odd=symmetry_hamiltonian(length,j1,j2,field,basis,-1)
 print(f'Full q=0 basis: {even.shape[0]} even + {odd.shape[0]} odd; construction {perf_counter() - started:.1f} s',flush=True)
 even_energy,even_vectors,even_residual=sparse_low_states(even,RNG_SEED)
 odd_energy,odd_vectors,odd_residual=sparse_low_states(odd,RNG_SEED+1)
 ground_energy=even_energy[0]
 _,_,_,odd_rows,magnetization,zz1,zz2=basis
 ground=even_vectors[:,0]
 diagonal_even=((1+j1/field)*zz1+j2/field*zz2)/length
 connected_mean=ground@(diagonal_even*ground)
 diagonal_even-=connected_mean
 sigma_source=magnetization[odd_rows]*ground[odd_rows]
 epsilon_source=diagonal_even*ground+(even@ground-ground_energy*ground)/(field*length)
 sigma_overlap=odd_vectors.T@sigma_source
 epsilon_overlap=even_vectors.T@epsilon_source
 epsilon_overlap[0]=0
 sigma_index=int(np.argmax(np.abs(sigma_overlap)))
 epsilon_index=int(np.argmax(np.abs(epsilon_overlap)))
 odd_vectors[:,sigma_index]*=1 if sigma_overlap[sigma_index]>=0 else-1
 even_vectors[:,epsilon_index]*=1 if epsilon_overlap[epsilon_index]>=0 else-1
 gap_sigma=odd_energy[sigma_index]-ground_energy
 gap_epsilon=even_energy[epsilon_index]-ground_energy
 print(f'Target gaps identified; eigen residual={max(even_residual, odd_residual):.3e}; fitting independent vacuum spectra',flush=True)
 sigma_fit=fit_spectral_channel(odd,sigma_source,odd_energy,odd_vectors,ground_energy,gap_sigma)
 epsilon_fit=fit_spectral_channel(even,epsilon_source,even_energy,even_vectors,ground_energy,gap_epsilon)
 calibration=dict(gap_sigma=sigma_fit['gap'],M_sigma=sigma_fit['M'],gap_epsilon=epsilon_fit['gap'],M_epsilon=epsilon_fit['M'])
 ne,no=(even.shape[0],odd.shape[0])
 targets=np.zeros((3,ne+no))
 targets[0,:ne]=ground
 targets[1,ne:]=odd_vectors[:,sigma_index]
 targets[2,:ne]=even_vectors[:,epsilon_index]
 even.setdiag(even.diagonal()-ground_energy)
 odd.setdiag(odd.diagonal()-ground_energy)
 matrix=block_diag((even,odd),format='csr')
 partner=np.full(ne+no,-1,dtype=np.int32)
 partner[odd_rows]=ne+np.arange(no,dtype=np.int32)
 partner[ne:]=odd_rows
 bridge=np.zeros(ne+no)
 bridge[odd_rows]=magnetization[odd_rows]
 bridge[ne:]=magnetization[odd_rows]
 readout=np.concatenate((diagonal_even,diagonal_even[odd_rows]))
 model=dict(data=matrix.data,indices=matrix.indices,indptr=matrix.indptr,partner=partner,bridge=bridge,targets=targets,readout=readout,readout_h_factor=1/(field*length))
 metadata=dict(dimension_even=ne,dimension_odd=no,complete_q0_dimension=ne+no,ground_energy=ground_energy,exact_gap_sigma=gap_sigma,exact_gap_epsilon=gap_epsilon,maximum_eigen_residual=max(even_residual,odd_residual),sigma_linear_fit_rms=sigma_fit['relative_residual_rms'],epsilon_linear_fit_rms=epsilon_fit['relative_residual_rms'],sigma_spectral_krylov_steps=sigma_fit['krylov_steps'],epsilon_spectral_krylov_steps=epsilon_fit['krylov_steps'],linear_spectrum_method='full_source_deflated_Lanczos_quadrature',model_calibration_seconds=perf_counter()-started)
 return(model,calibration,metadata)
def fit_linear_peak(omega,density,linewidth):
 omega=np.asarray(omega,dtype=float)
 density=np.asarray(density,dtype=float)
 if omega.ndim!=1 or density.shape!=omega.shape or omega.size<20 or(not np.all(np.isfinite(omega)))or(not np.all(np.isfinite(density)))or np.any(np.diff(omega)<=0.0):
  raise ValueError('Expected finite spectral samples on an increasing frequency grid')
 scale=float(density.max())
 if not np.isfinite(linewidth)or linewidth<=0.0 or scale<=0.0:
  raise ValueError('Expected a positive linewidth and a nonzero spectral peak')
 midpoint=(omega[0]+omega[-1])/2
 half_width=(omega[-1]-omega[0])/2
 x=(omega-midpoint)/half_width
 scaled_density=density/scale
 def residual(parameters):
  shift,height,b0,b1,b2=parameters
  detuning=omega-midpoint-half_width*shift
  line=linewidth**2/(detuning**2+linewidth**2)
  background=b0+b1*x+b2*x**2
  return height*line+background-scaled_density
 initial=[float(x[np.argmax(density)]),1.0,0.0,0.0,0.0]
 lower=[-0.8,0.0,-np.inf,-np.inf,-np.inf]
 upper=[0.8,np.inf,np.inf,np.inf,np.inf]
 fit=least_squares(residual,initial,bounds=(lower,upper),xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=1000)
 if not fit.success or not np.all(np.isfinite(fit.x)):
  raise RuntimeError(f'Linear spectral fit failed: {fit.message}')
 shift,height,b0,b1,b2=fit.x
 gap=midpoint+half_width*shift
 weight=height*scale*np.pi*linewidth
 if gap<=0.0 or not np.isfinite(weight)or weight<=0.0:
  raise ValueError('The fitted peak must have a positive gap and weight')
 background=scale*(b0+b1*x+b2*x**2)
 capture=(np.arctan((omega[-1]-gap)/linewidth)-np.arctan((omega[0]-gap)/linewidth))/np.pi
 return{'gap':float(gap),'M':float(np.sqrt(weight)),'weight':float(weight),'integrated_weight_corrected':float(trapezoid(density-background,x=omega)/capture),'capture_fraction':float(capture),'linewidth':float(linewidth),'background_coefficients':(scale*np.array([b0,b1,b2])).tolist(),'relative_residual_rms':float(np.sqrt(np.mean(fit.fun**2))),'optimizer_success':bool(fit.success),'sample_count':len(omega)}
def pulse_protocol(calibration,measurement_cycles=384,ramp_cycles=8,steps_per_cycle=128,nominal_area=0.005):
 gap_sigma=calibration['gap_sigma']
 gap_epsilon=calibration['gap_epsilon']
 if not 0<gap_sigma<gap_epsilon:
  raise ValueError('Expected 0 < gap_sigma < gap_epsilon')
 period=2*np.pi/gap_epsilon
 ramp=ramp_cycles*period
 measure=measurement_cycles*period
 total=ramp+measure
 amplitude=nominal_area/total
 return{'omega1':gap_sigma,'omega2':gap_epsilon-gap_sigma,'omega_lock':gap_epsilon,'lambda1':amplitude,'lambda2':amplitude,'phi1':0.0,'phi2':0.0,'T_ramp':ramp,'T_measure':measure,'T_total':total,'n_steps':(measurement_cycles+ramp_cycles)*steps_per_cycle,'readout_gamma':0.0}
def _exponential_integral(frequency,time):
 if abs(frequency)<1e-14:
  return time.astype(complex)
 return np.expm1(1j*frequency*time)/(1j*frequency)
def _first_order_integral(time,gap,omega,phase,amplitude,ramp):
 result=np.zeros_like(time,dtype=complex)
 ramp_time=np.minimum(time,ramp)
 flat_time=np.maximum(time-ramp,0.0)
 for sign in(-1,1):
  frequency=gap+sign*omega
  factor=amplitude/2*np.exp(1j*sign*phase)
  if ramp>0:
   ramp_frequency=np.pi/ramp
   ramp_integral=_exponential_integral(frequency,ramp_time)/2-(_exponential_integral(frequency+ramp_frequency,ramp_time)+_exponential_integral(frequency-ramp_frequency,ramp_time))/4
   result+=factor*ramp_integral
  result+=factor*np.exp(1j*frequency*ramp)*_exponential_integral(frequency,flat_time)
 return result
def unit_kernel(calibration,protocol,internal_refinement=2):
 gap_sigma=calibration['gap_sigma']
 gap_epsilon=calibration['gap_epsilon']
 m_sigma=calibration['M_sigma']
 m_epsilon=calibration['M_epsilon']
 steps=protocol['n_steps']
 total=protocol['T_total']
 ramp=protocol['T_ramp']
 fine_steps=steps*internal_refinement
 time=np.linspace(0.0,total,fine_steps+1)
 dt=total/fine_steps
 envelope=np.ones_like(time)
 rising=time<ramp
 if ramp>0:
  envelope[rising]=np.sin(np.pi*time[rising]/(2*ramp))**2
 omega1,omega2=(protocol['omega1'],protocol['omega2'])
 phase1,phase2=(protocol['phi1'],protocol['phi2'])
 amplitude1,amplitude2=(protocol['lambda1'],protocol['lambda2'])
 field1=amplitude1*envelope*np.cos(omega1*time+phase1)
 field2=amplitude2*envelope*np.cos(omega2*time+phase2)
 first1=_first_order_integral(time,gap_sigma,omega1,phase1,amplitude1,ramp)
 first2=_first_order_integral(time,gap_sigma,omega2,phase2,amplitude2,ramp)
 derivative=-m_sigma**2*np.exp(1j*(gap_epsilon-gap_sigma)*time)
 derivative*=field1*first2+field2*first1
 amplitude_epsilon=cumulative_simpson(derivative.real,dx=dt,initial=0.0)+1j*cumulative_simpson(derivative.imag,dx=dt,initial=0.0)
 first_sample=round(steps*ramp/total)
 samples=slice(first_sample*internal_refinement,None,internal_refinement)
 acquired_time=time[samples]
 readout=2*m_epsilon*(amplitude_epsilon[samples]*np.exp(-1j*gap_epsilon*acquired_time)).real
 weights=np.exp(-protocol['readout_gamma']*(acquired_time-ramp))
 lock_phase=np.exp(-1j*protocol['omega_lock']*acquired_time)
 kernel=2*trapezoid(weights*readout*lock_phase,x=acquired_time)
 kernel/=trapezoid(weights,x=acquired_time)
 if not np.isfinite(kernel)or abs(kernel)<1e-16:
  raise ValueError('The unit response is too small for stable normalization')
 return complex(kernel)
def freeze_protocols(calibration,coefficient,preparation_amplitude_ratio=0.02,followup_amplitude_ratios=(0.02,0.01),followup_phase=0.37):
 gap_sigma=calibration['gap_sigma']
 gap_epsilon=calibration['gap_epsilon']
 m_sigma=calibration['M_sigma']
 predicted_link=m_sigma*float(np.real(coefficient))
 if not np.isfinite(predicted_link)or predicted_link==0.0:
  raise ValueError('Expected a finite nonzero pulse-based matrix element')
 prep_amplitude=preparation_amplitude_ratio*gap_sigma
 preparation={'omega':gap_sigma,'lambda':prep_amplitude,'drive_phase':0.0,'duration':np.pi/(prep_amplitude*m_sigma),'lambda_over_omega':preparation_amplitude_ratio}
 omega_drive=gap_epsilon-gap_sigma
 followups=[]
 for ratio in followup_amplitude_ratios:
  amplitude=ratio*omega_drive
  rabi_frequency=amplitude*abs(predicted_link)
  followups.append({'lambda_over_omega':ratio,'lambda':amplitude,'omega':omega_drive,'drive_phase':followup_phase,'duration':2*np.pi/rabi_frequency,'predicted_Rabi_frequency':rabi_frequency})
 return(predicted_link,preparation,followups)
@njit(inline='always',cache=False)
def _csr_row(row,data,columns,offsets,state):
 value=0j
 for entry in range(offsets[row],offsets[row+1]):
  value+=data[entry]*state[columns[entry]]
 return value
@njit(parallel=True,cache=False)
def _pulse_rhs(out,state,drive,data,columns,offsets,partner,magnetization):
 for row in prange(state.size):
  value=_csr_row(row,data,columns,offsets,state)
  other=partner[row]
  if other>=0:
   value+=drive*magnetization[row]*state[other]
  out[row]=-1j*value
@njit(parallel=True,cache=False)
def _form_stage(out,state,slopes,coefficients,step):
 for row in prange(state.size):
  correction=0j
  for stage in range(coefficients.size):
   weight=coefficients[stage]
   if weight!=0.0:
    correction+=weight*slopes[stage,row]
  out[row]=state[row]+step*correction
@njit(parallel=True,cache=False)
def _embedded_error(candidate,slopes,weights5,weights3):
 norm2=0.0
 error5_2=0.0
 error3_2=0.0
 for row in prange(candidate.size):
  error5=0j
  error3=0j
  for stage in range(weights5.size):
   slope=slopes[stage,row]
   error5+=weights5[stage]*slope
   error3+=weights3[stage]*slope
  sample=candidate[row]
  norm2+=sample.real**2+sample.imag**2
  error5_2+=error5.real**2+error5.imag**2
  error3_2+=error3.real**2+error3.imag**2
 return(norm2,error5_2,error3_2)
def _drive_value(time,frequencies,amplitudes,phases,ramp_time):
 drive=0.0
 for index in range(frequencies.size):
  drive+=amplitudes[index]*np.cos(frequencies[index]*time+phases[index])
 if time<ramp_time:
  drive*=np.sin(np.pi*time/(2*ramp_time))**2
 return drive
@njit(parallel=True,cache=False)
def _interpolate(out,state,candidate,slopes,delta_weight,slope_weights):
 for row in prange(state.size):
  correction=0j
  for stage in range(slope_weights.size):
   slope_weight=slope_weights[stage]
   if slope_weight!=0.0:
    correction+=slope_weight*slopes[stage,row]
  out[row]=state[row]+delta_weight*(candidate[row]-state[row])+correction
@njit(parallel=True,cache=False)
def _measure_state(state,data,columns,offsets,readout_diagonal,readout_h_factor):
 norm2=0.0
 readout=0.0
 for row in prange(state.size):
  value=state[row]
  weight=value.real**2+value.imag**2
  norm2+=weight
  if readout_diagonal.size:
   static=_csr_row(row,data,columns,offsets,state)
   readout+=readout_diagonal[row]*weight+readout_h_factor*(value.real*static.real+value.imag*static.imag)
 return(norm2,readout)
@njit(parallel=True,cache=False)
def _measure_amplitude(vector,state):
 real=0.0
 imag=0.0
 for row in prange(state.size):
  real+=vector[row]*state[row].real
  imag+=vector[row]*state[row].imag
 return complex(real,imag)
def stream_dop853(data,indices,indptr,spin_partner,spin_weight,initial_state,sample_times,tone_omega,tone_amplitude,tone_phase,ramp_time,target_vectors,readout_diagonal,readout_h_factor,rtol,atol,max_step,progress=None):
 if sample_times.ndim!=1 or not sample_times.size or(not np.all(np.isfinite(sample_times)))or(sample_times[0]<0)or np.any(np.diff(sample_times)<=0):
  raise ValueError('sample_times must be finite, nonnegative and increasing')
 if not np.all(np.isfinite([rtol,atol,max_step,ramp_time]))or rtol<=0 or atol<0 or(max_step<=0)or(ramp_time<0):
  raise ValueError('invalid integration scales')
 if initial_state.ndim!=1 or not initial_state.size:
  raise ValueError('initial_state must be a nonempty state vector')
 if tone_omega.ndim!=1 or tone_omega.shape!=tone_amplitude.shape or tone_omega.shape!=tone_phase.shape:
  raise ValueError('tone arrays must have matching one-dimensional shapes')
 state=np.array(initial_state,dtype=np.complex128,copy=True)
 dimension=state.size
 if indptr.shape!=(dimension+1,)or spin_partner.shape!=(dimension,)or spin_weight.shape!=(dimension,)or(target_vectors.ndim!=2)or(target_vectors.shape[1]!=dimension)or(readout_diagonal.shape not in((0,),(dimension,))):
  raise ValueError('operator, state or observable dimensions are inconsistent')
 slopes=np.empty((dop.N_STAGES_EXTENDED,dimension),dtype=np.complex128)
 work=np.empty_like(state)
 candidate=np.empty_like(state)
 target_samples=np.empty((sample_times.size,len(target_vectors)),dtype=np.complex128)
 readout_samples=np.empty(sample_times.size if readout_diagonal.size else 0)
 norms=np.empty(sample_times.size)
 basis_args=(data,indices,indptr,spin_partner,spin_weight)
 tone_args=(tone_omega,tone_amplitude,tone_phase,ramp_time)
 standard_stages=dop.N_STAGES
 endpoint_time=float(sample_times[-1])
 def evaluate(out,wavefunction,time):
  _pulse_rhs(out,wavefunction,_drive_value(time,*tone_args),*basis_args)
 def store(index,wavefunction):
  norm2,readout=_measure_state(wavefunction,data,indices,indptr,readout_diagonal,readout_h_factor)
  if not np.isfinite(norm2)or not np.isfinite(readout):
   raise FloatingPointError('nonfinite observation during full-state propagation')
  norms[index]=norm2
  if readout_diagonal.size:
   readout_samples[index]=readout
  for target_index,vector in enumerate(target_vectors):
   target_samples[index,target_index]=_measure_amplitude(vector,wavefunction)
 norm=float(np.linalg.norm(state))
 if not np.isfinite(norm)or norm<=0:
  raise ValueError('initial state has an invalid norm')
 sample=0
 if sample_times[0]==0:
  store(0,state)
  sample=1
 if endpoint_time==0:
  return(state,target_samples,readout_samples,norms,0,0)
 time=0.0
 evaluate(slopes[0],state,time)
 nfev=1
 accepted=0
 rejected=0
 last_progress=perf_counter()
 step_size=min(0.1*max_step,endpoint_time)
 after_rejection=False
 while time<endpoint_time:
  next_time=min(time+min(step_size,max_step),endpoint_time)
  step=next_time-time
  if step<=0.0:
   raise FloatingPointError(f'DOP853 time step underflow at t={time:g}')
  for stage in range(1,standard_stages):
   _form_stage(work,state,slopes,dop.A[stage,:stage],step)
   evaluate(slopes[stage],work,time+step*dop.C[stage])
  _form_stage(candidate,state,slopes,dop.B,step)
  evaluate(slopes[standard_stages],candidate,next_time)
  nfev+=standard_stages
  norm2,err5_2,err3_2=_embedded_error(candidate,slopes,dop.E5,dop.E3)
  candidate_norm=np.sqrt(norm2)
  scale=atol+rtol*max(norm,candidate_norm)
  denominator=err5_2+0.01*err3_2
  if not np.isfinite(candidate_norm+denominator):
   error=np.inf
  elif denominator==0:
   error=0.0
  else:
   error=step*err5_2/(scale*np.sqrt(denominator))
  if error>=1.0:
   step_size=step*max(0.2,0.9*error**(-1/8))
   rejected+=1
   after_rejection=True
   continue
  factor=10.0 if error==0.0 else min(10.0,0.9*error**(-1/8))
  if after_rejection:
   factor=min(1.0,factor)
  step_size=step*factor
  after_rejection=False
  accepted+=1
  sample_stop=int(np.searchsorted(sample_times,next_time,side='right'))
  if sample_stop>sample and sample_times[sample]<next_time:
   for stage in range(standard_stages+1,dop.N_STAGES_EXTENDED):
    _form_stage(work,state,slopes,dop.A[stage,:stage],step)
    evaluate(slopes[stage],work,time+step*dop.C[stage])
    nfev+=1
  while sample<sample_stop:
   if sample_times[sample]==next_time:
    store(sample,candidate)
   else:
    x=(sample_times[sample]-time)/step
    y=1-x
    weight0,weight1,weight2=(x,x*y,x*x*y)
    tail=np.array([x*x*y*y,x**3*y*y,x**3*y**3,x**4*y**3])
    slope_weights=step*(tail@dop.D)
    slope_weights[0]+=step*(weight1-weight2)
    slope_weights[standard_stages]-=step*weight2
    _interpolate(work,state,candidate,slopes,weight0-weight1+2*weight2,slope_weights)
    store(sample,work)
   sample+=1
  state,candidate=(candidate,state)
  slopes[0]=slopes[standard_stages]
  norm=candidate_norm
  time=next_time
  if progress is not None and perf_counter()-last_progress>=60:
   progress(time,endpoint_time,accepted,rejected,nfev)
   last_progress=perf_counter()
 return(state,target_samples,readout_samples,norms,nfev,rejected)
def numa_layout():
 allowed=os.sched_getaffinity(0)
 output=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET,NODE'],text=True)
 groups={}
 seen=set()
 for line in output.splitlines():
  if line.startswith('#'):
   continue
  cpu,core,socket_id,node=map(int,line.split(','))
  key=(socket_id,core)
  if cpu in allowed and key not in seen:
   groups.setdefault(node,[]).append(cpu)
   seen.add(key)
 if len(groups)!=8 or any((len(cpus)!=6 for cpus in groups.values())):
  raise RuntimeError('This run is configured for a whole 48-core, eight-NUMA EPYC node')
 return groups
def bind_worker(cpus,nodes):
 os.sched_setaffinity(0,set(cpus))
 library=ctypes.CDLL('libnuma.so.1')
 library.numa_allocate_nodemask.restype=ctypes.c_void_p
 library.numa_bitmask_setbit.argtypes=[ctypes.c_void_p,ctypes.c_uint]
 library.numa_set_interleave_mask.argtypes=[ctypes.c_void_p]
 library.numa_bitmask_free.argtypes=[ctypes.c_void_p]
 mask=library.numa_allocate_nodemask()
 for node in nodes:
  library.numa_bitmask_setbit(mask,node)
 library.numa_set_interleave_mask(mask)
 library.numa_bitmask_free(mask)
 set_num_threads(len(cpus))
 threadpool_limits(limits=1,user_api='blas')
def propagate_task(model,initial,protocol,kind,cpus,nodes):
 bind_worker(cpus,nodes)
 local={key:value.copy()if isinstance(value,np.ndarray)else value for key,value in model.items()}
 initial=initial.copy()
 started=perf_counter()
 if kind.startswith('acquisition'):
  times=np.linspace(0,protocol['T_total'],protocol['n_steps']+1)
  frequency=np.array([protocol['omega1'],protocol['omega2']])
  amplitude=np.array([protocol['lambda1'],protocol['lambda2']])
  phase=np.array([protocol['phi1'],protocol['phi2']])
  ramp=protocol['T_ramp']
  targets=np.empty((0,initial.size))
  readout=local['readout']
 else:
  times=np.linspace(0,protocol['duration'],TRAJECTORY_POINTS)
  frequency=np.array([protocol['omega']])
  amplitude=np.array([protocol['lambda']])
  phase=np.array([protocol['drive_phase']])
  ramp=0.0
  targets=local['targets']
  readout=np.empty(0)
 max_step=2*np.pi/(frequency.max()*CARRIER_STEPS)
 def report(time,end,accepted,rejected,nfev):
  seconds=perf_counter()-started
  print(f'{kind}: {100 * time / end:.2f}% physical time, {seconds:.0f} s elapsed, accepted={accepted}, rejected={rejected}, RHS={nfev}',flush=True)
 print(f'{kind}: {len(cpus)} physical cores, NUMA {nodes}, dimension={initial.size}, T={times[-1]:.9g}',flush=True)
 final,projections,observations,norms,nfev,rejected=stream_dop853(local['data'],local['indices'],local['indptr'],local['partner'],local['bridge'],initial,times,frequency,amplitude,phase,ramp,targets,readout,local['readout_h_factor'],RTOL,ATOL,max_step,report)
 norm_error=float(np.max(np.abs(norms-np.vdot(initial,initial).real)))
 if not np.all(np.isfinite(projections))or norm_error>MAX_NORM_DRIFT:
  raise ValueError(f'Invalid {kind}: maximum norm drift={norm_error}')
 result=dict(times=times,projections=projections,norms=norms,norm_error=norm_error,nfev=nfev,rejected_steps=rejected,elapsed_seconds=perf_counter()-started)
 if kind.startswith('acquisition'):
  first=RAMP_CYCLES*ACQUISITION_STEPS
  acquired=times[first:]
  result['lock']=2*trapezoid(observations[first:]*np.exp(-1j*protocol['omega_lock']*acquired),x=acquired)/protocol['T_measure']
 elif kind=='preparation':
  result['final_state']=final
 print(f"{kind}: finished in {result['elapsed_seconds']:.1f} s, norm drift={norm_error:.3e}",flush=True)
 return result
def result_rows(calibration,metadata,coefficient,signal,kernel,preparation,prep_result,followups,validation_results,acquisition_results,runtime):
 predicted_link=calibration['M_sigma']*coefficient.real
 common=dict(L=LENGTH,J1=J1,J2=J2,h=FIELD,trajectory_source='fresh_full_q0_sparse_evolution',**calibration,**metadata,C_time_real=coefficient.real,C_time_imag=coefficient.imag,S_real=signal.real,S_imag=signal.imag,K_real=kernel.real,K_imag=kernel.imag,B_predicted_real=predicted_link,preparation_omega=preparation['omega'],preparation_lambda=preparation['lambda'],preparation_drive_phase=preparation['drive_phase'],preparation_duration=preparation['duration'],preparation_target_fidelity=abs(prep_result['projections'][-1,1])**2,preparation_max_norm_error=prep_result['norm_error'],preparation_nfev=prep_result['nfev'],target_projection_or_state_replacement=False,samples=TRAJECTORY_POINTS,prediction_initial_state='ideal sigma target; preparation errors retained in validation',validation_initial_state='entire unprojected state from the calibrated preparation pulse',evolution_method='streamed_DOP853_full_q0_Schrodinger_picture',rtol=RTOL,atol=ATOL,error_norm='Hilbert_norm_without_dimension_divisor',measurement_cycles=MEASURE_CYCLES,ramp_cycles=RAMP_CYCLES,acquisition_steps_per_cycle=ACQUISITION_STEPS,nominal_area=NOMINAL_AREA,kernel_refinement=KERNEL_REFINEMENT,acquisition_max_norm_error=max((item['norm_error']for item in acquisition_results)),server_node=socket.gethostname(),physical_cores=48,elapsed_seconds=runtime)
 for index,item in enumerate(acquisition_results):
  common[f'phase{index}_lock_real']=item['lock'].real
  common[f'phase{index}_lock_imag']=item['lock'].imag
  common[f'phase{index}_nfev']=item['nfev']
 rows=[]
 for protocol,result in zip(followups,validation_results):
  times=result['times']
  ground,sigma,epsilon=result['projections'].T
  population=np.abs(epsilon)**2
  predicted=np.sin(protocol['lambda']*predicted_link*times/2)**2
  exact_transition=metadata['exact_gap_epsilon']-metadata['exact_gap_sigma']
  coherence=2*epsilon*sigma.conj()*np.exp(1j*exact_transition*times)
  predicted_coherence=-1j*np.exp(-1j*protocol['drive_phase'])*np.sin(protocol['lambda']*predicted_link*times)
  outside=result['norms']-population-np.abs(sigma)**2
  error=population-predicted
  summary=dict(population_max_abs_error=float(np.max(np.abs(error))),population_max_abs_error_pp=float(100*np.max(np.abs(error))),complex_coherence_max_abs_error=float(np.max(np.abs(coherence-predicted_coherence))),max_population_outside_target_pair=float(outside.max()),validation_max_norm_error=result['norm_error'],validation_nfev=result['nfev'],validation_rejected_steps=result['rejected_steps'],validation_elapsed_seconds=result['elapsed_seconds'])
  for k,time in enumerate(times):
   rows.append(dict(**common,**protocol,**summary,time=time,P_epsilon_full=population[k],P_epsilon_predicted=predicted[k],coherence_full_real=coherence[k].real,coherence_full_imag=coherence[k].imag,coherence_predicted_real=predicted_coherence[k].real,coherence_predicted_imag=predicted_coherence[k].imag,P_outside_target_pair=outside[k],P_ground=abs(ground[k])**2,norm=result['norms'][k],predicted_Rabi_cycles=protocol['predicted_Rabi_frequency']*time/(2*np.pi),population_error_pp=100*error[k]))
 return rows
def launch_trajectory(stack,model,initial,protocol,kind,cpus,nodes):
 previous=os.environ.get('NUMBA_NUM_THREADS')
 os.environ['NUMBA_NUM_THREADS']=str(len(cpus))
 try:
  executor=stack.enter_context(ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')))
  return executor.submit(propagate_task,model,initial,protocol,kind,cpus,nodes)
 finally:
  if previous is None:
   os.environ.pop('NUMBA_NUM_THREADS')
  else:
   os.environ['NUMBA_NUM_THREADS']=previous
def main():
 started=perf_counter()
 layout=numa_layout()
 domains=sorted(layout)
 all_cpus=[cpu for node in domains for cpu in layout[node]]
 bind_worker(all_cpus,domains)
 with threadpool_limits(limits=12,user_api='blas'):
  model,calibration,metadata=build_model(LENGTH,J1,J2,FIELD)
 print(f'Independent linear calibration: {calibration}',flush=True)
 pulse=pulse_protocol(calibration,MEASURE_CYCLES,RAMP_CYCLES,ACQUISITION_STEPS,NOMINAL_AREA)
 kernel=unit_kernel(calibration,pulse,KERNEL_REFINEMENT)
 prep_amplitude=PREPARATION_RATIO*calibration['gap_sigma']
 preparation=dict(omega=calibration['gap_sigma'],**{'lambda':prep_amplitude},drive_phase=0.0,duration=np.pi/(prep_amplitude*calibration['M_sigma']),lambda_over_omega=PREPARATION_RATIO)
 ground=model['targets'][0].astype(np.complex128)
 prep_result=propagate_task(model,ground,preparation,'preparation',all_cpus,domains)
 prepared=prep_result.pop('final_state')
 with ExitStack()as stack:
  futures=[]
  for index,(phase1,phase2)in enumerate(PHASE_SHIFTS):
   shot=dict(pulse,phi1=phase1,phi2=phase2)
   nodes=domains[2*index:2*(index+1)]
   cpus=[cpu for node in nodes for cpu in layout[node]]
   futures.append(launch_trajectory(stack,model,ground,shot,f'acquisition-{index}',cpus,nodes))
  acquisition=[future.result()for future in futures]
 locks=np.array([item['lock']for item in acquisition])
 signal=np.array([1.0,-1.0,-1.0,1.0])@locks/4
 coefficient=signal/kernel
 if not np.isfinite(coefficient):
  raise ValueError('The finite-pulse coefficient is nonfinite')
 predicted_link,frozen_prep,followups=freeze_protocols(calibration,coefficient,PREPARATION_RATIO,DRIVE_RATIOS,FOLLOWUP_PHASE)
 if frozen_prep!=preparation:
  raise ValueError('Preparation was changed after acquisition')
 print(f'Frozen prediction: C={coefficient}, B={predicted_link}; starting independent complete-state validation',flush=True)
 with ExitStack()as stack:
  validation_futures=[]
  for index,protocol in enumerate(followups):
   nodes=domains[:2]if index==0 else domains[2:]
   cpus=[cpu for node in nodes for cpu in layout[node]]
   validation_futures.append(launch_trajectory(stack,model,prepared,protocol,f"validation-{protocol['lambda_over_omega']:.3f}",cpus,nodes))
  validation=[future.result()for future in validation_futures]
 rows=result_rows(calibration,metadata,coefficient,signal,kernel,preparation,prep_result,followups,validation,acquisition,perf_counter()-started)
 with OUTPUT.open('w',newline='',encoding='utf-8')as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
  writer.writeheader()
  writer.writerows(rows)
 print(f'Completed L={LENGTH}: {len(rows)} rows -> {OUTPUT}; total {perf_counter() - started:.1f} s',flush=True)
if __name__=='__main__':
 main()
