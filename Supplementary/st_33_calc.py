from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_simpson
from scipy.optimize import least_squares
from st_1_calc import make_orbits,projected_model
OUTPUT=Path(__file__).with_name('st_33.csv')
LENGTHS=range(6,27,2)
MODELS=('Ising','NI')
STATES_PER_PARITY=64
RAMP_CYCLES=8
MEASURE_CYCLES=96
NOMINAL_AREA=0.02
STEPS_PER_CYCLE=128
PHASE_SHIFTS=((0.0,0.0),(np.pi,0.0),(0.0,np.pi),(np.pi,np.pi))
PHASE_SIGNS=np.array([1.0,-1.0,-1.0,1.0])/4
CAL_KEYS={'gap_sigma','gap_epsilon','M_sigma','M_epsilon'}
PROTOCOL_KEYS={'omega1','omega2','omega_lock','lambda1','lambda2','phi1','phi2','T_ramp','T_measure','T_total','n_steps','readout_gamma'}
def as_complex(obj):
 return complex(float(obj['real']),float(obj['imag']))
def complex_json(z):
 return{'real':float(np.real(z)),'imag':float(np.imag(z))}
def validate_input(record):
 if set(record)!={'schema_version','case_id','calibration','protocol','measured_locks'}:
  raise ValueError('Unexpected analysis-input fields')
 cal=record['calibration']
 p=record['protocol']
 if set(cal)!=CAL_KEYS or set(p)!=PROTOCOL_KEYS:
  raise ValueError('Only declared linear calibration and waveform fields are allowed')
 if any((not np.isfinite(float(cal[k]))or float(cal[k])<=0 for k in CAL_KEYS)):
  raise ValueError('Calibrated gaps and phase-fixed linear amplitudes must be positive')
 if int(p['n_steps'])<4 or p['T_total']<=p['T_ramp']:
  raise ValueError('Invalid acquisition window')
 if not np.isclose(p['T_total']-p['T_ramp'],p['T_measure']):
  raise ValueError('Inconsistent acquisition durations')
 if len(record['measured_locks'])!=4:
  raise ValueError('Four ordered phase locks required')
 for v in record['measured_locks']:
  if set(v)!={'real','imag'}or not np.isfinite(as_complex(v)):
   raise ValueError('Measured lock must contain finite real and imag fields')
def envelope(t,Tr):
 t=np.asarray(t,dtype=float)
 if Tr<=0:
  return np.ones_like(t)
 return np.where(t<Tr,np.sin(np.pi*t/(2*Tr))**2,1.0)
def exp_integral(q,t):
 t=np.asarray(t,dtype=float)
 if abs(q)<1e-14:
  return t.astype(complex)
 return np.expm1(1j*q*t)/(1j*q)
def first_integral(t,energy,omega,phase,amplitude,Tr):
 t=np.asarray(t,dtype=float)
 if Tr<=0:
  return amplitude*0.5*(np.exp(1j*phase)*exp_integral(energy+omega,t)+np.exp(-1j*phase)*exp_integral(energy-omega,t))
 u=np.minimum(t,Tr)
 inside=np.zeros_like(t,dtype=complex)
 at_ramp=0j
 for sign in[-1.0,1.0]:
  coefficient=0.5*amplitude*np.exp(1j*sign*phase)
  for shift,weight in[(0.0,0.5),(np.pi/Tr,-0.25),(-np.pi/Tr,-0.25)]:
   q=energy+sign*omega+shift
   inside+=coefficient*weight*exp_integral(q,u)
   at_ramp+=coefficient*weight*exp_integral(q,np.array(Tr))
 after=np.full_like(t,at_ramp,dtype=complex)
 v=np.maximum(t-Tr,0.0)
 for sign in[-1.0,1.0]:
  q=energy+sign*omega
  after+=0.5*amplitude*np.exp(1j*sign*phase)*np.exp(1j*q*Tr)*exp_integral(q,v)
 return np.where(t<=Tr,inside,after)
def complex_cumulative_simpson(y,dx):
 return cumulative_simpson(y.real,dx=dx,initial=0.0)+1j*cumulative_simpson(y.imag,dx=dx,initial=0.0)
def unit_kernel(calibration,protocol,internal_refinement=2):
 p=protocol
 c=calibration
 n=int(p['n_steps'])
 r=int(internal_refinement)
 if r<1:
  raise ValueError('Positive internal refinement required')
 t=np.linspace(0.0,p['T_total'],n*r+1)
 dt=p['T_total']/(n*r)
 Es=c['gap_sigma']
 Ee=c['gap_epsilon']
 Ms=c['M_sigma']
 Me=c['M_epsilon']
 g=envelope(t,p['T_ramp'])
 f1=g*p['lambda1']*np.cos(p['omega1']*t+p['phi1'])
 f2=g*p['lambda2']*np.cos(p['omega2']*t+p['phi2'])
 I1=first_integral(t,Es,p['omega1'],p['phi1'],p['lambda1'],p['T_ramp'])
 I2=first_integral(t,Es,p['omega2'],p['phi2'],p['lambda2'],p['T_ramp'])
 rhs=-Ms**2*np.exp(1j*(Ee-Es)*t)*(f1*I2+f2*I1)
 ae=complex_cumulative_simpson(rhs,dt)
 t=t[::r]
 ce=ae[::r]*np.exp(-1j*Ee*t)
 signal=2*Me*np.real(ce)
 keep=t>=p['T_ramp']-1e-12
 wt=np.exp(-p['readout_gamma']*(t[keep]-p['T_ramp']))
 normalization=np.trapezoid(wt,x=t[keep])
 return complex(2*np.trapezoid(wt*signal[keep]*np.exp(-1j*p['omega_lock']*t[keep]),x=t[keep])/normalization)
def extract(record,internal_refinement=2):
 validate_input(record)
 locks=np.array([as_complex(z)for z in record['measured_locks']])
 signal=complex(PHASE_SIGNS@locks)
 kernel=unit_kernel(record['calibration'],record['protocol'],internal_refinement)
 if abs(kernel)<1e-16:
  raise ValueError('Unit kernel too small for a stable ratio')
 return{'schema_version':'1.0','case_id':record['case_id'],'phase_cycled_signal':complex_json(signal),'unit_kernel':complex_json(kernel),'estimate':complex_json(signal/kernel),'internal_refinement':int(internal_refinement)}
def fit_linear_peak(omega,density,linewidth):
 w=np.asarray(omega,dtype=float)
 y=np.asarray(density,dtype=float)
 if w.ndim!=1 or y.shape!=w.shape or len(w)<20:
  raise ValueError('Invalid spectral band')
 center=0.5*(w[0]+w[-1])
 W=0.5*(w[-1]-w[0])
 x=(w-center)/W
 scale=float(max(y))
 gamma=float(linewidth)
 if gamma<=0 or scale<=0:
  raise ValueError('Positive linewidth and spectrum required')
 def profile(v):
  shift,A,b0,b1,b2=v
  line=gamma**2/((w-center-W*shift)**2+gamma**2)
  return A*line+b0+b1*x+b2*x*x
 v0=[float(x[np.argmax(y)]),1.0,0.0,0.0,0.0]
 solve=least_squares(lambda v:profile(v)-y/scale,v0,bounds=([-0.8,0.0,-np.inf,-np.inf,-np.inf],[0.8,np.inf,np.inf,np.inf,np.inf]),xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=1000)
 shift,A,b0,b1,b2=solve.x
 gap=center+W*shift
 residue=A*scale*np.pi*gamma
 background=scale*(b0+b1*x+b2*x*x)
 capture=(np.arctan((w[-1]-gap)/gamma)-np.arctan((w[0]-gap)/gamma))/np.pi
 integrated=np.trapezoid(y-background,x=w)/capture
 return{'gap':float(gap),'M':float(np.sqrt(residue)),'weight':float(residue),'integrated_weight_corrected':float(integrated),'capture_fraction':float(capture),'linewidth':gamma,'background_coefficients':(scale*np.array([b0,b1,b2])).tolist(),'relative_residual_rms':float(np.sqrt(np.mean((profile(solve.x)-y/scale)**2))),'optimizer_success':bool(solve.success),'sample_count':len(w)}
def pulse_derivative(state,reference,phase,field,first_phase,second_phase,spin,m_sigma):
 derivative=-1j*field*phase.conj()[:,None]*(spin@(phase[:,None]*state))
 reference_derivative=np.empty_like(reference)
 reference_derivative[0]=-1j*m_sigma*field*first_phase
 reference_derivative[1]=-1j*m_sigma*field*second_phase*reference[0]
 return(derivative,reference_derivative)
def acquired_locks(energy,spin,readout,gap_sigma,gap_epsilon,m_sigma,m_epsilon,*,steps_per_cycle=STEPS_PER_CYCLE,measure_cycles=MEASURE_CYCLES,nominal_area=NOMINAL_AREA):
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
 return dict(measured_locks=[complex_json(z)for z in(2*dt/measure_time*signal_sum)[[0,2,1,3]]],n_steps=nsteps,T_total=total_time,C_real=coefficient.real,C_imag=coefficient.imag,S_real=signal.real,S_imag=signal.imag,K_real=kernel.real,K_imag=kernel.imag,max_norm_error=max_norm_error,lambda_value=drive_amplitude,dt=dt,T_ramp=ramp_time,T_measure=measure_time,resolution=2*np.pi/measure_time,actual_envelope_area=drive_amplitude*(measure_time+ramp_time/2))
def run_s33(label,length,energy,spin,readout,static):
 key=f'{label}_L{length}'
 spectra,sources,calibration,fits=({},{},{},{})
 for channel,operator in(('sigma',spin),('epsilon',readout)):
  gap=static['gap_'+channel]
  gamma=0.002*gap
  omega=np.linspace(0.9*gap,1.1*gap,4001)
  keep=energy>1e-12
  weights=abs(operator[keep,0])**2
  density=np.sum(weights[None,:]*gamma/(np.pi*((omega[:,None]-energy[keep][None,:])**2+gamma**2)),axis=1)
  fit=fit_linear_peak(omega,density,gamma)
  if not fit['optimizer_success']:
   raise RuntimeError('Linear fit failed')
  calibration['gap_'+channel],calibration['M_'+channel]=(fit['gap'],fit['M'])
  fits[channel]=fit
  spectra[channel]=dict(angular_frequency=omega.tolist(),S_over_2pi=density.tolist())
  sources[channel]=dict(known_lorentz_width=gamma,window_halfwidth=0.1*gap,band_selection='isolated positive-frequency target band',sample_count=4001,normalization='sum |<n|O|0>|^2 eta/[pi*((omega-E_n)^2+eta^2)]',data_type='noiseless broadened Lehmann spectrum in stated retained q=0 space')
 records={}
 values=[]
 diagnostics={}
 for cycles,area in((96,0.02),(384,0.005)):
  p=acquired_locks(energy,spin,readout,static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'],measure_cycles=cycles,nominal_area=area)
  case=f'{key}_N{cycles}_A{area:.3f}_s128_nominal'
  protocol=dict(omega1=static['gap_sigma'],omega2=static['gap_epsilon']-static['gap_sigma'],omega_lock=static['gap_epsilon'],lambda1=p['lambda_value'],lambda2=p['lambda_value'],phi1=0.0,phi2=0.0,T_ramp=p['T_ramp'],T_measure=p['T_measure'],T_total=p['T_total'],n_steps=p['n_steps'],readout_gamma=0.0)
  record=dict(schema_version='1.0',case_id=case,calibration=calibration,protocol=protocol,measured_locks=p['measured_locks'])
  records[case]=record
  values.append(as_complex(extract(record)['estimate']))
  diagnostics[case]={k:v for k,v in p.items()if k!='measured_locks'}
 short,long=values
 direct=static['C_direct']
 row=dict(model=label,L=length,C_direct_real=direct,C_real_96=short.real,C_imag_96=short.imag,C_real_384=long.real,C_imag_384=long.imag,relative_complex_error_percent=100*abs(long-direct)/abs(direct))
 return dict(key=key,row=row,spectra=spectra,specification=dict(calibration=calibration,fits=fits,sources=sources),pulse_records=records,direct_audit=[static['B_epsilon_sigma'],static['M_sigma']],metadata=static,diagnostics=diagnostics)
def calculate_fresh(lengths=range(6,27,2),labels=('Ising','NI'),states_per_parity=64):
 rows=[]
 for label in labels:
  j2,field=(0.0,1.0)if label=='Ising'else(0.05,1.08385)
  for length in lengths:
   orbits=make_orbits(int(length))
   energy,spin,readout,static=projected_model(int(length),j2,field,orbits,np.random.default_rng(2718),states_per_parity=states_per_parity)
   del orbits
   rows.append(run_s33(label,int(length),energy,spin,readout,static)['row'])
 return pd.DataFrame(rows).sort_values(['model','L'])
def fresh_calibration(energy,spin,readout,static):
 calibration={}
 for channel,operator in(('sigma',spin),('epsilon',readout)):
  gap=static['gap_'+channel]
  width=0.002*gap
  omega=np.linspace(0.9*gap,1.1*gap,4001)
  keep=energy>1e-12
  weight=abs(operator[keep,0])**2
  density=np.sum(weight[:,None]*width/np.pi/((omega[None,:]-energy[keep,None])**2+width**2),axis=0)
  fit=fit_linear_peak(omega,density,width)
  if not fit['optimizer_success']:
   raise RuntimeError('Independent linear spectral fit failed')
  calibration['gap_'+channel]=fit['gap']
  calibration['M_'+channel]=fit['M']
 return calibration
def calculate(lengths=LENGTHS,labels=MODELS,states_per_parity=STATES_PER_PARITY):
 return calculate_fresh(lengths,labels,states_per_parity)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
