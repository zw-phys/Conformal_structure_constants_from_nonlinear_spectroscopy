from pathlib import Path
import numpy as np
import pandas as pd
import math
from scipy.integrate import solve_ivp
from pulse_calc import build_model,forward_pulse
from st_33_calc import unit_kernel
LENGTH=8
COUPLING=0.05
FIELD=1.08387382
STATES_PER_PARITY=64
CYCLES=(96,192,384,768)
AREA=0.02
STEPS=128
RAMP=8
RTOL=2e-11
ATOL=2e-14
OUTPUT=Path(__file__).with_name('st_37.csv')
def make_time_grid(omega_lock:float,measurement_cycles:float,ramp_cycles:float,steps_per_cycle:int):
 if omega_lock<=0.0:
  raise ValueError('omega_lock must be positive')
 if measurement_cycles<=0.0 or ramp_cycles<0.0:
  raise ValueError('invalid pulse-cycle counts')
 if steps_per_cycle<16:
  raise ValueError('steps_per_cycle must be at least 16')
 period=2.0*math.pi/omega_lock
 T_ramp=ramp_cycles*period
 T_measure=measurement_cycles*period
 T_total=T_ramp+T_measure
 dt_guess=period/float(steps_per_cycle)
 n_steps=max(4,int(round(T_total/dt_guess)))
 dt=T_total/float(n_steps)
 return dict(omega_lock=float(omega_lock),period_lock=float(period),ramp_cycles=float(ramp_cycles),measurement_cycles=float(measurement_cycles),steps_per_cycle=int(steps_per_cycle),T_ramp=float(T_ramp),T_measure=float(T_measure),T_total=float(T_total),dt=float(dt),n_steps=int(n_steps))
def ramp_envelope(t:float,T_ramp:float)->float:
 if T_ramp<=0.0 or t>=T_ramp:
  return 1.0
 return math.sin(0.5*math.pi*t/T_ramp)**2
def quadratic(m,cycles=96,area=0.02,steps=128):
 g=make_time_grid(m['gap_epsilon'],cycles,RAMP,steps)
 lam=area/g['T_total']
 E=m['energies']
 a1=np.zeros((len(E),4),complex)
 a2=np.zeros_like(a1)
 p1=np.array([0,np.pi,0,np.pi])
 p2=np.array([0,0,np.pi,np.pi])
 sgn=np.array([1,-1,-1,1])/4
 tlist=[]
 signals=[]
 def rhs(t,x1,x2):
  phase=np.exp(-1j*E*t)
  field=lam*ramp_envelope(t,g['T_ramp'])*(np.cos(m['gap_sigma']*t+p1)+np.cos((m['gap_epsilon']-m['gap_sigma'])*t+p2))
  q1=-1j*np.conj(phase)[:,None]*m['O_sigma'][:,0,None]*field[None,:]
  q2=-1j*np.conj(phase)[:,None]*(m['O_sigma']@(phase[:,None]*x1))*field[None,:]
  return(q1,q2)
 for step in range(g['n_steps']+1):
  t=step*g['dt']
  if step>=RAMP*steps:
   phase=np.exp(-1j*E*t)
   s1=phase[:,None]*a1
   s2=phase[:,None]*a2
   sig=2*(m['O_epsilon_connected']@s2)[0,:].real+np.sum(np.conj(s1)*(m['O_epsilon_connected']@s1),axis=0).real
   tlist.append(t)
   signals.append(float(sig@sgn))
  if step==g['n_steps']:
   break
  dt=g['dt']
  q1=rhs(t,a1,a2)
  q2=rhs(t+dt/2,a1+dt*q1[0]/2,a2+dt*q1[1]/2)
  q3=rhs(t+dt/2,a1+dt*q2[0]/2,a2+dt*q2[1]/2)
  q4=rhs(t+dt,a1+dt*q3[0],a2+dt*q3[1])
  a1+=dt/6*(q1[0]+2*q2[0]+2*q3[0]+q4[0])
  a2+=dt/6*(q1[1]+2*q2[1]+2*q3[1]+q4[1])
 tt=np.array(tlist)
 ss=np.array(signals)
 num=2*np.trapezoid(ss*np.exp(-1j*g['omega_lock']*tt),tt)/(tt[-1]-tt[0])
 return complex(num)
def full_signal(model,protocol):
 size=len(model.energies)
 times=np.linspace(0,protocol['T_total'],protocol['n_steps']+1)
 initial=np.zeros((size,4),complex)
 initial[0]=1
 signs1=np.array([1.0,-1.0,1.0,-1.0])
 signs2=np.array([1.0,1.0,-1.0,-1.0])
 def rhs(time,flat):
  state=flat.reshape(size,4)
  field=ramp_envelope(time,protocol['T_ramp'])*(protocol['lambda1']*np.cos(protocol['omega1']*time)*signs1+protocol['lambda2']*np.cos(protocol['omega2']*time)*signs2)
  phase=np.exp(-1j*model.energies*time)
  return(-1j*field*phase.conj()[:,None]*(model.O_sigma@(phase[:,None]*state))).ravel()
 solution=solve_ivp(rhs,(0,protocol['T_total']),initial.ravel(),method='DOP853',t_eval=times,rtol=RTOL,atol=ATOL,max_step=2*np.pi/max(protocol['omega1'],protocol['omega2'])/32)
 if not solution.success:
  raise RuntimeError(solution.message)
 states=solution.y.reshape(size,4,len(times))
 norm_error=np.max(abs(np.sum(abs(states)**2,axis=0)-1))
 physical=states*np.exp(-1j*model.energies[:,None,None]*times)
 measured=np.sum(physical.conj()*(model.O_epsilon_connected@physical.reshape(size,-1)).reshape(physical.shape),axis=0).real
 start=round(protocol['n_steps']*protocol['T_ramp']/protocol['T_total'])
 locks=2*np.trapezoid(measured[:,start:]*np.exp(-1j*protocol['omega_lock']*times[start:]),x=times[start:],axis=1)/protocol['T_measure']
 if not np.all(np.isfinite(locks))or norm_error>1e-08:
  raise ValueError('Invalid propagated state or lock-in signal')
 return(locks@np.array([1.0,-1.0,-1.0,1.0])/4,float(norm_error))
def pulse_components(model,cycles,area,steps):
 grid=make_time_grid(model.gap_epsilon,cycles,RAMP,steps)
 amplitude=area/grid['T_total']
 calibration={name:getattr(model,name)for name in('gap_sigma','gap_epsilon','M_sigma','M_epsilon')}
 protocol=dict(omega1=model.gap_sigma,omega2=model.gap_epsilon-model.gap_sigma,omega_lock=model.gap_epsilon,lambda1=amplitude,lambda2=amplitude,phi1=0.0,phi2=0.0,T_ramp=grid['T_ramp'],T_measure=grid['T_measure'],T_total=grid['T_total'],n_steps=grid['n_steps'],readout_gamma=0.0)
 signal,norm_error=full_signal(model,protocol)
 kernel=unit_kernel(calibration,protocol)
 quadratic_model=dict(energies=model.energies,O_sigma=model.O_sigma,O_epsilon_connected=model.O_epsilon_connected,**calibration)
 second=quadratic(quadratic_model,cycles,area,steps)
 if abs(kernel)==0:
  raise ValueError('Vanishing target reference kernel')
 return(signal/kernel,second/kernel-model.C_direct,(signal-second)/kernel,norm_error)
def calculate(length=LENGTH,cycles=CYCLES,steps=STEPS):
 model=build_model(length,COUPLING,FIELD,STATES_PER_PARITY)
 settings=[(n,AREA)for n in cycles]+[(max(cycles),AREA/2)]
 rows=[]
 for duration,area in settings:
  full,background,higher,norm_error=pulse_components(model,duration,area,steps)
  np.testing.assert_allclose(full-model.C_direct,background+higher,rtol=0,atol=5e-15)
  rows.append(dict(measurement_cycles=duration,nominal_area=area,steps_per_cycle=steps,C_direct=model.C_direct,C_real=full.real,C_imag=full.imag,error_real=full.real-model.C_direct,background_real=background.real,higher_real=higher.real,background_imag=background.imag,higher_imag=higher.imag,max_norm_error=norm_error))
 return pd.DataFrame(rows)
def longest_chain_order_audit(length=18):
 rows=[]
 for label,j2,field in[('Ising',0.0,1.0),('NI',0.05,1.08385)]:
  for states,area in[(64,0.02),(96,0.02)]+([(64,0.01)]if j2 else[]):
   model=build_model(length,j2,field,states)
   pulse=forward_pulse(model,cycles=96,area=area,steps=STEPS,ramp=RAMP)
   full=pulse['estimate']
   quadratic_model=dict(energies=model.energies,O_sigma=model.O_sigma,O_epsilon_connected=model.O_epsilon_connected,gap_sigma=model.gap_sigma,gap_epsilon=model.gap_epsilon)
   second=quadratic(quadratic_model,96,area,STEPS)/pulse['denominator']
   higher=full-second
   rows.append(dict(model=label,n=states,area=area,delta_real=full.real-model.C_direct,C_imag=full.imag,complex_error_direct=abs(full-model.C_direct),complex_error_quadratic=abs(higher)))
 return pd.DataFrame(rows)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
