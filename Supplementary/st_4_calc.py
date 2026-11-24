from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import build_model,pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_4.csv')
J2=0.05
FIELD=1.08385
AUDIT_LENGTH=12
AUDIT_STATES_PER_PARITY=64
SCAN_PARAMETERS={'tag':'matched_main','J2':0.05,'h':1.08385,'cycles':96.0,'ramp':8.0,'area':0.02,'steps':128,'quadrature_stride':1,'phi1':0.0,'phi2':0.0,'envelope_scale':1.0,'phase_offset':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'L':6,'n':8},{'L':8,'n':20},{'L':10,'n':56},{'L':12,'n':64},{'L':14,'n':64},{'L':16,'n':64},{'L':18,'n':64},{'L':20,'n':64},{'L':22,'n':64},{'L':24,'n':64},{'L':26,'n':64}]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.loc[(pulse.tag=='matched_main')&(pulse.J2==J2)&(pulse.h==FIELD)].sort_values('L').copy()
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 S_1=pulse.denominator_real.to_numpy()+1j*pulse.denominator_imag.to_numpy()
 C=S_pc/S_1
 pulse['C_real']=C.real
 pulse['C_imag']=C.imag
 result=pulse[['L','C_direct','C_real','C_imag']]
 return result
import math
from dataclasses import dataclass,replace
@dataclass(frozen=True,slots=True)
class LinearInputs:
 gap_sigma:float
 gap_epsilon:float
 M_sigma:float
 M_epsilon:float
@dataclass(frozen=True)
class TimeGrid:
 omega_lock:float
 period_lock:float
 ramp_cycles:float
 measurement_cycles:float
 steps_per_cycle:int
 T_ramp:float
 T_measure:float
 T_total:float
 dt:float
 n_steps:int
def make_time_grid(omega_lock:float,measurement_cycles:float,ramp_cycles:float,steps_per_cycle:int)->TimeGrid:
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
 return TimeGrid(omega_lock=float(omega_lock),period_lock=float(period),ramp_cycles=float(ramp_cycles),measurement_cycles=float(measurement_cycles),steps_per_cycle=int(steps_per_cycle),T_ramp=float(T_ramp),T_measure=float(T_measure),T_total=float(T_total),dt=float(dt),n_steps=int(n_steps))
def ramp_envelope(t:float,T_ramp:float)->float:
 if T_ramp<=0.0 or t>=T_ramp:
  return 1.0
 return math.sin(0.5*math.pi*t/T_ramp)**2
def drive_value(t:float,grid:TimeGrid,omega1:float,omega2:float,lambda1:float,lambda2:float,phi1:float,phi2:float)->float:
 env=ramp_envelope(t,grid.T_ramp)
 return env*(lambda1*math.cos(omega1*t+phi1)+lambda2*math.cos(omega2*t+phi2))
def lockin_weight(t:float,grid:TimeGrid,lockin_gamma:float)->float:
 if t<grid.T_ramp:
  return 0.0
 if lockin_gamma<0.0:
  raise ValueError('lockin_gamma must be nonnegative')
 return math.exp(-lockin_gamma*(t-grid.T_ramp))
def lockin_calibration_C1(model:LinearInputs,grid:TimeGrid,omega1:float,omega2:float,lambda1:float,lambda2:float,phi1:float,phi2:float,lockin_gamma:float)->complex:
 E_sigma=model.gap_sigma
 E_epsilon=model.gap_epsilon
 M_sigma=model.M_sigma
 M_epsilon=model.M_epsilon
 B_cal=M_sigma
 c_sigma=0.0+0j
 c_epsilon=0.0+0j
 acc=0.0+0j
 norm_acc=0.0
 prev_val:complex|None=None
 prev_weight:float|None=None
 prev_t:float|None=None
 def rhs(t:float,cs:complex,ce:complex)->tuple[complex,complex]:
  f=drive_value(t,grid,omega1,omega2,lambda1,lambda2,phi1,phi2)
  dcs=-1j*(E_sigma*cs+f*M_sigma)
  dce=-1j*(E_epsilon*ce+f*B_cal*cs)
  return(dcs,dce)
 for k in range(grid.n_steps+1):
  t=k*grid.dt
  signal=2.0*float((M_epsilon*c_epsilon).real)
  if t>=grid.T_ramp:
   weight=lockin_weight(t,grid,lockin_gamma)
   val=weight*signal*np.exp(-1j*grid.omega_lock*t)
   if prev_val is not None and prev_weight is not None and(prev_t is not None):
    dt_local=t-prev_t
    acc+=0.5*(val+prev_val)*dt_local
    norm_acc+=0.5*(weight+prev_weight)*dt_local
   prev_val=complex(val)
   prev_weight=float(weight)
   prev_t=float(t)
  if k==grid.n_steps:
   break
  dt=grid.dt
  k1s,k1e=rhs(t,c_sigma,c_epsilon)
  k2s,k2e=rhs(t+0.5*dt,c_sigma+0.5*dt*k1s,c_epsilon+0.5*dt*k1e)
  k3s,k3e=rhs(t+0.5*dt,c_sigma+0.5*dt*k2s,c_epsilon+0.5*dt*k2e)
  k4s,k4e=rhs(t+dt,c_sigma+dt*k3s,c_epsilon+dt*k3e)
  c_sigma=c_sigma+dt/6.0*(k1s+2.0*k2s+2.0*k3s+k4s)
  c_epsilon=c_epsilon+dt/6.0*(k1e+2.0*k2e+2.0*k3e+k4e)
 if norm_acc<=0.0:
  raise RuntimeError('zero calibration lock-in normalization')
 return complex(2.0*acc/norm_acc)
def phase_cycle_calibration_C1(model:LinearInputs,grid:TimeGrid,omega1:float,omega2:float,lambda1:float,lambda2:float,phi1:float,phi2:float,lockin_gamma:float)->complex:
 total=0.0+0j
 for dphi1,dphi2,sign in((0.0,0.0,+1.0),(math.pi,0.0,-1.0),(0.0,math.pi,-1.0),(math.pi,math.pi,+1.0)):
  total+=sign*lockin_calibration_C1(model,grid,omega1,omega2,lambda1,lambda2,phi1+dphi1,phi2+dphi2,lockin_gamma)
 return complex(total/4.0)
def unit_coefficient_kernel(inputs,*,measurement_cycles=96,ramp_cycles=8,steps_per_cycle=128,nominal_area=0.02,phi1=0.0,phi2=0.0,gamma=0.0):
 if not isinstance(inputs,LinearInputs):
  raise TypeError('Use LinearInputs; no middle link or Hamiltonian is accepted')
 if any((not np.isfinite(x)or x<=0 for x in(inputs.gap_sigma,inputs.gap_epsilon,inputs.M_sigma,inputs.M_epsilon))):
  raise ValueError('Calibrated gaps and phase-fixed amplitudes must be positive')
 grid=make_time_grid(inputs.gap_epsilon,measurement_cycles,ramp_cycles,steps_per_cycle)
 amplitude=nominal_area/grid.T_total
 return phase_cycle_calibration_C1(inputs,grid,inputs.gap_sigma,inputs.gap_epsilon-inputs.gap_sigma,amplitude,amplitude,phi1,phi2,gamma)
def calibration_audit(inputs=None):
 if inputs is None:
  model=build_model(AUDIT_LENGTH,J2,FIELD,AUDIT_STATES_PER_PARITY)
  inputs=LinearInputs(model.gap_sigma,model.gap_epsilon,model.M_sigma,model.M_epsilon)
 nominal=unit_coefficient_kernel(inputs)
 rows=[{'quantity':'unit_coefficient_kernel','setting':'nominal','value':nominal.real,'imaginary':nominal.imag}]
 step=0.0001
 for leg in['M_sigma','M_epsilon']:
  values=[]
  for sign in[-1,1]:
   altered=replace(inputs,**{leg:getattr(inputs,leg)*np.exp(sign*step)})
   kernel=unit_coefficient_kernel(altered)
   values.append(np.log(abs(nominal/kernel)))
  rows.append({'quantity':'delta_log_absC_per_delta_log_amplitude','setting':leg,'value':(values[1]-values[0])/(2*step)})
 rejected=False
 try:
  LinearInputs(inputs.gap_sigma,inputs.gap_epsilon,inputs.M_sigma,inputs.M_epsilon,middle_link=1.0)
 except TypeError:
  rejected=True
 rows.append({'quantity':'unknown_middle_link_keyword_rejected','value':rejected})
 return pd.DataFrame(rows)
def shared_calibration_variance(covariance):
 covariance=np.asarray(covariance,dtype=float)
 if covariance.shape!=(3,3)or not np.isfinite(covariance).all():
  raise ValueError('A finite 3-by-3 log-input covariance is required')
 if not np.allclose(covariance,covariance.T):
  raise ValueError('Covariance must be symmetric')
 sensitivity=np.array([1.0,-1.0,-2.0])
 return float(sensitivity@covariance@sensitivity)
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
