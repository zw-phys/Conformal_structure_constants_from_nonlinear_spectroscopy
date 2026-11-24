from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from st_1_calc import make_orbits,projected_model
from st_33_calc import PHASE_SHIFTS,fresh_calibration,complex_json,as_complex,extract
OUTPUT=Path(__file__).with_name('st_23.csv')
LENGTH=8
J2=0.05
FIELD=1.08385
STATES_PER_PARITY=64
AREAS=(0.02,0.1,0.2,0.4,0.6)
MEASURE_CYCLES=96
RAMP_CYCLES=8
STEPS_PER_CYCLE=128
def _propagate_locks(model,protocol,phase_shifts):
 E,Os,Oe=(model['energies'],model['O_sigma'],model['O_epsilon_connected'])
 p=protocol
 n=int(p['n_steps'])
 times=np.linspace(0.0,p['T_total'],n+1)
 dt=times[1]
 state=np.zeros((len(E),4),complex)
 state[0]=1.0
 observations=np.empty((n+1,4))
 shifts=np.asarray(phase_shifts)
 def rhs(t,amplitude):
  envelope=1.0 if t>=p['T_ramp']else np.sin(np.pi*t/(2*p['T_ramp']))**2
  drive=envelope*(p['lambda1']*np.cos(p['omega1']*t+p['phi1']+shifts[:,0])+p['lambda2']*np.cos(p['omega2']*t+p['phi2']+shifts[:,1]))
  rotate=np.exp(-1j*E*t)
  return-1j*rotate.conj()[:,None]*(Os@(rotate[:,None]*amplitude))*drive[None,:]
 for k,t in enumerate(times):
  physical=np.exp(-1j*E*t)[:,None]*state
  observations[k]=np.sum(physical.conj()*(Oe@physical),axis=0).real
  if k==n:
   break
  k1=rhs(t,state)
  k2=rhs(t+dt/2,state+dt*k1/2)
  k3=rhs(t+dt/2,state+dt*k2/2)
  k4=rhs(t+dt,state+dt*k3)
  state+=dt*(k1+2*k2+2*k3+k4)/6
  state/=np.sqrt(np.sum(abs(state)**2,axis=0))[None,:]
 keep=times>=p['T_ramp']-1e-12
 weights=np.exp(-p['readout_gamma']*(times[keep]-p['T_ramp']))
 return 2*np.trapezoid(weights[:,None]*observations[keep]*np.exp(-1j*p['omega_lock']*times[keep])[:,None],x=times[keep],axis=0)/np.trapezoid(weights,x=times[keep])
def pulse_protocol(calibration,cycles,area,steps=128,ramp=8):
 period=2*np.pi/calibration['gap_epsilon']
 total=(cycles+ramp)*period
 return dict(omega1=calibration['gap_sigma'],omega2=calibration['gap_epsilon']-calibration['gap_sigma'],omega_lock=calibration['gap_epsilon'],lambda1=area/total,lambda2=area/total,phi1=0.0,phi2=0.0,T_ramp=ramp*period,T_measure=cycles*period,T_total=total,n_steps=int((cycles+ramp)*steps),readout_gamma=0.0)
def calculate(length=LENGTH,areas=AREAS,cycles=MEASURE_CYCLES,steps=STEPS_PER_CYCLE):
 energy,spin,readout,static=projected_model(length,J2,FIELD,make_orbits(length),np.random.default_rng(2718),states_per_parity=STATES_PER_PARITY)
 model=dict(energies=energy,O_sigma=spin,O_epsilon_connected=readout)
 calibration=fresh_calibration(energy,spin,readout,static)
 rows=[]
 for area in areas:
  protocol=pulse_protocol(calibration,cycles,area,steps,RAMP_CYCLES)
  locks=_propagate_locks(model,protocol,PHASE_SHIFTS)
  record=dict(schema_version='1.0',case_id=f'NI_L{length}_area{area}',calibration=calibration,protocol=protocol,measured_locks=[complex_json(z)for z in locks])
  recovered=extract(record,internal_refinement=2)
  signal=as_complex(recovered['phase_cycled_signal'])
  estimate=as_complex(recovered['estimate'])
  rows.append(dict(area=area,held_out=area==0.02,S_pc_abs=abs(signal),C_real=estimate.real,C_imag=estimate.imag))
 return pd.DataFrame(rows)
def heldout_area_audit(table=None):
 table=calculate()if table is None else table
 train=table.loc[~table.held_out].reset_index(drop=True)
 weak=table.loc[table.held_out].iloc[0]
 target=complex(weak.C_real,weak.C_imag)
 rows=[]
 for degree in[1,2]:
  for indices in[[0,1,2,3],[0,1,2],[1,2,3]]:
   points=train.iloc[indices]
   x=points.area.to_numpy()**2
   values=points.C_real.to_numpy()+1j*points.C_imag.to_numpy()
   design=np.column_stack([x**power for power in range(degree+1)])
   coefficients,_,rank,_=np.linalg.lstsq(design,values,rcond=None)
   prediction=sum((c*weak.area**(2*power)for power,c in enumerate(coefficients)))
   rows.append({'degree':degree,'training_areas':tuple(points.area),'rank':rank,'residual_dof':len(points)-degree-1,'holdout_prediction_real':prediction.real,'holdout_prediction_imag':prediction.imag,'holdout_complex_error':abs(prediction-target)})
 return pd.DataFrame(rows)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
