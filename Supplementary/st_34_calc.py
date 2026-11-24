from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from st_1_calc import make_orbits,projected_model
from st_33_calc import PHASE_SHIFTS,PHASE_SIGNS,fresh_calibration,unit_kernel
from st_23_calc import pulse_protocol,_propagate_locks
OUTPUT=Path(__file__).with_name('st_34.csv')
LENGTH=8
J2=0.05
FIELD=1.08385
STATES_PER_PARITY=64
MEASURE_CYCLES=384
NOMINAL_AREA=0.005
STEPS_PER_CYCLE=128
TRAJECTORY_POINTS=4097
PREPARATION_RATIO=0.02
DRIVE_RATIOS=(0.02,0.01)
FOLLOWUP_PHASE=0.37
RTOL=2e-10
ATOL=2e-12
CARRIER_STEPS=32
def evolve(energy,spin,initial,amplitude,omega,phase,duration):
 times=np.linspace(0.0,duration,TRAJECTORY_POINTS)
 def rhs(time,state):
  rotation=np.exp(-1j*energy*time)
  return-1j*amplitude*np.cos(omega*time+phase)*rotation.conj()*(spin@(rotation*state))
 solution=solve_ivp(rhs,(0.0,duration),initial,t_eval=times,method='DOP853',rtol=RTOL,atol=ATOL,max_step=2*np.pi/omega/CARRIER_STEPS)
 if not solution.success:
  raise RuntimeError(solution.message)
 norms=np.sum(abs(solution.y)**2,axis=0)
 if not np.isfinite(norms).all()or np.max(abs(norms-1))>1e-07:
  raise ValueError('Prepared-state trajectory has an invalid norm')
 return(times,solution.y,norms)
def calculate(length=LENGTH):
 energy,spin,readout,static=projected_model(length,J2,FIELD,make_orbits(length),np.random.default_rng(2718),states_per_parity=STATES_PER_PARITY)
 calibration=fresh_calibration(energy,spin,readout,static)
 model=dict(energies=energy,O_sigma=spin,O_epsilon_connected=readout)
 protocol=pulse_protocol(calibration,MEASURE_CYCLES,NOMINAL_AREA,STEPS_PER_CYCLE)
 locks=_propagate_locks(model,protocol,PHASE_SHIFTS)
 kernel=unit_kernel(calibration,protocol,internal_refinement=2)
 coefficient=PHASE_SIGNS@locks/kernel
 predicted_link=calibration['M_sigma']*coefficient.real
 if not np.isfinite(predicted_link)or predicted_link==0:
  raise ValueError('Invalid pulse-derived target matrix element')
 prep_omega=calibration['gap_sigma']
 prep_amplitude=PREPARATION_RATIO*prep_omega
 prep_duration=np.pi/(prep_amplitude*calibration['M_sigma'])
 drive_omega=calibration['gap_epsilon']-calibration['gap_sigma']
 protocols=[(ratio,ratio*drive_omega,2*np.pi/(ratio*drive_omega*abs(predicted_link)))for ratio in DRIVE_RATIOS]
 initial=np.zeros(len(energy),complex)
 initial[0]=1.0
 times,amplitudes,_=evolve(energy,spin,initial,prep_amplitude,prep_omega,0.0,prep_duration)
 prepared=np.exp(-1j*energy*times[-1])*amplitudes[:,-1]
 si,ei=(static['retained_even'],1)
 rows=[]
 for ratio,amplitude,duration in protocols:
  times,amplitudes,norms=evolve(energy,spin,prepared,amplitude,drive_omega,FOLLOWUP_PHASE,duration)
  omega=amplitude*predicted_link
  predicted_population=np.sin(omega*times/2)**2
  predicted_coherence=-1j*np.exp(-1j*FOLLOWUP_PHASE)*np.sin(omega*times)
  population=abs(amplitudes[ei])**2
  coherence=2*amplitudes[ei]*amplitudes[si].conj()
  outside=norms-population-abs(amplitudes[si])**2
  rows.append((ratio,omega,duration,np.max(abs(population-predicted_population)),np.max(abs(coherence-predicted_coherence)),np.max(outside)))
 return pd.DataFrame(rows,columns=['drive_ratio','Omega_sigma_epsilon','duration','max_population_error','max_complex_coherence_error','max_population_outside_target_pair'])
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
