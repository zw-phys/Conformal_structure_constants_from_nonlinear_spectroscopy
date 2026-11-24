from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_9.csv')
SCAN_PARAMETERS={'tag':'largest_q0_truncation','L':18,'cycles':96.0,'ramp':8.0,'area':0.02,'steps':128,'quadrature_stride':1,'phi1':0.0,'phi2':0.0,'envelope_scale':1.0,'phase_offset':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'J2':0.0,'h':1.0,'n':64},{'J2':0.0,'h':1.0,'n':96},{'J2':0.05,'h':1.0825,'n':64},{'J2':0.05,'h':1.0825,'n':96},{'J2':0.05,'h':1.08385,'n':64},{'J2':0.05,'h':1.08385,'n':96}]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.sort_values(['J2','h','n'])
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 S_1=pulse.denominator_real.to_numpy()+1j*pulse.denominator_imag.to_numpy()
 C=S_pc/S_1
 pulse['C_real']=C.real
 pulse['C_imag']=C.imag
 result=pulse[['J2','h','n','C_real','C_imag']]
 return result
def reintegrate_native_trace(time,phase_cycled_expectation,unit_reference_expectation,omega_lock,weight=None,quadratic_expectation=None):
 time=np.asarray(time,dtype=float)
 signal=np.asarray(phase_cycled_expectation,dtype=float)
 kernel=np.asarray(unit_reference_expectation,dtype=float)
 weights=np.ones_like(time)if weight is None else np.asarray(weight,dtype=float)
 if not(time.ndim==1 and len(time)>1 and np.all(np.diff(time)>0)):
  raise ValueError('Acquisition times must be strictly increasing')
 if any((x.shape!=time.shape or not np.isfinite(x).all()for x in(signal,kernel,weights))):
  raise ValueError('Native trace arrays must have matching finite values')
 phase=np.exp(-1j*omega_lock*time)
 norm=np.trapezoid(weights,x=time)
 if norm<=0:
  raise ValueError('Readout weight has no positive normalization')
 numerator=2*np.trapezoid(weights*signal*phase,x=time)/norm
 denominator=2*np.trapezoid(weights*kernel*phase,x=time)/norm
 if abs(denominator)==0:
  raise ValueError('Zero calibrated unit-reference lock')
 estimate=numerator/denominator
 result={'numerator':numerator,'denominator':denominator,'C':estimate}
 if quadratic_expectation is not None:
  quadratic=np.asarray(quadratic_expectation,dtype=float)
  if quadratic.shape!=time.shape or not np.isfinite(quadratic).all():
   raise ValueError('Quadratic expectation must match the native grid')
  quadratic_C=2*np.trapezoid(weights*quadratic*phase,x=time)/norm/denominator
  result.update(C_quadratic=quadratic_C,full_minus_quadratic=abs(estimate-quadratic_C))
 return result
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
