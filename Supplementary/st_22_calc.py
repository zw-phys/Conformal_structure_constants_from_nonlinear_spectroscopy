from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import build_model,forward_pulse,pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_22.csv')
SCAN_PARAMETERS={'L':12,'J2':0.05,'h':1.08385,'n':64,'cycles':96.0,'ramp':8.0,'area':0.02,'steps':128,'quadrature_stride':1,'phi1':0.0,'phi2':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'tag':'retuned_unknown_envelope','envelope_scale':0.98,'phase_offset':0.0},{'tag':'retuned_unknown_envelope','envelope_scale':1.0,'phase_offset':0.0},{'tag':'retuned_unknown_envelope','envelope_scale':1.02,'phase_offset':0.0},{'tag':'retuned_unknown_phase','envelope_scale':1.0,'phase_offset':-0.02},{'tag':'retuned_unknown_phase','envelope_scale':1.0,'phase_offset':0.0},{'tag':'retuned_unknown_phase','envelope_scale':1.0,'phase_offset':0.02}]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.loc[pulse.tag.isin(['retuned_unknown_envelope','retuned_unknown_phase'])].copy()
 pulse['error']=pulse.tag.str.removeprefix('retuned_unknown_')
 pulse['value']=np.where(pulse.error=='envelope',pulse.envelope_scale,pulse.phase_offset)
 pulse=pulse.sort_values(['error','value'])
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 S_1=pulse.denominator_real.to_numpy()+1j*pulse.denominator_imag.to_numpy()
 C=S_pc/S_1
 pulse['C_real']=C.real
 pulse['C_imag']=C.imag
 result=pulse[['error','value','C_real','C_imag']]
 return result
from dataclasses import dataclass,asdict
import math
EXPECTED_C=0.5
NOISE_LENGTH=12
NOISE_J2=0.05
NOISE_FIELD=1.08385
NOISE_STATES_PER_PARITY=64
NOISE_CYCLES=96
NOISE_AREA=0.02
NOISE_RAMP=8
NOISE_STEPS=128
@dataclass(frozen=True)
class NoiseRow:
 raw_shot_noise_fraction:float
 averages_per_phase:int
 n_samples:int
 C_real_median:float
 C_real_q025:float
 C_real_q975:float
 C_imag_median:float
 C_abs_median:float
 probability_real_within_0p01_of_direct:float
 probability_interval_contains_half:bool
def propagate_phase_cycle_noise(signal_pc:complex,calibration:complex,C_direct:float,raw_shot_noise_fraction:float,averages_per_phase:int,n_samples:int,seed:int)->NoiseRow:
 if raw_shot_noise_fraction<0.0:
  raise ValueError('noise fraction must be nonnegative')
 if averages_per_phase<1 or n_samples<100:
  raise ValueError('invalid noise Monte Carlo settings')
 rng=np.random.default_rng(seed)
 sigma_complex_component=raw_shot_noise_fraction*abs(signal_pc)/(2.0*math.sqrt(float(averages_per_phase)))
 noise=sigma_complex_component*(rng.standard_normal(n_samples)+1j*rng.standard_normal(n_samples))
 samples=(signal_pc+noise)/calibration
 real_samples=samples.real
 q025,median,q975=np.quantile(real_samples,[0.025,0.5,0.975])
 return NoiseRow(raw_shot_noise_fraction=float(raw_shot_noise_fraction),averages_per_phase=int(averages_per_phase),n_samples=int(n_samples),C_real_median=float(median),C_real_q025=float(q025),C_real_q975=float(q975),C_imag_median=float(np.median(samples.imag)),C_abs_median=float(np.median(np.abs(samples))),probability_real_within_0p01_of_direct=float(np.mean(np.abs(real_samples-C_direct)<=0.01)),probability_interval_contains_half=bool(q025<=EXPECTED_C<=q975))
def conditional_noise_audit():
 model=build_model(NOISE_LENGTH,NOISE_J2,NOISE_FIELD,NOISE_STATES_PER_PARITY)
 nominal=forward_pulse(model,cycles=NOISE_CYCLES,area=NOISE_AREA,ramp=NOISE_RAMP,steps=NOISE_STEPS)
 rows=[]
 for fraction in[0.01,0.03,0.05]:
  result=propagate_phase_cycle_noise(nominal['numerator'],nominal['denominator'],model.C_direct,fraction,16,20000,11073)
  row=asdict(result)
  rows.append({'r':fraction,'N':16,'C_real_median':row['C_real_median'],'C_real_q025':row['C_real_q025'],'C_real_q975':row['C_real_q975']})
 return pd.DataFrame(rows)
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
