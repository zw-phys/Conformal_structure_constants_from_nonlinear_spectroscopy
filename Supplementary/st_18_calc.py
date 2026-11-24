from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_18.csv')
TAG_PREFIX='retuned_detuning_'
SCAN_PARAMETERS={'L':12,'J2':0.05,'h':1.08385,'n':64,'cycles':96.0,'ramp':8.0,'area':0.02,'steps':128,'quadrature_stride':1,'phi1':0.0,'phi2':0.0,'envelope_scale':1.0,'phase_offset':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':-1.0,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':-0.5,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':-0.25,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':-0.1,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':-0.05,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':0.0,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':0.05,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':0.1,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':0.25,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':0.5,'output_detuning_units':0.0},{'tag':'retuned_detuning_intermediate','intermediate_detuning_units':1.0,'output_detuning_units':0.0},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':-1.0},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':-0.5},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':-0.25},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':-0.1},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':-0.05},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':0.0},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':0.05},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':0.1},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':0.25},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':0.5},{'tag':'retuned_detuning_output','intermediate_detuning_units':0.0,'output_detuning_units':1.0}]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.loc[pulse.tag.isin([TAG_PREFIX+'intermediate',TAG_PREFIX+'output'])].copy()
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 S_1=pulse.denominator_real.to_numpy()+1j*pulse.denominator_imag.to_numpy()
 C=S_pc/S_1
 pulse['C_real']=C.real
 pulse['C_imag']=C.imag
 pulse['detuning']=pulse.tag.str.removeprefix(TAG_PREFIX)
 delta=np.where(pulse.detuning=='intermediate',pulse.intermediate_detuning,pulse.output_detuning)
 on_resonance=(pulse.intermediate_detuning==0)&(pulse.output_detuning==0)
 nominal_resolution=pulse.loc[on_resonance,'resolution'].unique().item()
 pulse['detuning_resolution_units']=delta/nominal_resolution
 pulse['S_1_abs']=np.abs(S_1)
 pulse=pulse.sort_values(['detuning','detuning_resolution_units'])
 result=pulse[['detuning','detuning_resolution_units','C_real','C_imag','S_1_abs']]
 return result
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
