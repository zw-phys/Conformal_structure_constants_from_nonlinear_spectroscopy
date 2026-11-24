from pathlib import Path
import numpy as np
import pandas as pd
import pulse_calc as ising
import tci_model_calc as tci
LENGTHS=range(6,27,2)
ISING_BRANCHES=((0.0,1.0,'Ising'),(0.05,1.08385,'Retuned $J_2=0.05$'))
ISING_STATES=64
ISING_CYCLES=96
ISING_RAMP=8
ISING_AREA=0.02
ISING_STEPS=128
TCI_G=0.856
TCI_STATES=96
TCI_CYCLES=96
TCI_RAMP=6
TCI_AREA=0.005
TCI_STEPS=256
OUTPUT=Path(__file__).with_name('sf_3_a.csv')
ISING_COLUMNS=['tag','L','J2','h','n','cycles','ramp','area','actual_envelope_area','steps','quadrature_stride','phi1','phi2','intermediate_detuning','output_detuning','gamma','envelope_scale','phase_offset','T_measure','resolution','lambda_value','C_direct','C_real','C_imag','error_real','error_complex','numerator_real','numerator_imag','denominator_real','denominator_imag','max_norm_error','spectral_cutoff_even','spectral_cutoff_odd','target_gap']
TCI_COLUMNS=['L','g','n_q0_requested','n_even','n_odd','full_q0','measurement_cycles','ramp_cycles','area','steps_per_cycle','gap_sigma','gap_epsilon','gap_ratio','M_sigma','M_epsilon','C_real','C_imag','C_direct','complex_error','full_pc_real','full_pc_imag','calibration_real','calibration_imag','max_norm_error','T_measure','T_total','lambda_value','q0_even_dimension','q0_odd_dimension']
def calculate():
 pieces=[]
 for j2,field,series in ISING_BRANCHES:
  rows=[]
  for length in LENGTHS:
   model=ising.build_model(length,j2,field,ISING_STATES)
   pulse=ising.forward_pulse(model,cycles=ISING_CYCLES,ramp=ISING_RAMP,area=ISING_AREA,steps=ISING_STEPS)
   C=pulse['estimate']
   duration=(ISING_CYCLES+ISING_RAMP)*2*np.pi/model.gap_epsilon
   rows.append(dict(tag='matched_main',L=length,J2=j2,h=field,n=ISING_STATES,cycles=ISING_CYCLES,ramp=ISING_RAMP,area=ISING_AREA,actual_envelope_area=ISING_AREA*(ISING_CYCLES+ISING_RAMP/2)/(ISING_CYCLES+ISING_RAMP),steps=ISING_STEPS,quadrature_stride=1,phi1=0.0,phi2=0.0,intermediate_detuning=0.0,output_detuning=0.0,gamma=0.0,envelope_scale=1.0,phase_offset=0.0,T_measure=ISING_CYCLES*2*np.pi/model.gap_epsilon,resolution=model.gap_epsilon/ISING_CYCLES,lambda_value=ISING_AREA/duration,C_direct=model.C_direct,C_real=C.real,C_imag=C.imag,error_real=C.real-model.C_direct,error_complex=abs(C-model.C_direct),numerator_real=pulse['numerator'].real,numerator_imag=pulse['numerator'].imag,denominator_real=pulse['denominator'].real,denominator_imag=pulse['denominator'].imag,max_norm_error=pulse['max_norm_error'],target_gap=model.gap_epsilon,spectral_cutoff_even=model.static['spectral_cutoff_even'],spectral_cutoff_odd=model.static['spectral_cutoff_odd']))
  frame=pd.DataFrame(rows)[ISING_COLUMNS]
  frame['series']=series
  pieces.append(frame)
 rows=[]
 for length in LENGTHS:
  energy,spin,readout,static=tci.build_model(length,TCI_G,TCI_STATES)
  pulse=tci.finite_pulse(energy,spin,readout,static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'],steps_per_cycle=TCI_STEPS,cycles=TCI_CYCLES,ramp_cycles=TCI_RAMP,area=TCI_AREA)
  C=complex(pulse['C_real'],pulse['C_imag'])
  rows.append(dict(L=length,g=TCI_G,n_q0_requested=TCI_STATES,n_even=static['retained_even'],n_odd=static['retained_odd'],full_q0=static['full_q0'],measurement_cycles=TCI_CYCLES,ramp_cycles=TCI_RAMP,area=TCI_AREA,steps_per_cycle=TCI_STEPS,gap_sigma=static['gap_sigma'],gap_epsilon=static['gap_epsilon'],gap_ratio=static['gap_ratio'],M_sigma=static['M_sigma'],M_epsilon=static['M_epsilon'],C_real=C.real,C_imag=C.imag,C_direct=static['C_direct'],complex_error=abs(C-static['C_direct']),full_pc_real=pulse['S_real'],full_pc_imag=pulse['S_imag'],calibration_real=pulse['K_real'],calibration_imag=pulse['K_imag'],max_norm_error=pulse['max_norm_error'],T_measure=pulse['T_measure'],T_total=pulse['T_total'],lambda_value=pulse['lambda_value'],q0_even_dimension=static['dimension_even'],q0_odd_dimension=static['dimension_odd']))
 frame=pd.DataFrame(rows)[TCI_COLUMNS]
 frame['series']='TCI'
 pieces.append(frame)
 result=pd.concat(pieces,ignore_index=True,sort=False)
 if not np.isfinite(result[['L','C_real','C_imag','C_direct']].to_numpy()).all():
  raise ValueError('Nonfinite pulse coefficient')
 return result
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
