from pathlib import Path
import numpy as np
import pandas as pd
from pulse_calc import build_model,forward_pulse
LENGTHS=tuple(range(6,27,2))
MODELS=((0.0,1.0),(0.05,1.08387382))
CYCLES=768
AREA=0.01
STEPS=128
RAMP=8
STATES_PER_PARITY=64
OUTPUT=Path(__file__).with_name('st_38.csv')
def pulse_series(lengths=LENGTHS,models=MODELS,cycles=CYCLES,area=AREA,steps=STEPS,states_per_parity=STATES_PER_PARITY):
 rows=[]
 for j2,field in models:
  for length in lengths:
   model=build_model(length,j2,field,states_per_parity)
   pulse=forward_pulse(model,cycles=cycles,area=area,steps=steps,ramp=RAMP)
   signal,kernel=(pulse['numerator'],pulse['denominator'])
   estimate=signal/kernel
   rows.append(dict(L=length,J2=j2,h=field,**model.static,C_real=estimate.real,C_imag=estimate.imag,S_real=signal.real,S_imag=signal.imag,K_real=kernel.real,K_imag=kernel.imag,max_norm_error=pulse['max_norm_error']))
 return pd.DataFrame(rows)
def calculate(lengths=LENGTHS,cycles=CYCLES,area=AREA,steps=STEPS):
 data=pulse_series(lengths,cycles=cycles,area=area,steps=steps)
 data['pulse_real']=data.C_real
 data['pulse_imag']=data.C_imag
 data['delta_real']=data.C_real-data.C_direct
 quantities=['C_direct','pulse_real','delta_real','pulse_imag']
 rows=data.pivot(index='L',columns='J2',values=quantities)
 rows=rows.reindex(columns=pd.MultiIndex.from_product([quantities,[m[0]for m in MODELS]]))
 rows.columns=[f'{quantity}_{model}'for quantity in quantities for model in('tfic','nnn')]
 result=rows.reset_index()
 result['delta_micro_tfic']=1000000.0*result.delta_real_tfic
 result['delta_micro_nnn']=1000000.0*result.delta_real_nnn
 return result
def earlier_duration_audit(lengths=LENGTHS):
 data=pulse_series(lengths,cycles=96,area=0.02)
 data['absolute_real_discrepancy']=(data.C_real-data.C_direct).abs()
 return data.groupby('J2').absolute_real_discrepancy.max()
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
