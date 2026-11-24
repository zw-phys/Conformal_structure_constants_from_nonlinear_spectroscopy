from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from st_1_calc import make_orbits,projected_model
from st_2_calc import laurent_design,fit_response,calculate_response
OUTPUT=Path(__file__).with_name('st_3.csv')
LENGTH=8
J2=0.05
FIELD=1.08385
STATES_PER_PARITY=64
ETA=0.005
DEGREE=2
OFFSETS_ETA=np.array([-6.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,6.0])
FACTORS=(-1.0,0.0,0.75,1.0,1.25)
def linear_amplitude(energy,operator,gap,width=0.005):
 omega=np.linspace(gap-0.03,gap+0.03,241)
 keep=energy>1e-12
 weights=abs(operator[keep,0])**2
 density=np.sum(weights[:,None]*width/np.pi/((omega[None,:]-energy[keep,None])**2+width**2),axis=0)
 x=(omega-gap)/width
 design=np.column_stack([1/(np.pi*width*(1+x*x)),np.ones_like(x),x,x*x])
 coefficients=fit_response(design,density)
 if coefficients[0]<=0:
  raise ValueError('Nonpositive fitted linear spectral weight')
 return float(np.sqrt(coefficients[0]))
def calculate(length=LENGTH):
 energy,spin,readout,static=projected_model(length,J2,FIELD,make_orbits(length),np.random.default_rng(2718),states_per_parity=STATES_PER_PARITY)
 si,ei=(static['retained_even'],1)
 gap_sigma,gap_epsilon=(static['gap_sigma'],static['gap_epsilon'])
 ms=linear_amplitude(energy,spin,gap_sigma)
 me=linear_amplitude(energy,readout,gap_epsilon)
 delta1=np.repeat(OFFSETS_ETA*ETA,len(OFFSETS_ETA))
 delta_sum=np.tile(OFFSETS_ETA*ETA,len(OFFSETS_ETA))
 design=laurent_design(delta1,delta_sum,eta=ETA,degree=DEGREE)
 direct=spin[ei,si]/static['M_sigma']
 rows=[]
 for factor in FACTORS:
  drive=spin.copy()
  drive[ei,si]*=factor
  drive[si,ei]=drive[ei,si].conjugate()
  if not np.array_equal(drive[:,0],spin[:,0]):
   raise AssertionError('Excited-link deformation changed vacuum linear information')
  response=calculate_response(energy,drive,readout,si,gap_sigma,gap_epsilon,delta1,delta_sum,eta=ETA)
  estimate=fit_response(design,response)[0]/(me*ms**2)
  truth=factor*direct
  rows.append(dict(r=factor,C_direct=truth.real,C_freq_real=estimate.real,C_freq_imag=estimate.imag,complex_error=abs(estimate-truth)))
 return pd.DataFrame(rows)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
