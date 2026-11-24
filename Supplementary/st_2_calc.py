from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from st_1_calc import make_orbits,projected_model
OUTPUT=Path(__file__).with_name('st_2.csv')
LENGTHS=range(6,27,2)
STATES_PER_PARITY=64
ETA=0.005
DEGREE=2
OFFSETS_ETA=(-6.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,6.0)
def laurent_design(delta1,delta_sum,eta=ETA,degree=DEGREE):
 d1=delta1+1j*eta
 ds=delta_sum+1j*eta
 columns=[1.0/(d1*ds)]
 columns+=[delta_sum**q/d1 for q in range(degree+1)]
 columns+=[delta1**p/ds for p in range(degree+1)]
 columns+=[delta1**p*delta_sum**(total-p)+0j for total in range(degree+1)for p in range(total+1)]
 return np.column_stack(columns)
def fit_response(design,response):
 if not np.isfinite(design).all()or not np.isfinite(response).all():
  raise ValueError('Nonfinite Laurent input')
 norms=np.linalg.norm(design,axis=0)
 if np.any(norms==0)or design.shape[0]<design.shape[1]:
  raise ValueError('Empty or underdetermined Laurent design')
 scaled=design/norms
 coefficients,_,rank,_=np.linalg.lstsq(scaled,response,rcond=None)
 if rank!=design.shape[1]:
  raise ValueError('Rank-deficient Laurent fit')
 return coefficients/norms
def response_numerators(energies,spin,energy,n_even):
 ne=n_even
 even=energies[:ne]
 odd=energies[ne:]
 numerators=(energy[0,:ne,None]*spin[:ne,ne:]*spin[ne:,0][None,:],spin[0,ne:,None]*energy[ne:,ne:]*spin[ne:,0][None,:],spin[0,ne:,None]*energy[ne:,ne:]*spin[ne:,0][None,:],spin[0,ne:,None]*spin[ne:,:ne]*energy[:ne,0][None,:])
 return(even,odd,numerators)
def ordered_response(precomputed,omega_sum,omega_first,eta):
 even,odd,numerator=precomputed
 ec,er=(even[:,None],even[None,:])
 oc,ore=(odd[:,None],odd[None,:])
 transition=omega_sum+oc-ore+1j*eta
 terms=(np.sum(numerator[0]/((omega_sum-ec+1j*eta)*(omega_first-ore+1j*eta))),-np.sum(numerator[1]/(transition*(omega_first-ore+1j*eta))),-np.sum(numerator[2]/(transition*(omega_first+oc+1j*eta))),np.sum(numerator[3]/((omega_sum+er+1j*eta)*(omega_first+oc+1j*eta))))
 return complex(sum(terms))
def calculate_response(energies,spin,energy,n_even,gap_sigma,gap_epsilon,delta1,delta_sum,eta=ETA):
 precomputed=response_numerators(energies,spin,energy,n_even)
 values=[]
 for d1,ds in zip(delta1,delta_sum):
  omega1=gap_sigma+d1
  omega_sum=gap_epsilon+ds
  omega2=omega_sum-omega1
  omega_sum=omega1+omega2
  values.append(ordered_response(precomputed,omega_sum,omega1,eta)+ordered_response(precomputed,omega_sum,omega2,eta))
 return np.asarray(values)
def fresh_models(lengths=range(6,27,2),states_per_parity=64):
 models={}
 for length in lengths:
  orbits=make_orbits(int(length))
  energy,spin,readout,static=projected_model(int(length),0.0,1.0,orbits,np.random.default_rng(2718),states_per_parity=states_per_parity)
  models[int(length)]=dict(energies=energy,O_sigma=spin,O_epsilon=readout,n_even=static['retained_even'],sigma_index=static['retained_even'],epsilon_index=1,M_sigma=static['M_sigma'],M_epsilon=static['M_epsilon'],C_direct=static['C_direct'],metadata=static)
 return models
def calculate(models=None,lengths=LENGTHS,states_per_parity=STATES_PER_PARITY):
 models=fresh_models(lengths,states_per_parity)if models is None else models
 delta1=np.repeat(np.asarray(OFFSETS_ETA,dtype=float)*ETA,len(OFFSETS_ETA))
 delta_sum=np.tile(np.asarray(OFFSETS_ETA,dtype=float)*ETA,len(OFFSETS_ETA))
 design=laurent_design(delta1,delta_sum)
 rows=[]
 for length,model in sorted(models.items()):
  energies=model['energies']
  si,ei=(model['sigma_index'],model['epsilon_index'])
  response=calculate_response(energies,model['O_sigma'],model['O_epsilon'],model['n_even'],energies[si],energies[ei],delta1,delta_sum)
  coefficients=fit_response(design,response)
  estimate=coefficients[0]/(model['M_epsilon']*model['M_sigma']**2)
  direct=model['C_direct']
  rows.append({'L':length,'C_direct':direct,'C_freq_real':estimate.real,'C_freq_imag':estimate.imag,'complex_error':abs(estimate-direct),'C_from_chi':estimate.real,'C_from_chi_imag':estimate.imag,'eta':ETA,'fit_relative_rms':np.linalg.norm(design@coefficients-response)/np.linalg.norm(response),'n_even':model['n_even'],'n_odd':len(energies)-model['n_even']})
 return pd.DataFrame(rows)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
