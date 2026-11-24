from pathlib import Path
import numpy as np
import pandas as pd
from st_2_calc import calculate_response,fit_response,fresh_models,laurent_design
LENGTHS=tuple(range(6,27,2))
STATES_PER_PARITY=64
ETA=0.005
FIT_OFFSETS=np.linspace(-ETA,ETA,11)
BACKGROUND_DEGREE=2
OUTPUT=Path(__file__).with_name('st_35.csv')
def calculate(lengths=LENGTHS,states_per_parity=STATES_PER_PARITY):
 models=fresh_models(lengths,states_per_parity)
 first_grid,sum_grid=np.meshgrid(FIT_OFFSETS,FIT_OFFSETS)
 first_offsets,sum_offsets=(first_grid.ravel(),sum_grid.ravel())
 design=laurent_design(first_offsets,sum_offsets,ETA,BACKGROUND_DEGREE)
 frequency_rows=[]
 for length,model in sorted(models.items()):
  energies=model['energies']
  response=calculate_response(energies,model['O_sigma'],model['O_epsilon'],model['n_even'],energies[model['sigma_index']],energies[model['epsilon_index']],first_offsets,sum_offsets,ETA)
  coefficients=fit_response(design,response)
  estimate=coefficients[0]/(model['M_epsilon']*model['M_sigma']**2)
  relative_rms=np.sqrt(np.mean(abs(response-design@coefficients)**2))/np.sqrt(np.mean(abs(response)**2))
  frequency_rows.append(dict(L=length,C_freq_real=estimate.real,C_freq_imag=estimate.imag,fit_relative_rms=relative_rms,eta=ETA))
 frequency=pd.DataFrame(frequency_rows)
 first_length=min(models)
 first=models[first_length]
 first_frequency=frequency.loc[frequency.L==first_length].iloc[0]
 first_residue=(first_frequency.C_freq_real+1j*first_frequency.C_freq_imag)*first['M_epsilon']*first['M_sigma']**2
 rows=[]
 for length,model in sorted(models.items()):
  fit=frequency.loc[frequency.L==length].iloc[0]
  estimate=fit.C_freq_real+1j*fit.C_freq_imag
  residue=estimate*model['M_epsilon']*model['M_sigma']**2
  gap_sigma=model['energies'][model['sigma_index']]
  gap_epsilon=model['energies'][model['epsilon_index']]
  rows.append(dict(L=length,x_sigma_over_cft=2*length*gap_sigma/np.pi,x_epsilon_over_cft=length*gap_epsilon/(4*np.pi),M_sigma_relative=model['M_sigma']/first['M_sigma'],M_epsilon_relative=model['M_epsilon']/first['M_epsilon'],residue_relative=residue.real/first_residue.real,exact_energy_leg=first_length/length*np.cos(np.pi/(2*length))/np.cos(np.pi/(2*first_length)),C_direct=model['C_direct'],C_from_chi=estimate.real,C_from_chi_imag=estimate.imag,fit_relative_rms=fit.fit_relative_rms,retained_even=model['n_even'],retained_odd=len(model['energies'])-model['n_even'],eta=fit.eta,complex_error=abs(estimate-model['C_direct'])))
 return pd.DataFrame(rows)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
