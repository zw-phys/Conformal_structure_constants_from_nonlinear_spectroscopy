import csv
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
try:
 from scipy.sparse import csr_matrix
 from scipy.sparse.linalg import eigsh
except ImportError as error:
 if'_spropack'not in str(error)or'__thread_bss'not in str(error):
  raise
 def unavailable_propack(*args,**kwargs):
  raise ImportError('The installed SciPy PROPACK extension could not load')
 svd_module=ModuleType('scipy.sparse.linalg._svdp')
 svd_module._svdp=unavailable_propack
 sys.modules[svd_module.__name__]=svd_module
 from scipy.sparse import csr_matrix
 from scipy.sparse.linalg import eigsh
LENGTH=10
COUPLING=1.0
FIELD=1.0
STATES_PER_PARITY=24
RNG_SEED=1729
ETA=0.005
DISPLAY_OFFSETS=np.linspace(-6.0,6.0,81)
FIT_OFFSETS=np.array([-6.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,6.0])
BACKGROUND_DEGREE=2
OUTPUT=Path(__file__).with_name('Fig_1_b_result.csv')
def parity_hamiltonian(length,coupling,field,parity):
 dimension=1<<length-1
 mask=2*dimension-1
 words=np.arange(dimension,dtype=np.uint32)
 rotated=words<<1&mask|words>>length-1
 bond_sum=length-2*np.bitwise_count(words^rotated).astype(np.int32)
 flipped=words[:,None]^np.uint32(1)<<np.arange(length,dtype=np.uint32)
 complemented=flipped>=dimension
 columns=np.empty((dimension,length+1),dtype=np.int32)
 columns[:,:-1]=np.where(complemented,mask^flipped,flipped)
 columns[:,-1]=words
 values=np.empty(columns.shape,dtype=float)
 values[:,:-1]=-field*np.where(complemented,parity,1)
 values[:,-1]=-coupling*bond_sum
 offsets=(length+1)*np.arange(dimension+1,dtype=np.int32)
 matrix=csr_matrix((values.ravel(),columns.ravel(),offsets),shape=(dimension,dimension))
 return(matrix,words,bond_sum)
def projected_model(length,coupling,field,retained,rng):
 energies,vectors=([],[])
 for parity in(1,-1):
  matrix,words,bond_sum=parity_hamiltonian(length,coupling,field,parity)
  energy,vector=eigsh(matrix,k=retained,which='SA',tol=1e-10,ncv=min(matrix.shape[0],4*retained+1),v0=rng.standard_normal(matrix.shape[0]))
  order=np.argsort(energy)
  energies.append(energy[order])
  vectors.append(vector[:,order])
 even_vectors,odd_vectors=vectors
 magnetization=(length-2*np.bitwise_count(words).astype(np.int32))/length
 spin=even_vectors.T@(magnetization[:,None]*odd_vectors)
 readouts=[]
 for energy,vector in zip(energies,vectors):
  readout=(1.0+coupling/field)*(vector.T@(bond_sum[:,None]*vector))
  readouts.append((readout+np.diag(energy/field))/length)
 ground_readout=readouts[0][0,0]
 readout_even,readout_odd=[matrix-ground_readout*np.eye(retained)for matrix in readouts]
 ground_energy=energies[0][0]
 even_energy,odd_energy=[energy-ground_energy for energy in energies]
 gap_sigma,gap_epsilon=(odd_energy[0],even_energy[1])
 m_sigma,m_epsilon=(abs(spin[0,0]),abs(readout_even[0,1]))
 chain=readout_even[0,:,None]*spin*spin[0,None,:]
 middle=spin[0,:,None]*readout_odd*spin[0,None,:]
 return(even_energy,odd_energy,chain,middle,gap_sigma,gap_epsilon,m_sigma,m_epsilon)
def ordered_response(omega_sum,omega,eta,even_energy,odd_energy,chain,middle):
 forward=omega+1j*eta-odd_energy
 backward=omega+1j*eta+odd_energy
 output_forward=omega_sum+1j*eta-even_energy
 output_backward=omega_sum+1j*eta+even_energy
 difference=omega_sum+1j*eta+odd_energy[:,None]-odd_energy[None,:]
 t1=np.sum(chain/(output_forward[:,None]*forward[None,:]))
 t2=-np.sum(middle/(difference*forward[None,:]))
 t3=-np.sum(middle/(difference*backward[:,None]))
 t4=np.sum(chain/(output_backward[:,None]*backward[None,:]))
 return t1+t2+t3+t4
def two_tone_response(first_offsets,sum_offsets,eta,model):
 even_energy,odd_energy,chain,middle,gap_sigma,gap_epsilon,_,_=model
 response=np.empty(first_offsets.size,dtype=complex)
 for index,(first,total)in enumerate(zip(first_offsets,sum_offsets)):
  omega1=gap_sigma+eta*first
  omega_sum=gap_epsilon+eta*total
  omega2=omega_sum-omega1
  response[index]=ordered_response(omega_sum,omega1,eta,even_energy,odd_energy,chain,middle)+ordered_response(omega_sum,omega2,eta,even_energy,odd_energy,chain,middle)
 return response
def laurent_design(first_offsets,sum_offsets,degree):
 d1,dsum=(first_offsets+1j,sum_offsets+1j)
 columns=[1.0/(d1*dsum)]
 columns.extend((sum_offsets**power/d1 for power in range(degree+1)))
 columns.extend((first_offsets**power/dsum for power in range(degree+1)))
 for total_degree in range(degree+1):
  for power in range(total_degree+1):
   columns.append(first_offsets**power*sum_offsets**(total_degree-power))
 return np.column_stack(columns)
def main():
 model=projected_model(LENGTH,COUPLING,FIELD,STATES_PER_PARITY,np.random.default_rng(RNG_SEED))
 _,_,_,_,gap_sigma,gap_epsilon,m_sigma,m_epsilon=model
 normalization=m_epsilon*m_sigma**2
 fit_first,fit_sum=np.meshgrid(FIT_OFFSETS,FIT_OFFSETS)
 fit_first,fit_sum=(fit_first.ravel(),fit_sum.ravel())
 fit_response=ETA**2*two_tone_response(fit_first,fit_sum,ETA,model)
 design=laurent_design(fit_first,fit_sum,BACKGROUND_DEGREE)
 column_norms=np.linalg.norm(design,axis=0)
 coefficients,_,rank,_=np.linalg.lstsq(design/column_norms,fit_response,rcond=None)
 if rank!=design.shape[1]:
  raise ValueError('The Laurent fit is rank deficient')
 coefficients/=column_norms
 relative_rms=np.linalg.norm(design@coefficients-fit_response)/np.linalg.norm(fit_response)
 first,total=np.meshgrid(DISPLAY_OFFSETS,DISPLAY_OFFSETS)
 first,total=(first.ravel(),total.ravel())
 response=-ETA**2*two_tone_response(first,total,ETA,model)/normalization
 reconstruction=-laurent_design(first,total,BACKGROUND_DEGREE)@coefficients/normalization
 if not np.all(np.isfinite(response))or not np.all(np.isfinite(reconstruction)):
  raise ValueError('The response or reconstructed map contains nonfinite values')
 with OUTPUT.open('w',newline='',encoding='utf-8')as file:
  writer=csv.writer(file)
  writer.writerow(('delta1_eta','deltaSigma_eta','response_real','response_imag','response_abs','fit_real','fit_imag','L','eta','retained_per_parity','gap_sigma','gap_epsilon','M_sigma','M_epsilon','background_degree','R_real','R_imag','fit_relative_rms'))
  for index in range(first.size):
   writer.writerow((first[index],total[index],response[index].real,response[index].imag,abs(response[index]),reconstruction[index].real,reconstruction[index].imag,LENGTH,ETA,STATES_PER_PARITY,gap_sigma,gap_epsilon,m_sigma,m_epsilon,BACKGROUND_DEGREE,coefficients[0].real,coefficients[0].imag,relative_rms))
if __name__=='__main__':
 main()
