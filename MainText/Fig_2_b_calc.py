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
LENGTHS=range(6,27,2)
COUPLING=1.0
FIELD=COUPLING
STATES_PER_PARITY=32
RNG_SEED=1729
ETA=0.005
FIT_HALF_WIDTH=ETA
FIT_POINTS=11
BACKGROUND_DEGREE=2
STATIC_CORRECTION_POWERS=((1,),(2,),(3,),(1,2),(2,4))
STATIC_MINIMUM_LENGTHS=(6,8,10,12)
ORBIT_BATCH=1<<20
ROW_BATCH=1<<15
OUTPUT=Path(__file__).with_name('Fig_2_b_result.csv')
def rotation(states,shift,length,mask):
 return states<<shift&mask|states>>length-shift
def translation_orbits(length,batch_size):
 half=1<<length-1
 mask=2*half-1
 batches=[]
 for start in range(0,half,batch_size):
  words=np.arange(start,min(start+batch_size,half),dtype=np.uint32)
  for shift in range(1,length):
   rotated=rotation(words,shift,length,mask)
   words=words[(words<=rotated)&(words<=mask^rotated)]
  batches.append(words)
 representatives=np.concatenate(batches)
 orbit_labels=np.arange(1,representatives.size+1,dtype=np.int32)
 lookup=np.empty(half,dtype=np.int32)
 period=np.full(representatives.size,length,dtype=np.int32)
 self_complement=np.zeros(representatives.size,dtype=bool)
 rotated=representatives
 lookup[rotated]=orbit_labels
 for shift in range(1,length):
  rotated=rotation(rotated,1,length,mask)
  period[(period==length)&(rotated==representatives)]=shift
  self_complement|=rotated==mask^representatives
  complement=rotated>=half
  key=np.where(complement,mask^rotated,rotated)
  lookup[key]=np.where(complement,-orbit_labels,orbit_labels)
 orbit_size=np.where(self_complement,period,2*period)
 return(representatives,lookup,orbit_size,~self_complement)
def parity_hamiltonian(length,coupling,field,orbits,parity,batch_size):
 representatives,lookup,orbit_size,odd_allowed=orbits
 half=1<<length-1
 mask=2*half-1
 active=np.arange(representatives.size,dtype=np.int32)
 if parity==-1:
  active=active[odd_allowed]
 dimension=active.size
 sector_index=np.full(representatives.size,-1,dtype=np.int32)
 sector_index[active]=np.arange(dimension,dtype=np.int32)
 flip_masks=np.uint32(1)<<np.arange(length,dtype=np.uint32)
 columns=np.empty((dimension,length+1),dtype=np.int32)
 values=np.empty(columns.shape,dtype=np.float64)
 offsets=(length+1)*np.arange(dimension+1,dtype=np.int32)
 for start in range(0,dimension,batch_size):
  stop=min(start+batch_size,dimension)
  source=active[start:stop]
  words=representatives[source]
  flipped=words[:,None]^flip_masks
  complement=flipped>=half
  key=np.where(complement,mask^flipped,flipped)
  label=lookup[key]
  target=np.abs(label)-1
  column=sector_index[target]
  amplitude=-field*np.sqrt(orbit_size[source,None]/orbit_size[target])
  if parity==-1:
   spin_flip=(label<0)!=complement
   amplitude[spin_flip]*=-1.0
   amplitude[column<0]=0.0
   column[column<0]=0
  walls=np.bitwise_count(words^rotation(words,1,length,mask)).astype(np.int32)
  columns[start:stop,:-1]=column
  values[start:stop,:-1]=amplitude
  columns[start:stop,-1]=np.arange(start,stop,dtype=np.int32)
  values[start:stop,-1]=-coupling*(length-2*walls)
 return csr_matrix((values.ravel(),columns.ravel(),offsets),shape=(dimension,dimension))
def solve_sector(matrix,retained,rng):
 if matrix.shape[0]<=retained:
  return np.linalg.eigh(matrix.toarray())
 energies,vectors=eigsh(matrix,k=retained,which='SA',tol=1e-11,v0=rng.standard_normal(matrix.shape[0]))
 order=np.argsort(energies)
 return(energies[order],vectors[:,order])
def project_probes(length,coupling,field,orbits,even_vectors,odd_vectors,even_energy,odd_energy):
 words,_,_,odd_allowed=orbits
 mask=(1<<length)-1
 walls=np.bitwise_count(words^rotation(words,1,length,mask)).astype(np.int32)
 bond_sum=length-2*walls
 magnetization=(length-2*np.bitwise_count(words).astype(np.int32))/length
 spin=even_vectors[odd_allowed].T@(magnetization[odd_allowed,None]*odd_vectors)
 bond_factor=1.0+coupling/field
 readout_even=bond_factor*(even_vectors.T@(bond_sum[:,None]*even_vectors))
 readout_odd=bond_factor*(odd_vectors.T@(bond_sum[odd_allowed,None]*odd_vectors))
 readout_even=(readout_even+np.diag(even_energy/field))/length
 readout_odd=(readout_odd+np.diag(odd_energy/field))/length
 ground_readout=readout_even[0,0]
 readout_even-=ground_readout*np.eye(even_energy.size)
 readout_odd-=ground_readout*np.eye(odd_energy.size)
 return(spin,readout_even,readout_odd)
def ordered_response(omega_sum,omega,eta,even_energy,odd_energy,chain,middle):
 forward=omega+1j*eta-odd_energy
 backward=omega+1j*eta+odd_energy
 output_forward=omega_sum+1j*eta-even_energy
 output_backward=omega_sum+1j*eta+even_energy
 difference_pole=omega_sum+1j*eta+odd_energy[:,None]-odd_energy[None,:]
 t1=np.sum(chain/(output_forward[:,None]*forward[None,:]))
 t2=-np.sum(middle/(difference_pole*forward[None,:]))
 t3=-np.sum(middle/(difference_pole*backward[:,None]))
 t4=np.sum(chain/(output_backward[:,None]*backward[None,:]))
 return t1+t2+t3+t4
def extract_residue(delta_first,delta_sum,eta,response,degree):
 u,v=(delta_first/eta,delta_sum/eta)
 d1,dsum=(u+1j,v+1j)
 columns=[1.0/(d1*dsum)]
 for power in range(degree+1):
  columns.extend([v**power/d1,u**power/dsum])
 for total_degree in range(degree+1):
  for power in range(total_degree+1):
   columns.append(u**power*v**(total_degree-power))
 design=np.column_stack(columns)
 scale=np.linalg.norm(design,axis=0)
 data=eta**2*response
 coefficients,_,rank,_=np.linalg.lstsq(design/scale,data,rcond=None)
 if rank!=design.shape[1]:
  raise ValueError('The Laurent design is rank deficient')
 residual=data-design/scale@coefficients
 relative_rms=np.linalg.norm(residual)/np.linalg.norm(data)
 return(coefficients[0]/scale[0],relative_rms)
def calculate_size(length,rng):
 orbits=translation_orbits(length,ORBIT_BATCH)
 even_matrix=parity_hamiltonian(length,COUPLING,FIELD,orbits,1,ROW_BATCH)
 even_dimension=even_matrix.shape[0]
 even_energy,even_vectors=solve_sector(even_matrix,STATES_PER_PARITY,rng)
 del even_matrix
 odd_matrix=parity_hamiltonian(length,COUPLING,FIELD,orbits,-1,ROW_BATCH)
 odd_dimension=odd_matrix.shape[0]
 odd_energy,odd_vectors=solve_sector(odd_matrix,STATES_PER_PARITY,rng)
 del odd_matrix
 spin,readout_even,readout_odd=project_probes(length,COUPLING,FIELD,orbits,even_vectors,odd_vectors,even_energy,odd_energy)
 del orbits,even_vectors,odd_vectors
 ground_energy=even_energy[0]
 even_energy=even_energy-ground_energy
 odd_energy=odd_energy-ground_energy
 gap_sigma,gap_epsilon=(odd_energy[0],even_energy[1])
 spin_ground=spin[0,:]
 readout_ground=readout_even[0,:]
 m_sigma,m_epsilon=(abs(spin_ground[0]),abs(readout_ground[1]))
 normalization=m_epsilon*m_sigma**2
 if normalization<=0 or not 0<gap_sigma<gap_epsilon:
  raise ValueError(f'Invalid target transition at L={length}')
 chain=readout_ground[:,None]*spin*spin_ground[None,:]
 middle=spin_ground[:,None]*readout_odd*spin_ground[None,:]
 offsets=np.linspace(-FIT_HALF_WIDTH,FIT_HALF_WIDTH,FIT_POINTS)
 first_window,sum_window=np.meshgrid(offsets,offsets)
 delta_first,delta_sum=(first_window.ravel(),sum_window.ravel())
 response=np.empty(delta_first.size,dtype=np.complex128)
 for point,(shift_first,shift_sum)in enumerate(zip(delta_first,delta_sum)):
  omega_sum=gap_epsilon+shift_sum
  omega1=gap_sigma+shift_first
  omega2=omega_sum-omega1
  response[point]=ordered_response(omega_sum,omega1,ETA,even_energy,odd_energy,chain,middle)+ordered_response(omega_sum,omega2,ETA,even_energy,odd_energy,chain,middle)
 residue,relative_rms=extract_residue(delta_first,delta_sum,ETA,response,BACKGROUND_DEGREE)
 c_from_chi=residue/normalization
 c_direct=chain[1,0]/normalization
 if not np.all(np.isfinite([c_from_chi.real,c_from_chi.imag,c_direct,relative_rms])):
  raise ValueError(f'Nonfinite response result at L={length}')
 return{'L':length,'inverse_L2':1.0/length**2,'gap_sigma':gap_sigma,'gap_epsilon':gap_epsilon,'M_sigma':m_sigma,'M_epsilon':m_epsilon,'R_real':residue.real,'R_imag':residue.imag,'C_from_chi':c_from_chi.real,'C_from_chi_imag':c_from_chi.imag,'C_direct':c_direct,'fit_relative_rms':relative_rms,'retained_even':even_energy.size,'retained_odd':odd_energy.size,'dimension_even':even_dimension,'dimension_odd':odd_dimension,'eta':ETA}
def static_limit_range(lengths,direct_values,corrections,minimum_lengths):
 intercepts=[]
 for powers in corrections:
  for minimum_length in minimum_lengths:
   keep=lengths>=minimum_length
   l_inverse=1.0/lengths[keep]
   columns=[np.ones(l_inverse.size)]
   columns.extend((l_inverse**power for power in powers))
   design=np.column_stack(columns)
   fitted,_,rank,_=np.linalg.lstsq(design,direct_values[keep],rcond=None)
   if rank!=design.shape[1]:
    raise ValueError('The static finite-size fit is rank deficient')
   intercepts.append(fitted[0])
 return(min(intercepts),max(intercepts))
def write_result(rows,output):
 lengths=np.array([row['L']for row in rows])
 inverse_l2=np.array([row['inverse_L2']for row in rows])
 direct_values=np.array([row['C_direct']for row in rows])
 static_lower,static_upper=static_limit_range(lengths,direct_values,STATIC_CORRECTION_POWERS,STATIC_MINIMUM_LENGTHS)
 coefficients=np.polynomial.polynomial.polyfit(inverse_l2,[row['C_from_chi']for row in rows],2)
 for row in rows:
  row['C_fit']=np.polynomial.polynomial.polyval(row['inverse_L2'],coefficients)
  row['C_inf_fit'],row['fit_L_minus_2'],row['fit_L_minus_4']=coefficients
  row['C_cft']=0.5
  row['static_C_inf_lower']=static_lower
  row['static_C_inf_upper']=static_upper
 with output.open('w',newline='',encoding='utf-8')as file:
  writer=csv.DictWriter(file,fieldnames=list(rows[0]))
  writer.writeheader()
  writer.writerows(rows)
def main():
 rng=np.random.default_rng(RNG_SEED)
 rows=[]
 for length in LENGTHS:
  row=calculate_size(length,rng)
  rows.append(row)
  print(f"L={length}: C from response={row['C_from_chi']:.10f}, direct C={row['C_direct']:.10f}",flush=True)
 write_result(rows,OUTPUT)
if __name__=='__main__':
 main()
