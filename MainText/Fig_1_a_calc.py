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
VELOCITY=2.0*COUPLING
ORBIT_BATCH=1<<20
ROW_BATCH=1<<15
OUTPUT=Path(__file__).with_name('Fig_1_a_result.csv')
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
def calculate_gaps(length,coupling,field,rng):
 orbits=translation_orbits(length,ORBIT_BATCH)
 even=parity_hamiltonian(length,coupling,field,orbits,1,ROW_BATCH)
 even_energies=eigsh(even,k=2,which='SA',return_eigenvectors=False,v0=rng.standard_normal(even.shape[0]),tol=1e-11)
 del even
 ground,energy=np.sort(even_energies)
 odd=parity_hamiltonian(length,coupling,field,orbits,-1,ROW_BATCH)
 spin=eigsh(odd,k=1,which='SA',return_eigenvectors=False,v0=rng.standard_normal(odd.shape[0]),tol=1e-11)[0]
 return(spin-ground,energy-ground)
def main():
 rng=np.random.default_rng(1729)
 rows=[]
 for length in LENGTHS:
  gap_sigma,gap_epsilon=calculate_gaps(length,COUPLING,FIELD,rng)
  cft_scale=2.0*np.pi*VELOCITY/length
  rows.append((length,gap_sigma,gap_epsilon,gap_sigma/(cft_scale/8.0),gap_epsilon/cft_scale,1.0))
  print(f'L={length}: Delta_sigma={gap_sigma:.12g}, Delta_epsilon={gap_epsilon:.12g}',flush=True)
 with OUTPUT.open('w',newline='',encoding='utf-8')as file:
  writer=csv.writer(file)
  writer.writerow(('L','gap_sigma','gap_epsilon','x_sigma_over_cft','x_epsilon_over_cft','cft_limit'))
  writer.writerows(rows)
if __name__=='__main__':
 main()
