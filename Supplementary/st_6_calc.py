from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
OUTPUT=Path(__file__).resolve().with_name('st_6.csv')
LENGTHS=range(6,28,2)
def calculate():
 return calculate_fresh(LENGTHS)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import brentq
J1=1.0
STATES_PER_PARITY=8
EIGEN_TOLERANCE=1e-11
ORBIT_BATCH=1<<20
ROW_BATCH=1<<15
def rotate(words,shift,length,mask):
 return words<<shift&mask|words>>length-shift
def make_orbits(length):
 half=1<<length-1
 mask=2*half-1
 batches=[]
 for start in range(0,half,ORBIT_BATCH):
  words=np.arange(start,min(start+ORBIT_BATCH,half),dtype=np.uint32)
  for shift in range(1,length):
   shifted=rotate(words,shift,length,mask)
   words=words[(words<=shifted)&(words<=mask^shifted)]
  batches.append(words)
 words=np.concatenate(batches)
 labels=np.arange(1,len(words)+1,dtype=np.int32)
 lookup=np.empty(half,dtype=np.int32)
 period=np.full(len(words),length,dtype=np.int32)
 self_complement=np.zeros(len(words),dtype=bool)
 shifted=words
 lookup[words]=labels
 for shift in range(1,length):
  shifted=rotate(shifted,1,length,mask)
  period[(period==length)&(shifted==words)]=shift
  self_complement|=shifted==mask^words
  flipped=shifted>=half
  key=np.where(flipped,mask^shifted,shifted)
  lookup[key]=np.where(flipped,-labels,labels)
 orbit_size=np.where(self_complement,period,2*period)
 odd_allowed=~self_complement
 d1=length-2*np.bitwise_count(words^rotate(words,1,length,mask)).astype(int)
 d2=length-2*np.bitwise_count(words^rotate(words,2,length,mask)).astype(int)
 magnetization=(length-2*np.bitwise_count(words).astype(int))/length
 return(words,lookup,orbit_size,odd_allowed,d1,d2,magnetization)
def hamiltonian(length,j2,field,orbits,parity):
 words,lookup,orbit_size,odd_allowed,d1,d2,_=orbits
 half=1<<length-1
 mask=2*half-1
 active=np.arange(len(words),dtype=np.int32)
 if parity==-1:
  active=active[odd_allowed]
 dimension=len(active)
 sector_index=np.full(len(words),-1,dtype=np.int32)
 sector_index[active]=np.arange(dimension,dtype=np.int32)
 flips=np.uint32(1)<<np.arange(length,dtype=np.uint32)
 columns=np.empty((dimension,length+1),dtype=np.int32)
 values=np.empty(columns.shape,dtype=np.float64)
 for start in range(0,dimension,ROW_BATCH):
  stop=min(start+ROW_BATCH,dimension)
  source=active[start:stop]
  flipped_words=words[source,None]^flips
  complemented=flipped_words>=half
  key=np.where(complemented,mask^flipped_words,flipped_words)
  labels=lookup[key]
  target=np.abs(labels)-1
  col=sector_index[target]
  amplitude=-field*np.sqrt(orbit_size[source,None]/orbit_size[target])
  if parity==-1:
   amplitude[(labels<0)!=complemented]*=-1
   amplitude[col<0]=0
   col[col<0]=0
  columns[start:stop,:-1]=col
  values[start:stop,:-1]=amplitude
  columns[start:stop,-1]=np.arange(start,stop)
  values[start:stop,-1]=-J1*d1[source]-j2*d2[source]
 offsets=(length+1)*np.arange(dimension+1,dtype=np.int32)
 return csr_matrix((values.ravel(),columns.ravel(),offsets),shape=(dimension,dimension))
def low_states(matrix,rng,states_per_parity=STATES_PER_PARITY):
 dimension=matrix.shape[0]
 if dimension<=states_per_parity:
  energy,vectors=np.linalg.eigh(matrix.toarray())
 else:
  energy,vectors=eigsh(matrix,k=states_per_parity,which='SA',tol=EIGEN_TOLERANCE,ncv=min(dimension,2*states_per_parity+1),v0=rng.standard_normal(dimension))
  order=np.argsort(energy)
  energy,vectors=(energy[order],vectors[:,order])
 residual=max((np.linalg.norm(matrix@vectors[:,i]-energy[i]*vectors[:,i])for i in range(len(energy))))
 if not np.isfinite(residual)or residual>1e-07:
  raise ValueError(f'Invalid eigensolution: maximum residual {residual}')
 return(energy,vectors,residual)
_SPECTRAL_CACHE={}
def spectrum(length,field,n_states=8):
 if length not in _SPECTRAL_CACHE:
  _SPECTRAL_CACHE[length]=make_orbits(length)
 orbits=_SPECTRAL_CACHE[length]
 rng=np.random.default_rng(2718)
 energies,vectors=([],[])
 for parity in(1,-1):
  matrix=hamiltonian(length,0.05,field,orbits,parity)
  e,v,_=low_states(matrix,rng,n_states)
  energies.append(e)
  vectors.append(v)
 _,_,_,odd_allowed,d1,d2,magnetization=orbits
 even,odd=vectors
 vacuum=even[:,0]
 spin_amplitudes=odd.T@(magnetization[odd_allowed]*vacuum[odd_allowed])
 energy_diagonal=((1+1/field)*d1+0.05/field*d2)/length
 energy_amplitudes=even[:,1:].T@(energy_diagonal*vacuum)
 si=int(np.argmax(abs(spin_amplitudes)))
 ei=1+int(np.argmax(abs(energy_amplitudes)))
 return(float(energies[1][si]-energies[0][0]),float(energies[0][ei]-energies[0][0]))
def calculate_fresh(lengths=range(6,28,2)):
 lengths=sorted(set((int(x)for x in lengths)))
 rows=[]
 for length in lengths:
  def gap_ratio_residual(field):
   ds,de=spectrum(length,field)
   return de/ds-8.0
  root=brentq(gap_ratio_residual,1.04,1.11,xtol=2e-11)
  prg=np.nan
  if length+2 in lengths:
   def prg_residual(field):
    return length*spectrum(length,field)[0]-(length+2)*spectrum(length+2,field)[0]
   prg=brentq(prg_residual,1.04,1.11,xtol=2e-11)
  rows.append({'L':length,'h_gap_ratio_8':root,'h_PRG':prg})
 return pd.DataFrame(rows)
def spectral_sequence_fits(roots=None):
 points=calculate()if roots is None else roots
 rows=[]
 for column in['h_gap_ratio_8','h_PRG']:
  sequence=points.dropna(subset=[column])
  for low in[6,8,10,12]:
   subset=sequence.loc[sequence.L>=low]
   lengths=subset.L.to_numpy(dtype=float)
   x=lengths if column=='h_gap_ratio_8'else np.sqrt(lengths*(lengths+2))
   for powers in[(3,),(3,5)]:
    if len(x)<=len(powers)+1:
     continue
    design=np.column_stack([np.ones(len(x)),*(x**(-p)for p in powers)])
    norms=np.linalg.norm(design,axis=0)
    values=subset[column].to_numpy(dtype=float)
    coefficients=np.linalg.lstsq(design/norms,values,rcond=None)[0]/norms
    rows.append({'criterion':column,'L_min':low,'powers':powers,'h_infinity':coefficients[0],'max_residual':np.max(abs(values-design@coefficients))})
 return pd.DataFrame(rows)
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
