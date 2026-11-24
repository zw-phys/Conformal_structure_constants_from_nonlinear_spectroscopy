from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
OUTPUT=Path(__file__).with_name('st_1.csv')
LENGTHS=range(6,27,2)
J1=1.0
STATES_PER_PARITY=64
EIGEN_TOLERANCE=1e-11
ORBIT_BATCH=1<<20
ROW_BATCH=1<<15
def exact_source_norms(length):
 correlations=[1.0]
 for distance in range(1,int(length)):
  i,j=np.indices((distance,distance))
  correlations.append(float(np.linalg.det(1/(length*np.sin(np.pi*(i-j+0.5)/length)))))
 return(sum(correlations)/length,4.0/length)
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
def projected_model(length,j2,field,orbits,rng,*,states_per_parity=STATES_PER_PARITY):
 even_h=hamiltonian(length,j2,field,orbits,1)
 dimension_even=even_h.shape[0]
 even_e,even_v,residual_even=low_states(even_h,rng,states_per_parity)
 del even_h
 odd_h=hamiltonian(length,j2,field,orbits,-1)
 dimension_odd=odd_h.shape[0]
 odd_e,odd_v,residual_odd=low_states(odd_h,rng,states_per_parity)
 del odd_h
 _,_,_,odd_allowed,d1,d2,magnetization=orbits
 bridge=even_v[odd_allowed].T@(magnetization[odd_allowed,None]*odd_v)
 diagonal=(1+J1/field)*d1+j2/field*d2
 qe=(even_v.T@(diagonal[:,None]*even_v)+np.diag(even_e/field))/length
 qo=(odd_v.T@(diagonal[odd_allowed,None]*odd_v)+np.diag(odd_e/field))/length
 spin_source=magnetization[odd_allowed]*even_v[odd_allowed,0]
 sigma_source_norm2=float(spin_source@spin_source)
 energy_source=(diagonal+even_e[0]/field)*even_v[:,0]/length
 energy_source-=even_v[:,0]*float(even_v[:,0]@energy_source)
 epsilon_source_norm2=float(energy_source@energy_source)
 del even_v,odd_v
 sign_even=np.ones(len(even_e))
 sign_odd=np.ones(len(odd_e))
 sign_even[1]=np.sign(qe[1,0])
 sign_odd[0]=np.sign(bridge[0,0])
 bridge*=sign_even[:,None]*sign_odd[None,:]
 qe*=sign_even[:,None]*sign_even[None,:]
 qo*=sign_odd[:,None]*sign_odd[None,:]
 ne,no=(len(even_e),len(odd_e))
 energy=np.concatenate((even_e,odd_e))-even_e[0]
 spin=np.zeros((ne+no,ne+no),dtype=np.complex128)
 spin[:ne,ne:]=bridge
 spin[ne:,:ne]=bridge.T
 readout=np.zeros_like(spin)
 readout[:ne,:ne]=qe
 readout[ne:,ne:]=qo
 readout-=qe[0,0]*np.eye(ne+no)
 gap_sigma,gap_epsilon=(energy[ne],energy[1])
 m_sigma,m_epsilon=(bridge[0,0],qe[1,0])
 if not(0<gap_sigma<gap_epsilon and m_sigma>0 and(m_epsilon>0)):
  raise ValueError(f'Invalid target transitions at L={length}, J2={j2}')
 static=dict(ground_energy=even_e[0],gap_sigma=gap_sigma,gap_epsilon=gap_epsilon,M_sigma=m_sigma,M_epsilon=m_epsilon,B_epsilon_sigma=bridge[1,0],C_direct=bridge[1,0]/m_sigma,dimension_even=dimension_even,dimension_odd=dimension_odd,retained_even=ne,retained_odd=no,spectral_cutoff_even=energy[ne-1],spectral_cutoff_odd=energy[-1],eigen_residual=max(residual_even,residual_odd),sigma_source_norm2=sigma_source_norm2,epsilon_source_norm2=epsilon_source_norm2,f_sigma=m_sigma**2/sigma_source_norm2,f_epsilon=m_epsilon**2/epsilon_source_norm2)
 return(energy,spin,readout,static)
def calculate_fresh(lengths=LENGTHS):
 rows=[]
 for length in lengths:
  orbits=make_orbits(int(length))
  _,_,_,p=projected_model(int(length),0.0,1.0,orbits,np.random.default_rng(2718),states_per_parity=4)
  ns,ne=exact_source_norms(length)
  if not np.allclose([p['sigma_source_norm2'],p['epsilon_source_norm2']],[ns,ne],rtol=1e-08,atol=1e-11):
   raise RuntimeError('Full-basis and exact source norms disagree')
  rows.append(dict(L=length,x_sigma=length*p['gap_sigma']/(4*np.pi),x_epsilon=length*p['gap_epsilon']/(4*np.pi),M_sigma=p['M_sigma'],M_epsilon=p['M_epsilon'],f_sigma=p['f_sigma'],f_epsilon=p['f_epsilon'],C_direct=p['C_direct']))
 return pd.DataFrame(rows)
def calculate(lengths=LENGTHS):
 return calculate_fresh(lengths)
def static_window_audit(static=None):
 static=calculate()if static is None else static
 rows=[]
 for powers in[(1,),(2,),(3,),(1,2),(2,4)]:
  intercepts=[]
  for lower in[6,8,10,12]:
   points=static.loc[static.L>=lower]
   lengths=points.L.to_numpy(dtype=float)
   values=points.C_direct.to_numpy()
   design=np.column_stack([np.ones(len(lengths)),*(lengths**(-p)for p in powers)])
   norms=np.linalg.norm(design,axis=0)
   coefficients=np.linalg.lstsq(design/norms,values,rcond=None)[0]/norms
   intercepts.append(coefficients[0])
   if lower==6:
    n,p=design.shape
    residual=values-design@coefficients
    rss=residual@residual
    aicc=n*np.log(rss/n)+2*p+2*p*(p+1)/(n-p-1)
  rows.append({'powers':'+'.join(map(str,powers)),'window_L_min':6,'window_L_min_max':12,'C_infinity_min':min(intercepts),'C_infinity_max':max(intercepts),'AIC_score_L6':aicc})
 return pd.DataFrame(rows)
def normalization_audit(static=None):
 static=calculate()if static is None else static
 base=static.iloc[0]
 base_residue=base.M_epsilon*base.M_sigma**2*base.C_direct
 rows=[]
 for point in static.itertuples(index=False):
  length=int(point.L)
  exact=4*np.cos(np.pi/(2*length))/length
  ds=point.M_sigma/base.M_sigma/(length/6)**(-1/8)-1
  de=point.M_epsilon/base.M_epsilon/(length/6)**(-1)-1
  residue=point.M_epsilon*point.M_sigma**2*point.C_direct
  dr=residue/base_residue/(length/6)**(-1.25)-1
  row={'L':length,'energy_leg_exact':exact,'energy_leg_exact_difference':point.M_epsilon-exact,'connected_energy_source_norm':4/length,'multiplicative_bias_identity_error':(1+de)*(1+ds)**2*point.C_direct/base.C_direct-(1+dr)}
  normalized=point.M_epsilon*point.C_direct*point.M_sigma**2/(point.M_epsilon*point.M_sigma**2)
  row['three_point_normalized_ratio']=normalized
  row['external_leg_identity_error']=abs(normalized-point.C_direct)
  rows.append(row)
 return pd.DataFrame(rows)
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
