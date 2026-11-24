from pathlib import Path
import numpy as np
import pandas as pd
import tci_model_calc as tci
LENGTHS=range(6,29,2)
G_VALUES=(0.855,0.856,0.857)
STATES=8
MAX_LENGTHS=(26,28)
MIN_LENGTHS=(10,12,14,16)
SPECTRAL_POWERS=((1.2,2.0),(1.2,2.0,2.4),(1.2,2.0,3.0))
STATIC_POWERS=(0.8,2.0)
CROSSING_LENGTHS=(8,10,12,14)
CROSSING_COUPLINGS=(0.846,0.851,0.855,0.856,0.857,0.861,0.866,0.871,0.876,0.886,0.896,0.906,0.916)
OUTPUT=Path(__file__).with_name('st_44.csv')
def fit(design,values):
 if len(values)-design.shape[1]<2:
  raise ValueError('At least two residual degrees of freedom are required')
 if not np.isfinite(design).all()or not np.isfinite(values).all():
  raise ValueError('Nonfinite fit input')
 return tci.scaled_lstsq(design,values)
def calculate():
 lengths=np.asarray(list(LENGTHS),dtype=float)
 normalized=np.empty((2,len(G_VALUES),len(lengths)))
 direct=np.empty((len(G_VALUES),len(lengths)))
 for j,g in enumerate(G_VALUES):
  for k,length in enumerate(lengths):
   energy,spin,readout,static=tci.build_model(int(length),g,STATES)
   ne=static['retained_even']
   normalized[0,j,k]=static['gap_epsilon']/static['gap_sigma']/(8/3)
   normalized[1,j,k]=energy[ne+1]/static['gap_epsilon']/(35/8)
   direct[j,k]=static['C_direct']
 rows=[]
 for maximum in MAX_LENGTHS:
  used=lengths<=maximum
  L=lengths[used]
  spectra,coefficients=(normalized[:,:,used],direct[:,used])
  slopes=(spectra[:,2]-spectra[:,0])/(G_VALUES[2]-G_VALUES[0])
  for powers in SPECTRAL_POWERS:
   decay=np.column_stack([L**(-p)for p in powers])
   n,k=(len(L),len(powers))
   design=np.zeros((2*n,2*k+1))
   for channel in range(2):
    design[channel*n:(channel+1)*n,channel*k:(channel+1)*k]=decay
    design[channel*n:(channel+1)*n,-1]=slopes[channel]
   for lower in MIN_LENGTHS:
    keep=L>=lower
    if keep.sum()-len(powers)-1<2:
     raise ValueError('Insufficient per-channel residual degrees of freedom')
    selected=np.tile(keep,2)
    b=fit(design[selected],(spectra[:,1]-1).ravel()[selected])
    gc=G_VALUES[1]-b[-1]
    if not G_VALUES[0]<=gc<=G_VALUES[-1]:
     continue
    u=(gc-G_VALUES[1])/(G_VALUES[2]-G_VALUES[1])
    conditioned=coefficients[1]+u*(coefficients[2]-coefficients[0])/2+u*u*(coefficients[2]+coefficients[0]-2*coefficients[1])/2
    size_design=np.column_stack([np.ones(len(L)),*(L**(-p)for p in STATIC_POWERS)])
    c0=fit(size_design[keep],coefficients[1,keep])[0]
    c1=fit(size_design[keep],conditioned[keep])[0]
    rows.append((maximum,gc,c1,c1-c0))
 scan=pd.DataFrame(rows,columns=['L_max','g','C','delta'])
 if scan.empty:
  raise ValueError('No spectral coupling lies inside the computed bracket')
 return scan.groupby('L_max',as_index=False).agg(g_min=('g','min'),g_max=('g','max'),C_min=('C','min'),C_max=('C','max'),delta_min=('delta','min'),delta_max=('delta','max'))
def finite_size_coupling_crossings():
 rows=[]
 g=np.asarray(CROSSING_COUPLINGS)
 for length in CROSSING_LENGTHS:
  differences=np.array([tci.build_model(length,value,STATES)[3]['gap_ratio']-8/3 for value in g])
  crossings=np.flatnonzero(differences[:-1]*differences[1:]<0)
  if len(crossings)!=1:
   raise ValueError('Expected one bracketed gap-ratio crossing')
  i=crossings[0]
  root=g[i]-differences[i]*(g[i+1]-g[i])/(differences[i+1]-differences[i])
  rows.append((length,root,g[i],g[i+1]))
 return pd.DataFrame(rows,columns=['L','g_pseudocritical','bracket_lower','bracket_upper'])
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
