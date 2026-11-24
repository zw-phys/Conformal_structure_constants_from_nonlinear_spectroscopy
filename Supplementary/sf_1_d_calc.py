from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from st_6_calc import calculate_fresh
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sf_1_d.csv'
CRITERIA=[('gap_ratio_8','Gap ratio'),('PRG_L_gap_sigma','PRG crossing')]
FIELD_RANGE_PRECISION=1e-05
DETUNED_FIELD=1.0825
LENGTHS=range(6,28,2)
def spectral_roots():
 fields=calculate_fresh(LENGTHS)
 rows=[]
 for row in fields.itertuples(index=False):
  rows.append(('gap_ratio_8',row.L,row.L,row.h_gap_ratio_8))
  if np.isfinite(row.h_PRG):
   rows.append(('PRG_L_gap_sigma',row.L,row.L+2,row.h_PRG))
 return pd.DataFrame(rows,columns=['criterion','L','L2','h_pseudocritical'])
def fit_spectral_roots(roots):
 fits=[]
 for criterion,_ in CRITERIA:
  for minimum_length in(6,8,10,12):
   sample=roots[(roots.criterion==criterion)&(roots.L>=minimum_length)]
   for powers in((3,),(3,5)):
    if len(sample)<=len(powers)+1:
     continue
    ell=np.sqrt(sample.L*sample.L2).to_numpy()
    design=np.column_stack([np.ones(len(sample))]+[ell**(-p)for p in powers])
    scale=np.linalg.norm(design,axis=0)
    coeff,_,rank,_=np.linalg.lstsq(design/scale,sample.h_pseudocritical,rcond=None)
    if rank!=design.shape[1]:
     raise ValueError('Rank-deficient spectral field fit')
    coefficients=coeff/scale
    residual=sample.h_pseudocritical-design@coefficients
    fits.append({'criterion':criterion,'minimum_length':minimum_length,'powers':powers,'h_infinity':coefficients[0],'rms':np.sqrt(np.mean(residual**2))})
 return pd.DataFrame(fits)
def calculate():
 roots=spectral_roots()
 fits=fit_spectral_roots(roots)
 if not np.isfinite(fits.h_infinity).all():
  raise ValueError('Nonfinite spectral field fit')
 field_range=(np.floor(fits.h_infinity.min()/FIELD_RANGE_PRECISION)*FIELD_RANGE_PRECISION,np.ceil(fits.h_infinity.max()/FIELD_RANGE_PRECISION)*FIELD_RANGE_PRECISION)
 rows=[]
 for criterion,label in CRITERIA:
  for row in roots.loc[roots.criterion==criterion].itertuples(index=False):
   ell=np.sqrt(row.L*row.L2)
   rows.append(('crossing',criterion,label,ell**(-3),row.h_pseudocritical,row.L,row.L2,None))
 rows.append(('band','field_range','Systematic field range',None,field_range[0],None,None,field_range[1]))
 rows.append(('horizontal','detuned_field','',None,DETUNED_FIELD,None,None,None))
 return pd.DataFrame(rows,columns=['kind','series','label','x','y','L','L2','y_upper'])
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
