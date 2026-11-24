from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'sf_4_b.csv'
LENGTH=14
N_EVEN=N_ODD=24
ETA=0.005
BACKGROUND_DEGREE=2
GRID_POINTS=100
COORDINATES=np.linspace(-6.0,6.0,GRID_POINTS)
FIT_POINTS_PER_AXIS=11
DEFAULT_FIT_OFFSETS=np.array([-6,-4,-3,-2,-1,0,1,2,3,4,6],dtype=float)
from sf_4_a_calc import projected_operators
def projected_model(length=LENGTH,states=N_EVEN):
 e,o,s,z=projected_operators(length,states)
 ne=len(e)
 if not 0<o[0]<e[1]or not np.isfinite(s).all()or(not np.isfinite(z).all()):
  raise ValueError('Invalid fresh Ising spectrum or projected probes')
 for matrix in(s,z):
  np.testing.assert_allclose(matrix,matrix.T,atol=1e-12)
 numerators=(z[0,:ne,None]*s[:ne,ne:]*s[ne:,0][None,:],s[0,ne:,None]*z[ne:,ne:]*s[ne:,0][None,:],s[0,ne:,None]*z[ne:,ne:]*s[ne:,0][None,:],s[0,ne:,None]*s[ne:,:ne]*z[:ne,0][None,:])
 return(e,o,numerators,s,z)
def ordered_response(model,omega_sum,omega_first,eta):
 e,o,n=model[:3]
 transition=omega_sum+o[:,None]-o[None,:]+1j*eta
 t1=np.sum(n[0]/((omega_sum-e[:,None]+1j*eta)*(omega_first-o[None,:]+1j*eta)))
 t2=-np.sum(n[1]/(transition*(omega_first-o[None,:]+1j*eta)))
 t3=-np.sum(n[2]/(transition*(omega_first+o[:,None]+1j*eta)))
 t4=np.sum(n[3]/((omega_sum+e[None,:]+1j*eta)*(omega_first+o[:,None]+1j*eta)))
 return complex(t1+t2+t3+t4)
def response(model,delta1,delta_sum,eta):
 first=model[1][0]+delta1
 second=model[0][1]+delta_sum-first
 total=first+second
 return ordered_response(model,total,first,eta)+ordered_response(model,total,second,eta)
def laurent_row(delta1,delta_sum,eta,degree):
 d1,ds=(delta1+1j*eta,delta_sum+1j*eta)
 row=[1.0/(d1*ds)]
 row.extend((delta_sum**q/d1 for q in range(degree+1)))
 row.extend((delta1**p/ds for p in range(degree+1)))
 row.extend((delta1**p*delta_sum**(total-p)+0j for total in range(degree+1)for p in range(total+1)))
 return row
def fit_laurent(model,eta=ETA,width=6,degree=BACKGROUND_DEGREE,offsets=None):
 if offsets is None:
  offsets=np.linspace(-width,width,FIT_POINTS_PER_AXIS)
 design,values=([],[])
 for x in offsets:
  for y in offsets:
   d1,ds=(float(x)*eta,float(y)*eta)
   design.append(laurent_row(d1,ds,eta,degree))
   values.append(response(model,d1,ds,eta))
 design,values=(np.asarray(design),np.asarray(values))
 if not np.isfinite(design).all()or not np.isfinite(values).all():
  raise ValueError('Nonfinite fit input')
 norms=np.linalg.norm(design,axis=0)
 if design.shape[0]<design.shape[1]or np.any(norms==0):
  raise ValueError('Underdetermined design or zero column')
 scaled=design/norms
 coeff,_,rank,singular=np.linalg.lstsq(scaled,values,rcond=None)
 if rank!=design.shape[1]:
  raise ValueError('Rank-deficient Laurent fit')
 coeff=coeff/norms
 residual=values-design@coeff
 rms=float(np.sqrt(np.mean(abs(residual)**2)))
 metadata=dict(singular_min_scaled=singular[-1],singular_max_scaled=singular[0],rank=rank,n_rows=len(values),n_columns=design.shape[1],rms_abs=rms,relative_rms=rms/np.sqrt(np.mean(abs(values)**2)),condition_number_scaled=singular[0]/singular[-1])
 return(coeff,metadata)
def calculate_grid():
 model=projected_model()
 coefficients,meta=fit_laurent(model,offsets=DEFAULT_FIT_OFFSETS)
 full=np.empty((GRID_POINTS,GRID_POINTS),dtype=complex)
 fitted=np.empty_like(full)
 for i,x in enumerate(COORDINATES):
  for j,y in enumerate(COORDINATES):
   d1,ds=(x*ETA,y*ETA)
   full[i,j]=response(model,d1,ds,ETA)
   fitted[i,j]=np.asarray(laurent_row(d1,ds,ETA,BACKGROUND_DEGREE))@coefficients
 residual=full-fitted
 if not np.isfinite(residual).all():
  raise ValueError('Nonfinite frequency-grid residual')
 i,j=np.indices(residual.shape)
 ds,d1=np.meshgrid(COORDINATES,COORDINATES)
 result=pd.DataFrame(dict(image_row=i.ravel(),image_column=j.ravel(),delta1_eta=d1.ravel(),deltaSigma_eta=ds.ravel(),response=full.real.ravel(),fit=fitted.real.ravel(),residual=residual.real.ravel(),response_imag=full.imag.ravel(),fit_imag=fitted.imag.ravel(),residual_imag=residual.imag.ravel()))
 for key,value in dict(L=LENGTH,eta=ETA,background_degree=BACKGROUND_DEGREE,n_even=N_EVEN,n_odd=N_ODD,fit_points=meta['n_rows'],fit_rank=meta['rank']).items():
  result[key]=value
 return result
def calculate_scan():
 model=projected_model()
 direct=model[3][1,len(model[0])]/model[3][len(model[0]),0]
 normalization=model[4][1,0]*model[3][len(model[0]),0]**2
 rows=[]
 for degree in(1,2,3):
  configuration=0
  for eta in(0.0025,0.005,0.01):
   for width in(3,6,9):
    coeff,meta=fit_laurent(model,eta,width,degree)
    estimate=coeff[0]/normalization
    error=abs(estimate-direct)
    rows.append(dict(L=LENGTH,eta=eta,width_eta=width,degree=degree,C_real=estimate.real,C_imag=estimate.imag,complex_audit_error=error,**meta,configuration_index=configuration,complex_error_recomputed=error,C_direct_for_recheck=direct))
    configuration+=1
 return pd.DataFrame(rows)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=OUTPUT,help='Destination CSV')
 args=parser.parse_args()
 result=calculate_grid()
 coordinates=['image_row','image_column','delta1_eta','deltaSigma_eta']
 metadata=['L','eta','background_degree','n_even','n_odd','fit_points','fit_rank']
 result=result[[*coordinates,'response_imag','fit_imag','residual_imag',*metadata]].rename(columns={'response_imag':'response','fit_imag':'fit','residual_imag':'residual'})
 result.to_csv(args.output,index=False)
if __name__=='__main__':
 main()
