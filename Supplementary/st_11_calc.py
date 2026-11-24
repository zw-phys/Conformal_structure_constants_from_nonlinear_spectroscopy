from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pulse_calc import pulse_scan
OUTPUT=Path(__file__).resolve().with_name('st_11.csv')
J2=0.05
FIELD=1.08385
SCAN_ORDER=['q0_time_step','q0_quadrature']
SCAN_PARAMETERS={'J2':0.05,'h':1.08385,'n':64,'cycles':96.0,'ramp':8.0,'area':0.02,'phi1':0.0,'phi2':0.0,'envelope_scale':1.0,'phase_offset':0.0,'intermediate_detuning_units':0.0,'output_detuning_units':0.0,'gamma_units':0.0}
SCAN_POINTS=[{'tag':'q0_time_step','L':12,'steps':64,'quadrature_stride':1},{'tag':'q0_time_step','L':12,'steps':128,'quadrature_stride':1},{'tag':'q0_time_step','L':12,'steps':256,'quadrature_stride':1},{'tag':'q0_quadrature','L':12,'steps':256,'quadrature_stride':1},{'tag':'q0_quadrature','L':12,'steps':256,'quadrature_stride':2},{'tag':'q0_quadrature','L':12,'steps':256,'quadrature_stride':4},{'tag':'q0_quadrature','L':12,'steps':256,'quadrature_stride':8},{'tag':'q0_time_step','L':18,'steps':64,'quadrature_stride':1},{'tag':'q0_time_step','L':18,'steps':128,'quadrature_stride':1},{'tag':'q0_time_step','L':18,'steps':256,'quadrature_stride':1},{'tag':'q0_quadrature','L':18,'steps':256,'quadrature_stride':1},{'tag':'q0_quadrature','L':18,'steps':256,'quadrature_stride':2},{'tag':'q0_quadrature','L':18,'steps':256,'quadrature_stride':4},{'tag':'q0_quadrature','L':18,'steps':256,'quadrature_stride':8}]
def calculate():
 pulse=pulse_scan([dict(SCAN_PARAMETERS,**point)for point in SCAN_POINTS])
 pulse=pulse.loc[(pulse.J2==J2)&(pulse.h==FIELD)&pulse.tag.isin(SCAN_ORDER)].copy()
 pulse['tag']=pd.Categorical(pulse.tag,categories=SCAN_ORDER,ordered=True)
 pulse=pulse.sort_values(['L','tag','steps','quadrature_stride'])
 S_pc=pulse.numerator_real.to_numpy()+1j*pulse.numerator_imag.to_numpy()
 S_1=pulse.denominator_real.to_numpy()+1j*pulse.denominator_imag.to_numpy()
 C=S_pc/S_1
 pulse['C_real']=C.real
 pulse['C_imag']=C.imag
 result=pulse[['L','steps','quadrature_stride','C_real','C_imag','tag']]
 return result
def main():
 calculate().to_csv(OUTPUT,index=False)
if __name__=='__main__':
 main()
