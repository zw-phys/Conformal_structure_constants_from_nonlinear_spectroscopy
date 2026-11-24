from pathlib import Path
import numpy as np
import pandas as pd
import tci_model_calc as tci
LENGTHS=range(6,27,2)
G=0.856
STATES=96
CYCLES=384
RAMP=6
AREA=0.0025
STEPS=256
CALIBRATION_LENGTH=18
PROBES=('bare','improved')
OUTPUT=Path(__file__).with_name('sf_5_b.csv')
def calculate():
 data=tci.paired_series(LENGTHS,G,STATES,CYCLES,RAMP,AREA,STEPS,CALIBRATION_LENGTH)
 data['pulse_real']=data.full_real
 quantities=['C_direct','pulse_real']
 result=data.pivot(index='L',columns='probe',values=quantities)
 result=result.reindex(columns=pd.MultiIndex.from_product([quantities,PROBES]))
 result.columns=[f'{quantity}_{probe}'for quantity in quantities for probe in PROBES]
 result=result.reset_index()
 result['C_TCI']=tci.C_CFT
 return result
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
