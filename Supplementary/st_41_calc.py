from pathlib import Path
import numpy as np
import pandas as pd
import tci_model_calc as tci
LENGTHS=range(6,27,2)
G=0.856
STATES=96
SHORT_CYCLES=96
LONG_CYCLES=384
RAMP=6
SHORT_AREA=0.005
LONG_AREA=0.0025
STEPS=256
MODELS={'p08':(4/5,),'p082':(4/5,2),'p1':(1,),'p12':(1,2),'p0812':(4/5,6/5)}
OUTPUT=Path(__file__).with_name('st_41.csv')
def calculate():
 samples=[]
 for length in LENGTHS:
  energy,spin,readout,static=tci.build_model(length,G,STATES)
  frequency=tci.frequency_residue(energy,spin,readout,static['retained_even'],static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'])
  pulses=[]
  for cycles,area in((SHORT_CYCLES,SHORT_AREA),(LONG_CYCLES,LONG_AREA)):
   pulses.append(tci.finite_pulse(energy,spin,readout,static['gap_sigma'],static['gap_epsilon'],static['M_sigma'],static['M_epsilon'],steps_per_cycle=STEPS,cycles=cycles,ramp_cycles=RAMP,area=area))
  samples.append((frequency['C_frequency_real'],pulses[0]['C_real'],pulses[1]['C_real'],static['C_direct']))
 L=np.asarray(list(LENGTHS),dtype=float)
 values=np.asarray(samples)
 rows=[]
 for name,powers in MODELS.items():
  design=np.column_stack([np.ones(len(L)),*(L**(-p)for p in powers)])
  intercepts=[tci.scaled_lstsq(design,values[:,column])[0]for column in range(4)]
  rows.append((name,*intercepts))
 return pd.DataFrame(rows,columns=['model','C_frequency','C_pulse_96','C_pulse_384','C_static'])
if __name__=='__main__':
 calculate().to_csv(OUTPUT,index=False)
