import csv
import math
from pathlib import Path
HERE=Path(__file__).resolve().parent
INPUT=HERE/'Fig_2_b_result.csv'
OUTPUT=HERE/'Fig_2_a_result.csv'
REFERENCE_LENGTH=6
def relative_amplitudes(amplitudes,reference_length):
 sigma_0,epsilon_0,residue_0=amplitudes[reference_length]
 reference_cosine=math.cos(math.pi/(2.0*reference_length))
 rows=[]
 for length,(sigma,epsilon,residue)in sorted(amplitudes.items()):
  relative_length=length/reference_length
  exact_energy_leg=math.cos(math.pi/(2.0*length))/reference_cosine/relative_length
  rows.append((length,sigma/sigma_0,epsilon/epsilon_0,residue/residue_0,relative_length**(-1.0/8.0),1.0/relative_length,relative_length**(-5.0/4.0),exact_energy_leg))
 return rows
def main():
 amplitudes={}
 with INPUT.open(encoding='utf-8',newline='')as file:
  for row in csv.DictReader(file):
   length=int(row['L'])
   residue=math.hypot(float(row['R_real']),float(row['R_imag']))
   amplitudes[length]=(float(row['M_sigma']),float(row['M_epsilon']),residue)
 rows=relative_amplitudes(amplitudes,REFERENCE_LENGTH)
 with OUTPUT.open('w',encoding='utf-8',newline='')as file:
  writer=csv.writer(file)
  writer.writerow(('L','M_sigma_relative','M_epsilon_relative','residue_relative','sigma_cft','epsilon_cft','residue_cft','exact_energy_leg'))
  writer.writerows(rows)
if __name__=='__main__':
 main()
