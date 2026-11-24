import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
INPUT=HERE/'Fig_1_b_result.csv'
OUTPUT=HERE/'Fig_1_c_result.csv'
def main():
 with INPUT.open(newline='',encoding='utf-8')as file:
  reader=csv.DictReader(file)
  columns=reader.fieldnames
  rows=[row for row in reader if float(row['deltaSigma_eta'])==0.0]
 if not rows:
  raise ValueError('The response CSV has no zero total-detuning cut')
 rows.sort(key=lambda row:float(row['delta1_eta']))
 with OUTPUT.open('w',newline='',encoding='utf-8')as file:
  writer=csv.DictWriter(file,fieldnames=columns)
  writer.writeheader()
  writer.writerows(rows)
if __name__=='__main__':
 main()
