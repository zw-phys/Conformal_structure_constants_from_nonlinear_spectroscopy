from __future__ import annotations
from pathlib import Path
import argparse
import subprocess
import sys
sys.dont_write_bytecode=True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator,FixedLocator,FuncFormatter,NullFormatter
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
DATA_DIR=HERE
OUTPUT_PNG=HERE.parent/'supplementary_figure_3.png'
PR_COLUMN_WIDTH_IN=3.375
COLORS=['#F20D0D','#15803D','#0D2DFF']
LINESTYLES=['-.','-','--']
MARKERS=['s','o','^']
def marker_positions(x):
 return np.unique(np.round(np.linspace(0,len(x)-1,min(8,len(x)))).astype(int))
def render(data_dir,output_png):
 points=pd.read_csv(data_dir/'sf_3_a.csv',float_precision='round_trip')
 plt.rcParams.update({'text.usetex':True,'text.latex.preamble':'\\usepackage{amsmath}\\usepackage{bm}','font.family':'serif','font.serif':['Times'],'font.size':8,'axes.labelsize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':0.65,'lines.linewidth':1.5,'lines.markersize':3.5,'lines.markeredgewidth':1,'savefig.dpi':600})
 fig,axis=plt.subplots(figsize=(PR_COLUMN_WIDTH_IN,PR_COLUMN_WIDTH_IN/1.618),layout='constrained')
 fig.get_layout_engine().set(w_pad=0.03,h_pad=0.025)
 for index,series in enumerate(['Ising','Retuned $J_2=0.05$','TCI']):
  values=points[points.series==series]
  axis.plot(values.L,values.C_imag,color=COLORS[index],linestyle=LINESTYLES[index],marker=MARKERS[index],markevery=marker_positions(values.L),label=series,clip_on=False)
 axis.set_xlabel('$\\boldsymbol{L}$',labelpad=2)
 axis.set_ylabel('$\\boldsymbol{\\mathrm{Im}\\,\\widehat C}$',labelpad=2)
 axis.set_xscale('linear')
 axis.set_yscale('linear')
 x_ticks=np.arange(6,27,4)
 y_ticks=np.linspace(0.00324,0.00444,5)
 axis.xaxis.set_major_locator(FixedLocator(x_ticks))
 axis.yaxis.set_major_locator(FixedLocator(y_ticks))
 axis.set_xlim(x_ticks[0]-0.4,x_ticks[-1]+0.4)
 axis.set_ylim(y_ticks[0],y_ticks[-1])
 formatter=FuncFormatter(lambda value,position:f'${value:g}$')
 for coordinate in[axis.xaxis,axis.yaxis]:
  coordinate.set_major_formatter(formatter)
  coordinate.set_minor_formatter(NullFormatter())
  coordinate.set_minor_locator(AutoMinorLocator(2))
 axis.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'${value * 1000.0:g}$'))
 axis.text(0,1.015,'$\\times 10^{-3}$',transform=axis.transAxes,fontsize=8,va='bottom')
 axis.legend(frameon=False,loc='center right',bbox_to_anchor=(0.98,0.5),handlelength=1.5,handletextpad=0.5,labelspacing=0.2,borderaxespad=0.2)
 axis.tick_params(which='both',direction='in',top=True,right=True,width=0.65)
 axis.tick_params(which='major',length=2.5,pad=2)
 axis.tick_params(which='minor',length=1.4)
 fig.savefig(output_png,dpi=600,bbox_inches='tight',pad_inches=0.01)
 plt.close(fig)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=OUTPUT_PNG,help='Figure path; default is outside the supplementary data directory')
 args=parser.parse_args()
 for panel in'a':
  data=HERE/f'sf_3_{panel}.csv'
  if not data.exists():
   subprocess.run([sys.executable,'-B',str(HERE/f'sf_3_{panel}_calc.py')],check=True)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 render(DATA_DIR,args.output)
if __name__=='__main__':
 main()
