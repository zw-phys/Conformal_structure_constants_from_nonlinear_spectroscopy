from pathlib import Path
import argparse
import subprocess
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator,FixedLocator,FuncFormatter,NullFormatter
HERE=Path(__file__).resolve().parent
DATA_DIR=HERE
OUTPUT_PNG=HERE.parent/'supplementary_figure_4.png'
PR_2COLUMN_WIDTH_IN=6.7
FIGURE_HEIGHT_IN=0.8*PR_2COLUMN_WIDTH_IN/3.236
def format_axis(axis,x_ticks,y_ticks,y_unit=1.0):
 axis.set_xlim(x_ticks[0],x_ticks[-1])
 axis.set_ylim(y_ticks[0],y_ticks[-1])
 axis.xaxis.set_major_locator(FixedLocator(x_ticks))
 axis.yaxis.set_major_locator(FixedLocator(y_ticks))
 axis.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'${value:g}$'))
 axis.yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'${value / y_unit:g}$'))
 axis.xaxis.set_minor_locator(AutoMinorLocator(2))
 axis.yaxis.set_minor_locator(AutoMinorLocator(2))
 axis.xaxis.set_minor_formatter(NullFormatter())
 axis.yaxis.set_minor_formatter(NullFormatter())
 axis.tick_params(which='both',direction='in',top=True,right=True,labelsize=8,width=0.65)
 axis.tick_params(which='major',length=2.5,pad=1.5)
 axis.tick_params(which='minor',length=1.4)
 axis.xaxis.labelpad=1.5
 axis.yaxis.labelpad=1.5
def plot(data_dir:Path,output_png:Path)->None:
 for letter in'ab':
  if not(data_dir/f'sf_4_{letter}.csv').exists():
   subprocess.run([sys.executable,str(HERE/f'sf_4_{letter}_calc.py'),'--output',str(data_dir/f'sf_4_{letter}.csv')],check=True)
 real=pd.read_csv(data_dir/'sf_4_a.csv')
 imaginary=pd.read_csv(data_dir/'sf_4_b.csv')
 extent=(real.deltaSigma_eta.min(),real.deltaSigma_eta.max(),real.delta1_eta.min(),real.delta1_eta.max())
 plt.rcParams.update({'text.usetex':True,'font.family':'serif','font.serif':['Times'],'text.latex.preamble':'\\usepackage{amsmath}\\usepackage{bm}','font.size':8,'axes.labelsize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':0.65,'lines.linewidth':1.5,'lines.markersize':3.5,'lines.markeredgewidth':1,'savefig.dpi':600})
 fig,ax=plt.subplots(1,2,figsize=(PR_2COLUMN_WIDTH_IN,FIGURE_HEIGHT_IN),layout='constrained')
 fig.get_layout_engine().set(w_pad=0.035,h_pad=0.025,wspace=0.035)
 for i,(table,component)in enumerate([(real,'real'),(imaginary,'imag')]):
  image=table.pivot(index='delta1_eta',columns='deltaSigma_eta',values='residual')
  residual=image.to_numpy()
  displayed=np.where(np.abs(residual)<0.0001,0,residual)
  lower=np.floor(residual.min()/0.0001)*0.0001
  upper=np.ceil(residual.max()/0.0001)*0.0001
  im=ax[i].pcolormesh(image.columns.to_numpy(),image.index.to_numpy(),displayed,cmap='RdBu_r',shading='nearest',edgecolors='none',norm=TwoSlopeNorm(vmin=lower,vcenter=0,vmax=upper))
  colorbar=fig.colorbar(im,ax=ax[i],fraction=0.045,pad=0.025,aspect=30,ticks=[lower,0,upper],format=FuncFormatter(lambda value,position:f'${value / 0.0001:g}$'))
  colorbar.ax.set_xlabel('$\\boldsymbol{10^{-4}}$',fontsize=10,labelpad=3)
  colorbar.ax.xaxis.set_label_position('top')
  colorbar.ax.tick_params(which='both',direction='in',labelsize=8,length=2,width=0.65,pad=1.5)
  colorbar.ax.yaxis.set_minor_locator(AutoMinorLocator(2))
  colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
  ax[i].set_xlabel('$\\boldsymbol{\\delta_\\Sigma/\\eta}$')
  ax[i].set_ylabel('$\\boldsymbol{\\delta_1/\\eta}$')
  format_axis(ax[i],np.linspace(extent[0],extent[1],3),np.linspace(extent[2],extent[3],3))
 for axis,letter in zip(ax,'ab'):
  axis.text(0.0,1.015,f'\\textbf{{({letter})}}',transform=axis.transAxes,fontsize=11,va='bottom')
 fig.savefig(output_png,bbox_inches='tight',pad_inches=0.01,dpi=600)
 plt.close(fig)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=OUTPUT_PNG)
 parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
 args=parser.parse_args()
 plot(args.data_dir,args.output)
