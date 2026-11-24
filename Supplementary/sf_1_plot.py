import csv
from pathlib import Path
import argparse
import subprocess
import sys
sys.dont_write_bytecode=True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator,FixedLocator,FuncFormatter,NullFormatter
import numpy as np
SCRIPT=Path(__file__).resolve().parent
DATA_DIR=SCRIPT
OUTPUT_PNG=SCRIPT.parent/'supplementary_figure_1.png'
PR_2COLUMN_WIDTH_IN=6.7
FIGURE_HEIGHT_IN=2*0.8*PR_2COLUMN_WIDTH_IN/3.236
COLORS=['#F20D0D','#0D2DFF','#15803D']
LINESTYLES=['-','--','-.']
MARKERS=['o','s','^']
MODEL_MARKERS=['s','^','o']
def read(data_dir,letter):
 with(Path(data_dir)/f'sf_1_{letter}.csv').open(newline='')as stream:
  return list(csv.DictReader(stream))
def xy(rows):
 return([float(r['x'])for r in rows],[float(r['y'])for r in rows])
def marker_positions(x):
 return np.unique(np.round(np.linspace(0,len(x)-1,min(8,len(x)))).astype(int))
def format_axis(axis,x_ticks,y_ticks):
 axis.set_xscale('linear')
 axis.set_yscale('linear')
 axis.xaxis.set_major_locator(FixedLocator(x_ticks))
 axis.yaxis.set_major_locator(FixedLocator(y_ticks))
 axis.set_xlim(x_ticks[0],x_ticks[-1])
 axis.set_ylim(y_ticks[0],y_ticks[-1])
 formatter=FuncFormatter(lambda value,position:f'${value:g}$')
 axis.xaxis.set_major_formatter(formatter)
 axis.yaxis.set_major_formatter(formatter)
 axis.xaxis.set_minor_formatter(NullFormatter())
 axis.yaxis.set_minor_formatter(NullFormatter())
 axis.xaxis.set_minor_locator(AutoMinorLocator(2))
 axis.yaxis.set_minor_locator(AutoMinorLocator(2))
 axis.tick_params(which='both',direction='in',top=True,right=True,labelsize=8,width=0.65,pad=2)
 axis.tick_params(which='major',length=2.5)
 axis.tick_params(which='minor',length=1.4)
def plot(data_dir,output_png):
 plt.rcParams.update({'text.usetex':True,'text.latex.preamble':'\\usepackage{amsmath}\\usepackage{bm}','font.family':'serif','font.serif':['Times'],'font.size':8,'axes.labelsize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':0.65,'lines.linewidth':1.5,'lines.markersize':3.5,'lines.markeredgewidth':1,'savefig.dpi':600})
 panels={letter:read(data_dir,letter)for letter in'abcd'}
 fig,axs=plt.subplots(2,2,figsize=(PR_2COLUMN_WIDTH_IN,FIGURE_HEIGHT_IN),layout='constrained')
 fig.get_layout_engine().set(w_pad=0.045,h_pad=0.025,wspace=0.035,hspace=0.04)
 for index,series in enumerate(['ising','detuned','tuned']):
  color=COLORS[index]
  direct=[r for r in panels['a']if r['series']==series and r['kind']=='direct']
  estimate=[r for r in panels['a']if r['series']==series and r['kind']=='estimate']
  imag=[r for r in panels['b']if r['series']==series]
  label=direct[0]['label'].replace('tuned ','')
  x,y=xy(direct)
  axs[0,0].plot(x,y,marker='o',linestyle=LINESTYLES[index],markevery=marker_positions(x),markerfacecolor='none',color=color,label=label,clip_on=False)
  axs[0,0].plot(*xy(estimate),marker='x',linestyle='none',color=color,clip_on=False)
  x,y=xy(imag)
  axs[0,1].plot(x,y,marker=MODEL_MARKERS[index],linestyle=LINESTYLES[index],markevery=marker_positions(x),color=color,label=label,clip_on=False)
 benchmark=next((r for r in panels['a']if r['kind']=='horizontal'))
 axs[0,0].axhline(float(benchmark['y']),color='#777777',ls=':')
 axs[0,0].set_xlabel('$\\boldsymbol{1/L^2}$',labelpad=1.5)
 axs[0,0].set_ylabel('$\\boldsymbol{C_L^{\\mathrm{dir}},\\ \\mathrm{Re}\\,\\widehat C}$',labelpad=1.5)
 axs[0,0].legend(loc='lower left',frameon=False,handlelength=1.5,handletextpad=0.4,labelspacing=0.25,borderaxespad=0.3)
 axs[0,1].set_xlabel('$\\boldsymbol{L}$',labelpad=1.5)
 axs[0,1].set_ylabel('$\\boldsymbol{\\mathrm{Im}\\,\\widehat C}$\\textbf{ (96 cycles)}',labelpad=1.5)
 axs[0,1].legend(loc='lower right',frameon=False,handlelength=1.5,handletextpad=0.4,labelspacing=0.25,borderaxespad=0.3)
 area_marker_sizes=[6.5,4.0,2.5]
 for index,series in enumerate(['0.02','0.01','0.005']):
  rows=[r for r in panels['c']if r['series']==series]
  x,y=xy(rows)
  axs[1,0].plot(x,y,marker=MARKERS[index],linestyle=LINESTYLES[index],markevery=marker_positions(x),color=COLORS[index],markersize=area_marker_sizes[index],zorder=3+index,markerfacecolor='none'if index<2 else COLORS[index],label=f'Area $={float(series):g}$',clip_on=False)
 axs[1,0].set_xlabel('\\textbf{Measurement cycles}',labelpad=1.5)
 axs[1,0].set_ylabel('$\\boldsymbol{\\mathrm{Im}\\,\\widehat C\\ (L=12)}$',labelpad=1.5)
 axs[1,0].legend(loc='upper right',frameon=False,handlelength=1.5,handletextpad=0.4,labelspacing=0.25,borderaxespad=0.3)
 for index,series in enumerate(['gap_ratio_8','PRG_L_gap_sigma']):
  rows=[r for r in panels['d']if r['series']==series]
  x,y=xy(rows)
  axs[1,1].plot(x,y,marker=MARKERS[index],linestyle=LINESTYLES[index],markevery=marker_positions(x),color=COLORS[index],label=rows[0]['label'],clip_on=False)
 band=next((r for r in panels['d']if r['kind']=='band'))
 reference=next((r for r in panels['d']if r['kind']=='horizontal'))
 axs[1,1].axhspan(float(band['y']),float(band['y_upper']),color='#999999',alpha=0.3,label=band['label'])
 axs[1,1].axhline(float(reference['y']),color='#777777',ls=':')
 axs[1,1].set_xlabel('$\\boldsymbol{\\ell^{-3}}$',labelpad=1.5)
 axs[1,1].set_ylabel('\\textbf{Pseudocritical field}',labelpad=1.5)
 axs[1,1].legend(loc='lower left',frameon=False,handlelength=1.5,handletextpad=0.4,labelspacing=0.25,borderaxespad=0.3)
 tick_values=[(np.linspace(0,0.03,4),np.linspace(0.491,0.501,6)),(np.linspace(6,26,6),np.linspace(0.00325,0.00375,6)),(np.linspace(0,400,5),np.linspace(0,0.01,5)),(np.linspace(0,0.005,5),np.linspace(1.073,1.087,5))]
 for ax,label in zip(axs.flat,'abcd'):
  ax.text(0.0,1.015,f'\\textbf{{({label})}}',transform=ax.transAxes,fontsize=11,ha='left',va='bottom')
 for ax,(x_ticks,y_ticks)in zip(axs.flat,tick_values):
  format_axis(ax,x_ticks,y_ticks)
 axs[1,0].set_ylim(top=0.0103)
 fig.add_artist(Rectangle((0.01,FIGURE_HEIGHT_IN/2),PR_2COLUMN_WIDTH_IN-0.02,0.01,transform=fig.dpi_scale_trans,facecolor='none',edgecolor='none'))
 fig.savefig(output_png,dpi=600,bbox_inches='tight',pad_inches=0.01)
 return fig
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=OUTPUT_PNG,help='Figure path; default is outside the supplementary data directory')
 args=parser.parse_args()
 for panel in'abcd':
  data=SCRIPT/f'sf_1_{panel}.csv'
  if not data.exists():
   subprocess.run([sys.executable,'-B',str(SCRIPT/f'sf_1_{panel}_calc.py')],check=True)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 plt.close(plot(DATA_DIR,args.output))
if __name__=='__main__':
 main()
