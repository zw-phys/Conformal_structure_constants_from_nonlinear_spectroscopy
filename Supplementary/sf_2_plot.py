import csv
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
SCRIPT=Path(__file__).resolve().parent
DATA_DIR=SCRIPT
OUTPUT_PNG=SCRIPT.parent/'supplementary_figure_2.png'
PR_2COLUMN_WIDTH_IN=6.7
FIGURE_HEIGHT_IN=2*0.8*PR_2COLUMN_WIDTH_IN/3.236
COLORS=['#F20D0D','#0D2DFF','#15803D']
LINESTYLES=['-','--','-.']
MARKERS=['o','s','^']
def read(data_dir,letter):
 with(Path(data_dir)/f'sf_2_{letter}.csv').open(newline='')as stream:
  return list(csv.DictReader(stream))
def xy(rows,y_key='y'):
 return([float(r['x'])for r in rows],[float(r[y_key])for r in rows])
def draw_curve(axis,x,y,style=0,label=None,marker=None,markersize=3.5,**kwargs):
 markevery=np.unique(np.round(np.linspace(0,len(x)-1,min(8,len(x)))).astype(int))
 return axis.plot(x,y,color=COLORS[style],linestyle=LINESTYLES[style],marker=marker or MARKERS[style],markevery=markevery,linewidth=1.5,markersize=markersize,markeredgewidth=1,label=label,clip_on=False,**kwargs)
def format_axis(axis,x_ticks,y_ticks,y_scale=1):
 axis.set_xscale('linear')
 axis.set_yscale('linear')
 axis.xaxis.set_major_locator(FixedLocator(x_ticks))
 axis.yaxis.set_major_locator(FixedLocator(y_ticks))
 axis.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'${value:g}$'))
 axis.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'${value / y_scale:g}$'))
 for direction in[axis.xaxis,axis.yaxis]:
  direction.set_minor_locator(AutoMinorLocator(2))
  direction.set_minor_formatter(NullFormatter())
 x_pad=0.03*(x_ticks[-1]-x_ticks[0])
 y_pad=0.04*(y_ticks[-1]-y_ticks[0])
 axis.set_xlim(x_ticks[0]-x_pad,x_ticks[-1]+x_pad)
 axis.set_ylim(y_ticks[0]-y_pad,y_ticks[-1]+y_pad)
 axis.tick_params(which='both',direction='in',top=True,right=True,labelsize=8)
 axis.tick_params(which='major',length=2.5,width=0.65,pad=2)
 axis.tick_params(which='minor',length=1.4,width=0.65)
 axis.xaxis.labelpad=2
 axis.yaxis.labelpad=2
 if y_scale!=1:
  exponent=int(round(np.log10(y_scale)))
  axis.text(1,1.015,f'$\\times10^{{{exponent}}}$',transform=axis.transAxes,fontsize=8,ha='right',va='bottom')
def plot(data_dir,output_png):
 plt.rcParams.update({'text.usetex':True,'text.latex.preamble':'\\usepackage{amsmath}\\usepackage{bm}','font.family':'serif','font.serif':['Times'],'font.size':8,'axes.labelsize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':0.65,'lines.linewidth':1.5,'lines.markersize':3.5,'lines.markeredgewidth':1})
 panels={letter:read(data_dir,letter)for letter in'abcde'}
 fig=plt.figure(figsize=(PR_2COLUMN_WIDTH_IN,FIGURE_HEIGHT_IN),layout='constrained')
 gs=fig.add_gridspec(2,6)
 axes=[fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[0,2:4]),fig.add_subplot(gs[0,4:6]),fig.add_subplot(gs[1,:3]),fig.add_subplot(gs[1,3:])]
 fig.get_layout_engine().set(w_pad=0.035,h_pad=0.025,wspace=0.045,hspace=0.035)
 for style,series in enumerate(['intermediate','output']):
  real=[r for r in panels['a']if r['series']==series]
  imag=[r for r in panels['b']if r['series']==series]
  draw_curve(axes[0],*xy(real),style=style,label=real[0]['label'])
  draw_curve(axes[1],*xy(imag),style=style,label=imag[0]['label'])
 axes[0].set(xlabel='\\textbf{Detuning / }$\\boldsymbol{(2\\pi/T_m)}$',ylabel='$\\boldsymbol{\\Delta\\mathrm{Re}\\widehat C}$')
 axes[0].legend(frameon=False,loc='lower left',handlelength=1.2,handletextpad=0.35,borderaxespad=0.25,labelspacing=0.25)
 axes[1].set(xlabel='\\textbf{Detuning / }$\\boldsymbol{(2\\pi/T_m)}$',ylabel='$\\boldsymbol{\\mathrm{Im}\\widehat C}$')
 draw_curve(axes[2],*xy(panels['c']))
 axes[2].set(xlabel='$\\boldsymbol{\\gamma/(2\\pi/T_m)}$',ylabel='\\textbf{Unit kernel /}'+'\n'+'\\textbf{rectangular}')
 draw_curve(axes[3],*xy(panels['d'],'quadratic_reference'),style=1,label='Quadratic reference',markerfacecolor='none',markersize=5,zorder=2)
 draw_curve(axes[3],*xy(panels['d']),label='Signal',zorder=3)
 axes[3].set(xlabel='\\textbf{Nominal area}',ylabel='$\\boldsymbol{|S_4|/4}$')
 handles,labels=axes[3].get_legend_handles_labels()
 axes[3].legend(handles[::-1],labels[::-1],frameon=False,loc='upper left',handlelength=1,handletextpad=0.3,borderaxespad=0.25,labelspacing=0.25,bbox_to_anchor=(0,0.9))
 draw_curve(axes[4],*xy(panels['e']))
 axes[4].set(xlabel='$\\boldsymbol{\\phi_1+\\phi_2}$',ylabel='$\\boldsymbol{\\arg S_4}$'+'\n'+'\\textbf{(unwrapped)}')
 settings=[([-1.1,-0.55,0,0.55,1.1],[-0.02,-0.01,0,0.01,0.02],0.01),([-1.1,-0.55,0,0.55,1.1],np.linspace(0.0044,0.0096,5),0.001),([0,0.25,0.5,0.75,1],[0.2,0.4,0.6,0.8,1],1),([0.005,0.01,0.015,0.02,0.025,0.03],np.linspace(0,6.5e-06,6),1e-06),([0,0.8,1.6,2.4,3.2],[-3.2,-2.1,-1,0.1],1)]
 for axis,label,(x_ticks,y_ticks,y_scale)in zip(axes,'abcde',settings):
  format_axis(axis,x_ticks,y_ticks,y_scale)
  axis.text(0,1.015,f'\\textbf{{({label})}}',transform=axis.transAxes,fontsize=11,ha='left',va='bottom')
 fig.savefig(output_png,dpi=600,bbox_inches='tight',pad_inches=0.01)
 return fig
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=OUTPUT_PNG,help='Figure path; default is outside the supplementary data directory')
 args=parser.parse_args()
 for panel in'abcde':
  data=SCRIPT/f'sf_2_{panel}.csv'
  if not data.exists():
   subprocess.run([sys.executable,'-B',str(SCRIPT/f'sf_2_{panel}_calc.py')],check=True)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 plt.close(plot(DATA_DIR,args.output))
if __name__=='__main__':
 main()
