from pathlib import Path
import argparse
import subprocess
import sys
sys.dont_write_bytecode=True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import AutoMinorLocator,FuncFormatter,LogFormatterMathtext
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
LINEAR=HERE/'sf_5_a.csv'
PULSE=HERE/'sf_5_b.csv'
OUTPUT=HERE.parent/'supplementary_figure_5.png'
PR_2COLUMN_WIDTH_IN=6.7
FIGURE_HEIGHT_IN=0.8*PR_2COLUMN_WIDTH_IN/3.236
CALIBRATION_LENGTH=18
COLORS={'bare':'#F20D0D','improved':'#0D2DFF'}
plt.rcParams.update({'text.usetex':True,'text.latex.preamble':'\\usepackage{amsmath}\\usepackage{bm}','font.family':'serif','font.serif':['Times'],'font.size':8,'axes.labelsize':10,'axes.labelpad':2,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'legend.frameon':False,'axes.linewidth':0.65})
def plot(output_png):
 linear=pd.read_csv(LINEAR)
 pulse=pd.read_csv(PULSE)
 fig,(ax_linear,ax_pulse)=plt.subplots(1,2,figsize=(PR_2COLUMN_WIDTH_IN,FIGURE_HEIGHT_IN),layout='constrained')
 fig.get_layout_engine().set(w_pad=0.03,h_pad=0.025,wspace=0.045)
 calibrated_ratio=linear.fixed_secondary_ratio.mask(linear.L==CALIBRATION_LENGTH)
 marks=np.unique(np.round(np.linspace(0,len(linear.L)-1,min(8,len(linear.L)))).astype(int))
 finite=np.flatnonzero(calibrated_ratio.notna())
 calibrated_marks=finite[np.searchsorted(finite,marks)]
 ax_linear.plot(linear.L,linear.bare_secondary_ratio,'o-',color=COLORS['bare'],markevery=marks,linewidth=1.5,markersize=3.5,markeredgewidth=1,label='Original probe')
 ax_linear.plot(linear.L,calibrated_ratio,'s--',color=COLORS['improved'],markevery=calibrated_marks,linewidth=1.5,markersize=3.5,markeredgewidth=1,label='Calibrated probe')
 ax_linear.axvline(CALIBRATION_LENGTH,color='#929AA3',linewidth=1.5,linestyle=':',zorder=1)
 ax_linear.set_yscale('log')
 ax_linear.yaxis.set_major_formatter(LogFormatterMathtext())
 ax_linear.set_ylim(0.0003,0.24)
 ax_linear.set_ylabel("$\\boldsymbol{|M_{p,\\sigma'}/M_{p,\\sigma}|}$")
 legend_options=dict(frameon=False,fontsize=8,handlelength=1.3,handletextpad=0.35,labelspacing=0.2,borderaxespad=0.35)
 ax_linear.legend(loc='center right',bbox_to_anchor=(0.99,0.56),**legend_options)
 pulse_handles=[]
 for probe,label,marker in(('bare','Original','o'),('improved','Calibrated','s')):
  color=COLORS[probe]
  direct,=ax_pulse.plot(pulse.L,pulse[f'C_direct_{probe}'],linestyle='none',marker=marker,markersize=5,markerfacecolor='none',markeredgewidth=1,color=color)
  center,=ax_pulse.plot(pulse.L,pulse[f'pulse_real_{probe}'],linestyle='none',marker=marker,markersize=2.2,markeredgewidth=0,color=color,label=label)
  pulse_handles.append((direct,center))
 cft=ax_pulse.axhline(pulse.C_TCI.iloc[0],color='#5B6573',linewidth=1.5,linestyle=':',label='TCI CFT',zorder=1)
 ax_pulse.set_ylim(0.735,0.93)
 ax_pulse.yaxis.set_minor_locator(AutoMinorLocator(2))
 ax_pulse.set_ylabel('$\\boldsymbol{C_L^{\\mathrm{dir}},\\ \\mathrm{Re}\\,\\widehat C}$')
 ax_pulse.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'${value:g}$'))
 ax_pulse.legend([*pulse_handles,cft],[pair[1].get_label()for pair in pulse_handles]+[cft.get_label()],handler_map={tuple:HandlerTuple(ndivide=1)},loc='lower right',**legend_options)
 for axis,panel,last_length in((ax_linear,'a',28),(ax_pulse,'b',26)):
  axis.set_xlim(5.5,last_length+0.5)
  axis.set_xticks(np.arange(6,last_length+1,4))
  axis.xaxis.set_minor_locator(AutoMinorLocator(2))
  axis.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'${value:g}$'))
  axis.tick_params(which='both',direction='in',top=True,right=True,width=0.65,pad=2)
  axis.tick_params(which='major',length=2.5)
  axis.tick_params(which='minor',length=1.4)
  axis.set_xlabel('$\\boldsymbol{L}$')
  axis.text(0.0,1.015,f'\\textbf{{({panel})}}',transform=axis.transAxes,va='bottom',fontsize=11)
 fig.savefig(output_png,dpi=600,bbox_inches='tight',pad_inches=0.01)
 plt.close(fig)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=OUTPUT,help='Figure path; default is outside the supplementary data directory')
 args=parser.parse_args()
 for panel in'ab':
  data=HERE/f'sf_5_{panel}.csv'
  if not data.exists():
   subprocess.run([sys.executable,'-B',str(HERE/f'sf_5_{panel}_calc.py')],check=True)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 plot(args.output)
if __name__=='__main__':
 main()
