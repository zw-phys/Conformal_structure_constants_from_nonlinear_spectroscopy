from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator,FixedLocator,FuncFormatter
HERE=Path(__file__).resolve().parent
OUTPUT=HERE.parent/'Fig1.png'
PR_2COLUMN_WIDTH_IN=6.7
FIGURE_HEIGHT_IN=0.8*PR_2COLUMN_WIDTH_IN/3.236
COLORS=['#F20D0D','#0D2DFF','#15803D','#7C3AED','#5B6573']
LINESTYLES=['-','--','-.',':',(0,(6,2,2,2))]
MARKERS=['o','s','^','D','v']
DRIVE_DETUNING='(\\omega_1-\\Delta_\\sigma)/\\eta'
SUM_DETUNING='(\\omega_1+\\omega_2-\\Delta_\\epsilon)/\\eta'
NORMALIZED_RESPONSE='(\\mathrm{i}\\eta)^2\\chi/(M_\\epsilon M_\\sigma^2)'
def latex_tick(value,position):
 return f'${value:g}$'
def quarter_tick(value,position):
 fractions={-0.25:'$-1/4$',0.0:'$0$',0.25:'$1/4$',0.5:'$1/2$'}
 return fractions.get(value,f'${value:g}$')
def draw_curve(axis,x,y,style,label,filled=False):
 x=np.asarray(x)
 markevery=np.unique(np.round(np.linspace(0,len(x)-1,min(8,len(x)))).astype(int))
 color=COLORS[style]
 axis.plot(x,y,color=color,linestyle=LINESTYLES[style],marker=MARKERS[style],markevery=markevery,linewidth=1.5,markersize=3.5,markeredgewidth=1,markerfacecolor=color if filled else'none',label=label)
def format_axis(axis):
 axis.xaxis.set_major_formatter(FuncFormatter(latex_tick))
 axis.yaxis.set_major_formatter(FuncFormatter(latex_tick))
 axis.xaxis.set_minor_locator(AutoMinorLocator(2))
 axis.yaxis.set_minor_locator(AutoMinorLocator(2))
 axis.tick_params(which='both',direction='in',top=True,right=True,labelsize=8,width=0.65,pad=2)
 axis.tick_params(which='major',length=2.5)
 axis.tick_params(which='minor',length=1.4)
def main():
 gaps=pd.read_csv(HERE/'Fig_1_a_result.csv',float_precision='round_trip')
 response=pd.read_csv(HERE/'Fig_1_b_result.csv',float_precision='round_trip')
 cut=pd.read_csv(HERE/'Fig_1_c_result.csv',float_precision='round_trip')
 magnitude=response.pivot(index='deltaSigma_eta',columns='delta1_eta',values='response_abs')
 plt.rcParams.update({'text.usetex':True,'font.family':'serif','font.serif':['Times'],'text.latex.preamble':'\\usepackage{amsmath}\\usepackage{bm}','font.size':8,'axes.labelsize':10,'axes.labelpad':2,'axes.linewidth':0.65,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'legend.frameon':False})
 fig,axes=plt.subplots(1,3,figsize=(PR_2COLUMN_WIDTH_IN,FIGURE_HEIGHT_IN),gridspec_kw={'width_ratios':[1,1.15,1.22]},layout='constrained')
 fig.get_layout_engine().set(w_pad=0.03,h_pad=0.025,wspace=0.045)
 draw_curve(axes[0],1/gaps.L,gaps.x_sigma_over_cft,0,'$j=\\sigma$',filled=True)
 draw_curve(axes[0],1/gaps.L,gaps.x_epsilon_over_cft,1,'$j=\\epsilon$',filled=True)
 draw_curve(axes[0],1/gaps.L,gaps.cft_limit,4,'CFT limit')
 gap_marker_indices=np.flatnonzero(np.isin(gaps.L.to_numpy(),[6,10,14,18,22,26]))
 for line in axes[0].lines:
  line.set_markevery(gap_marker_indices)
 axes[0].set_xlabel('$\\boldsymbol{1/L}$',fontsize=10)
 axes[0].set_ylabel('$\\boldsymbol{\\Delta_j/\\Delta_j^{\\mathrm{CFT}}}$',fontsize=10)
 axes[0].xaxis.set_major_locator(FixedLocator([0.05,0.1,0.15]))
 axes[0].yaxis.set_major_locator(FixedLocator([0.99,0.995,1.0,1.005]))
 axes[0].margins(x=0.025,y=0.04)
 axes[0].set_xlim(0.035,0.17)
 axes[0].legend(loc='lower left',fontsize=8,frameon=False,handlelength=1.8,handletextpad=0.4,labelspacing=0.2,borderaxespad=0.4)
 heatmap=axes[1].pcolormesh(magnitude.columns,magnitude.index,magnitude.to_numpy(),cmap='viridis',shading='auto')
 colorbar=fig.colorbar(heatmap,ax=axes[1],fraction=0.07,pad=0.04,aspect=20,ticks=[0.0,0.2,0.4],format=FuncFormatter(latex_tick))
 colorbar.set_label(f'$\\boldsymbol{{\\left|{NORMALIZED_RESPONSE}\\right|}}$',fontsize=10,labelpad=2)
 colorbar.ax.yaxis.set_minor_locator(AutoMinorLocator(2))
 colorbar.ax.tick_params(which='both',direction='in',labelsize=8,width=0.65)
 axes[1].axhline(0.0,color='white',linestyle='--',linewidth=1.5)
 axes[1].axvline(0.0,color='white',linestyle='--',linewidth=1.5)
 axes[1].set_xlabel(f'$\\boldsymbol{{{DRIVE_DETUNING}}}$',fontsize=10)
 axes[1].set_ylabel(f'$\\boldsymbol{{{SUM_DETUNING}}}$',fontsize=10)
 axes[1].xaxis.set_major_locator(FixedLocator([-6,-3,0,3,6]))
 axes[1].yaxis.set_major_locator(FixedLocator([-6,-3,0,3,6]))
 for component,label,styles in[('real','Re',(0,2)),('imag','Im',(1,3))]:
  for quantity,description,style in zip(('response','fit'),('response','Laurent fit'),styles):
   draw_curve(axes[2],cut.delta1_eta,cut[f'{quantity}_{component}'],style,f'$\\mathrm{{{label}}}$ {description}')
 marker_indices=np.flatnonzero(np.isclose(cut.delta1_eta.to_numpy()[:,None],[-6.0,-3.0,0.0,3.0,6.0],rtol=0.0,atol=1e-12).any(axis=1))
 for line in axes[2].lines:
  line.set_markevery(marker_indices)
 axes[2].set_xlabel(f'$\\boldsymbol{{{DRIVE_DETUNING}}}$',fontsize=10)
 axes[2].set_ylabel(f'$\\boldsymbol{{{NORMALIZED_RESPONSE}}}$',fontsize=10,labelpad=-4)
 axes[2].yaxis.set_label_position('right')
 axes[2].xaxis.set_major_locator(FixedLocator([-6,-3,0,3,6]))
 axes[2].yaxis.set_major_locator(FixedLocator([-0.25,0.0,0.25,0.5]))
 axes[2].margins(x=0.025,y=0.04)
 axes[2].legend(loc='lower right',fontsize=6,frameon=False,handlelength=1,handletextpad=0.25,labelspacing=0.2,borderaxespad=0.15,borderpad=0.1)
 for axis,letter in zip(axes,'abc'):
  format_axis(axis)
  axis.text(0.0,1.015,f'\\textbf{{({letter})}}',transform=axis.transAxes,va='bottom',fontsize=11)
 axes[2].yaxis.set_major_formatter(FuncFormatter(quarter_tick))
 axes[2].tick_params(axis='y',labelleft=False,labelright=True)
 fig.savefig(OUTPUT,dpi=600,bbox_inches='tight',pad_inches=0.01)
 plt.close(fig)
if __name__=='__main__':
 main()
