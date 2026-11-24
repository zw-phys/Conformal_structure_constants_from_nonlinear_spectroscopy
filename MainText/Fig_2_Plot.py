from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator,FixedLocator,FuncFormatter
HERE=Path(__file__).resolve().parent
OUTPUT=HERE.parent/'Fig2.png'
PR_2COLUMN_WIDTH_IN=6.7
FIGURE_HEIGHT_IN=0.8*PR_2COLUMN_WIDTH_IN/3.236
COLORS=['#F20D0D','#0D2DFF','#15803D','#5B6573','#929AA3']
LINESTYLES_2=['-','--','-.',':',(0,(6,2)),(0,(2,2)),(0,(6,2,2,2))]
LINESTYLES_3=['-','-.','--',':','--']
MARKERS=['o','s','^','D','v','P','X']
MARKER_L=[6,10,14,18,22,26]
SIZE_MODELS=('probe','probe_stress','empirical','empirical_stress','probe_descendant')
def latex_tick(value,position):
 return f'${value:g}$'
def draw_curve(axis,x,y,style,label,filled=False,zorder=3,linestyles=LINESTYLES_2,color_index=None):
 x=np.asarray(x)
 markevery=np.flatnonzero(np.isclose(x[:,None],1/np.asarray(MARKER_L,dtype=float),rtol=0.0,atol=1e-12).any(axis=1))
 color=COLORS[style if color_index is None else color_index]
 return axis.plot(x,y,color=color,linestyle=linestyles[style],marker=MARKERS[style],markevery=markevery,linewidth=1.5,markersize=3.5,markeredgewidth=1,markerfacecolor=color if filled else'none',label=label,zorder=zorder,clip_on=False)[0]
def main(data_directory=HERE,output=OUTPUT,pulse_data=None):
 amplitudes=pd.read_csv(data_directory/'Fig_2_a_result.csv',float_precision='round_trip')
 response=pd.read_csv(data_directory/'Fig_2_b_result.csv',float_precision='round_trip')
 if pulse_data is None:
  pulse_data=data_directory/'Fig_3_a_result.csv'
 panel_a=pd.read_csv(pulse_data,float_precision='round_trip')
 panel_b=pd.read_csv(data_directory/'Fig_3_b_result.csv',float_precision='round_trip')
 fit_parameters=response.iloc[0]
 fit=panel_b.iloc[0]
 plt.rcParams.update({'text.usetex':True,'font.family':'serif','font.serif':['Times'],'text.latex.preamble':'\\usepackage{amsmath}\\usepackage{bm}','font.size':8,'axes.labelsize':10,'axes.linewidth':0.65,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7,'legend.frameon':False,'hatch.linewidth':0.5})
 fig,axes=plt.subplots(1,4,figsize=(PR_2COLUMN_WIDTH_IN,FIGURE_HEIGHT_IN),layout='constrained')
 fig.get_layout_engine().set(w_pad=0.055,h_pad=0.025,wspace=0.055)
 legend_options=dict(fontsize=7,frameon=False,handlelength=0.85,handletextpad=0.25,labelspacing=0.15,borderaxespad=0.35,borderpad=0)
 sigma=draw_curve(axes[0],1/amplitudes.L,amplitudes.M_sigma_relative,0,'$M_\\sigma$',filled=True,zorder=4)
 epsilon=draw_curve(axes[0],1/amplitudes.L,amplitudes.M_epsilon_relative,1,'$M_\\epsilon$',filled=True,zorder=4)
 residue=draw_curve(axes[0],1/amplitudes.L,amplitudes.residue_relative,2,'$|R_{\\epsilon\\sigma\\sigma}|$',filled=True,zorder=4)
 exact=draw_curve(axes[0],1/amplitudes.L,amplitudes.exact_energy_leg,3,'Exact $M_\\epsilon$')
 cft=draw_curve(axes[0],1/amplitudes.L,amplitudes.sigma_cft,4,'CFT',zorder=1)
 draw_curve(axes[0],1/amplitudes.L,amplitudes.epsilon_cft,5,'_nolegend_',zorder=1,color_index=4)
 draw_curve(axes[0],1/amplitudes.L,amplitudes.residue_cft,6,'_nolegend_',zorder=1,color_index=4)
 axes[0].set_ylabel('\\textbf{Relative amplitude}',fontsize=10,labelpad=2)
 axes[0].set_ylim(0.1,1.035)
 axes[0].yaxis.set_major_locator(FixedLocator([0.1,0.4,0.7,1.0]))
 data_legend=axes[0].legend(handles=[sigma,epsilon,residue],loc='lower right',**legend_options)
 axes[0].add_artist(data_legend)
 axes[0].legend(handles=[exact,cft],loc='upper left',bbox_to_anchor=(0.02,0.76),**legend_options)
 extracted=draw_curve(axes[1],1/response.L,response.C_from_chi,0,'From $\\chi$',filled=True,zorder=4)
 chain_length=np.linspace(response.L.min(),response.L.max(),201)
 inverse_l2=chain_length**(-2)
 fitted=fit_parameters.C_inf_fit+fit_parameters.fit_L_minus_2*inverse_l2+fit_parameters.fit_L_minus_4*inverse_l2**2
 response_fit=draw_curve(axes[1],1/chain_length,fitted,3,'$C_\\infty+a/L^2+b/L^4$',color_index=1)
 axes[1].axhline(fit_parameters.C_cft,color=COLORS[4],linestyle='--',linewidth=1.5)
 static_range=axes[1].axhspan(fit_parameters.static_C_inf_lower,fit_parameters.static_C_inf_upper,facecolor=(21/255,128/255,61/255,0.18),edgecolor=COLORS[2],hatch='///',linewidth=0,label='Static $L\\to\\infty$ range',zorder=0)
 axes[1].set_ylabel('$\\boldsymbol{\\widehat C_{\\epsilon\\sigma\\sigma}}$',fontsize=10,labelpad=2)
 axes[1].set_ylim(0.49,0.505)
 axes[1].yaxis.set_major_locator(FixedLocator([0.49,0.495,0.5,0.505]))
 axes[1].legend(handles=[extracted,response_fit,static_range],loc='lower left',**legend_options)
 for series_index,(series,label,pulse_style,direct_style)in enumerate((('Ising','Ising',1,3),('Retuned J2=0.05','$J_2=0.05$',0,2))):
  points=panel_a[panel_a.series==series].sort_values('L')
  draw_curve(axes[2],1/points.L,points.C_real,pulse_style,label,filled=True,zorder=4,linestyles=LINESTYLES_3,color_index=2*series_index)
  draw_curve(axes[2],1/points.L,points.C_direct,direct_style,label+' direct',linestyles=LINESTYLES_3,color_index=2*series_index+1)
 axes[2].axhline(0.5,color=COLORS[4],linestyle='--',linewidth=1.5,label='CFT $1/2$',zorder=1)
 axes[2].set_ylim(0.49,0.505)
 axes[2].yaxis.set_major_locator(FixedLocator([0.49,0.495,0.5,0.505]))
 axes[2].set_ylabel('$\\boldsymbol{\\mathrm{Re}\\,\\widehat C_{\\epsilon\\sigma\\sigma}}$',fontsize=10,labelpad=2)
 axes[2].legend(loc='lower left',**legend_options)
 inverse_sizes=np.linspace(0.0,1/panel_b.L.min(),301)
 model_curves=[]
 for name in SIZE_MODELS:
  curve=fit[f'{name}_C_inf']+fit[f'{name}_a1']*inverse_sizes**fit[f'{name}_p1']
  if f'{name}_p2'in fit.index:
   curve+=fit[f'{name}_a2']*inverse_sizes**fit[f'{name}_p2']
  model_curves.append(curve)
 envelope=axes[3].fill_between(inverse_sizes,np.min(model_curves,axis=0),np.max(model_curves,axis=0),facecolor=(21/255,128/255,61/255,0.16),edgecolor=COLORS[2],hatch='///',linewidth=0,label='Five-model\nenvelope',zorder=0)
 pulse=draw_curve(axes[3],1/panel_b.L,panel_b.C_real,2,'Finite pulse',filled=True,zorder=4,linestyles=LINESTYLES_3,color_index=0)
 direct=draw_curve(axes[3],1/panel_b.L,panel_b.C_direct,3,'Direct ratio',linestyles=LINESTYLES_3,color_index=1)
 axes[3].vlines(0.0,fit.C_inf_lower,fit.C_inf_upper,color=COLORS[2],linewidth=1.5,zorder=5)
 axes[3].hlines([fit.C_inf_lower,fit.C_inf_upper],-0.0025,0.0025,color=COLORS[2],linewidth=1.5,zorder=5)
 axes[3].axhline(fit.C_cft,color=COLORS[4],linestyle='--',linewidth=1.5,zorder=1)
 axes[3].text(0.975,fit.C_cft+0.005,f'TCI CFT: ${fit.C_cft:.5f}$',transform=axes[3].get_yaxis_transform(),ha='right',va='bottom',fontsize=7)
 axes[3].set_xlim(-0.007,0.171)
 axes[3].xaxis.set_major_locator(FixedLocator([0.0,0.05,0.1,0.15]))
 axes[3].set_ylim(0.735,0.95)
 axes[3].yaxis.set_major_locator(FixedLocator([0.75,0.81,0.87,0.94]))
 axes[3].set_ylabel('$\\boldsymbol{\\widehat C_{\\epsilon\\sigma\\sigma}}$',fontsize=10,labelpad=2)
 axes[3].legend(handles=[pulse,direct,envelope],loc='upper right',bbox_to_anchor=(1.0,0.845),**legend_options)
 axes[3].text(0.02,0.035,'$L\\to\\infty$ fits'+'\n'+f'$[{fit.C_inf_lower:.5f},{fit.C_inf_upper:.5f}]$',transform=axes[3].transAxes,ha='left',va='bottom',fontsize=7)
 for axis in axes[:3]:
  axis.set_xlim(0.035,0.17)
  axis.xaxis.set_major_locator(FixedLocator([0.05,0.1,0.15]))
 for axis,letter in zip(axes,'abcd'):
  axis.set_xlabel('$\\boldsymbol{1/L}$',fontsize=10,labelpad=2)
  axis.text(0.0,1.015,f'\\textbf{{({letter})}}',transform=axis.transAxes,va='bottom',fontsize=11)
  axis.xaxis.set_major_formatter(FuncFormatter(latex_tick))
  axis.yaxis.set_major_formatter(FuncFormatter(latex_tick))
  axis.xaxis.set_minor_locator(AutoMinorLocator(2))
  y_ticks=axis.get_yticks()
  axis.yaxis.set_minor_locator(FixedLocator((y_ticks[:-1]+y_ticks[1:])/2))
  axis.tick_params(which='both',direction='in',top=True,right=True,labelsize=8,width=0.65,pad=2)
  axis.tick_params(which='major',length=2.5)
  axis.tick_params(which='minor',length=1.4)
 fig.add_artist(Rectangle((0.01,FIGURE_HEIGHT_IN/2),PR_2COLUMN_WIDTH_IN-0.02,0.01,transform=fig.dpi_scale_trans,facecolor='none',edgecolor='none'))
 fig.savefig(output,dpi=600,bbox_inches='tight',pad_inches=0.01)
 plt.close(fig)
if __name__=='__main__':
 main()
