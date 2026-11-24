from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch,PathPatch
from matplotlib.path import Path as MplPath
PR_COLUMN_WIDTH_IN=3.375
FIGURE_HEIGHT_IN=1.82
colors=['#F20D0D','#0D2DFF']
ink='#272439'
readout_color='#423951'
output_path=Path(__file__).resolve().parent.parent/'Fig_0.png'
ring_center=np.array([PR_COLUMN_WIDTH_IN/2,0.45])
ring_radii=np.array([1.36,0.255])
spin_directions=np.array([1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,1,-1])
spin_arrow_length_scale=1.2
level_y=[0.94,1.235,1.635]
state_labels=['$|0\\rangle$','$|\\sigma\\rangle$','$|\\epsilon\\rangle$']
plt.rcParams.update({'text.usetex':True,'text.latex.preamble':'\\usepackage{amsmath}\\usepackage{bm}','font.family':'serif','font.serif':['Times'],'font.size':8,'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'lines.linewidth':1.5,'lines.markersize':3.5,'lines.markeredgewidth':1,'savefig.facecolor':'white'})
def stroke(ax,points,color,linewidth=1.5,linestyle='-',zorder=2):
 ax.add_patch(PathPatch(MplPath(points),fill=False,edgecolor=color,linewidth=linewidth,linestyle=linestyle,capstyle='round',joinstyle='round',zorder=zorder))
def arrow(ax,start,end,color,linestyle='-',linewidth=1.5,mutation_scale=8,zorder=3,arrowstyle='-|>'):
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle=arrowstyle,color=color,linewidth=linewidth,linestyle=linestyle,mutation_scale=mutation_scale,shrinkA=0,shrinkB=0,zorder=zorder))
def matte_sphere():
 coordinates=np.linspace(-1,1,161)
 x,y=np.meshgrid(coordinates,coordinates)
 radius_squared=x**2+y**2
 z=np.sqrt(np.clip(1-radius_squared,0,1))
 diffuse=np.clip(0.16-0.48*x+0.53*y+0.61*z,0,1)
 palette=np.array([to_rgb(color)for color in('#174286','#4B84D2','#C1E1FA')])
 rgb=np.stack([np.interp(diffuse,(0,0.52,1),channel)for channel in palette.T],axis=-1)
 grain=np.random.default_rng(17).normal(0,0.013,x.shape)
 rgb=np.clip(rgb+grain[...,None],0,1)
 alpha=np.clip((1-radius_squared)*65,0,1)
 return np.dstack((rgb,alpha))
fig=plt.figure(figsize=(PR_COLUMN_WIDTH_IN,FIGURE_HEIGHT_IN))
ax=fig.add_axes([0,0,1,1])
ax.set_xlim(0,PR_COLUMN_WIDTH_IN)
ax.set_ylim(0,FIGURE_HEIGHT_IN)
ax.set_aspect('equal')
ax.axis('off')
ring_angles=np.linspace(0,2*np.pi,1025)
ring_points=ring_center+np.column_stack((np.cos(ring_angles),np.sin(ring_angles)))*ring_radii
halfway=(ring_angles.size-1)//2
stroke(ax,ring_points[:halfway+1],'#000000',linewidth=0.72,zorder=1)
stroke(ax,ring_points[halfway:],'#000000',linewidth=0.85,zorder=1)
segment_lengths=np.linalg.norm(np.diff(ring_points,axis=0),axis=1)
arc_lengths=np.r_[0,np.cumsum(segment_lengths)]
site_distances=np.linspace(0,arc_lengths[-1],spin_directions.size,endpoint=False)
theta=np.interp(site_distances,arc_lengths,ring_angles)
sites=ring_center+np.column_stack((np.cos(theta),np.sin(theta)))*ring_radii
site_radii=0.071-0.009*np.sin(theta)
sphere=matte_sphere()
spin_axis=np.array([0.16,0.987])
for site,radius,sign in zip(sites,site_radii,spin_directions):
 x,y=site
 ax.imshow(sphere,extent=(x-radius,x+radius,y-radius,y+radius),origin='lower',interpolation='bicubic',zorder=3)
 direction=sign*spin_axis
 arrow(ax,site-spin_arrow_length_scale*(radius+0.022)*direction,site+spin_arrow_length_scale*(radius+0.054)*direction,'#000000',linewidth=0.72,mutation_scale=4.8,zorder=4,arrowstyle='-|>,head_length=0.57,head_width=0.19')
for y,label in zip(level_y,state_labels):
 stroke(ax,[(0.65,y),(2.21,y)],ink,linewidth=0.82)
 ax.text(2.31,y,label,color=ink,fontsize=9,va='center')
for index,(x,cycles)in enumerate([(0.99,1.6),(1.68,2.4)]):
 start,end=(level_y[index],level_y[index+1])
 t=np.linspace(0,1,220)
 wave_y=start+0.016+t*(end-start-0.071)
 wave_x=x+0.025*np.sin(2*np.pi*cycles*t)*np.sin(np.pi*t)
 stroke(ax,np.column_stack([wave_x,wave_y]),colors[index],linewidth=1.05)
 arrow(ax,(x,end-0.058),(x,end-0.01),colors[index],linewidth=1.05,mutation_scale=6.3)
 ax.text(x+0.1,(start+end)/2,f'$\\omega_{index + 1},O_\\sigma$',color=colors[index],fontsize=8,va='center')
readout_tip=level_y[0]
readout_x=2.72
readout_head_start=readout_tip+0.068
stroke(ax,[(readout_x,level_y[2]),(readout_x,readout_head_start)],readout_color,linestyle=(0,(2.4,2.1)),linewidth=0.8)
arrow(ax,(readout_x,readout_head_start),(readout_x,readout_tip),readout_color,linewidth=0.8,mutation_scale=5.5)
ax.text(readout_x,1.72,'$\\Omega$',color=readout_color,fontsize=8.5,ha='center',va='center')
ax.text(readout_x+0.106,(level_y[0]+level_y[2])/2,'$O_\\epsilon$',color=readout_color,fontsize=8,va='center')
fig.savefig(output_path,dpi=600)
plt.close(fig)
