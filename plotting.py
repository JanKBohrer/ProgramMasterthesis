#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:45:21 2019

@author: bohrer
"""

# INWORK: add title and ax labels
def plot_scalar_field_2D( grid_centers_x_, grid_centers_y_, field_,
                         tick_ranges_, no_ticks_=[5,5],
                         no_contour_colors_ = 10, no_contour_lines_ = 5,
                         colorbar_fraction_=0.046, colorbar_pad_ = 0.02):
    fig, ax = plt.subplots(figsize=(8,8))

    contours = plt.contour(grid_centers_x_, grid_centers_y_,
                           field_, no_contour_lines_, colors = 'black')
    ax.clabel(contours, inline=True, fontsize=8)
    CS = ax.contourf( grid_centers_x_, grid_centers_y_,
                     field_,
                     levels = no_contour_colors_,
                     vmax = field_.max(),
                     vmin = field_.min(),
                    cmap = plt.cm.coolwarm)
    ax.set_xticks( np.linspace( tick_ranges_[0,0], tick_ranges_[0,1],
                                no_ticks_[0] ) )
    ax.set_yticks( np.linspace( tick_ranges_[1,0], tick_ranges_[1,1],
                                no_ticks_[1] ) )
    plt.colorbar(CS, fraction=colorbar_fraction_ , pad=colorbar_pad_)
    
    
pdf_dict = {
#    "backend" : "pgf",    
    "text.usetex": True,
#    "pgf.rcfonts": False,   # Do not set up fonts from rc parameters.
#    "pgf.texsystem": "lualatex",
#    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": [
        r'\usepackage[ttscale=.9]{libertine}',
        r'\usepackage[libertine]{newtxmath}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage[]{siunitx}',        
#        r'\usepackage[no-math]{fontspec}',
        ],
    "font.family": "serif"
}
pgf_dict = {
#    "backend" : "pgf",    
    "text.usetex": True,
#    "pgf.rcfonts": False,   # Do not set up fonts from rc parameters.
#    "pgf.texsystem": "lualatex",
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
        r'\usepackage[ttscale=.9]{libertine}',
        r'\usepackage[libertine]{newtxmath}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage[]{siunitx}',
#        r'\usepackage[no-math]{fontspec}',
        ],
    "font.family": "serif"
}
    
def generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI):
    dict_ = {'lines.linewidth' : LW,
             'lines.markersize' : MS,
             'axes.titlesize' : TTFS,
             'axes.labelsize' : LFS,
             'legend.fontsize' : LFS-2,
             'xtick.labelsize' : TKFS,
             'ytick.labelsize' : TKFS,
             #{'center', 'top', 'bottom', 'baseline', 'center_baseline'}
             #center_baseline seems to be def, center is OK
             'xtick.alignment' : "center",
             'ytick.alignment' : "center",
             'savefig.dpi' : DPI
             }
    
    return dict_

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)