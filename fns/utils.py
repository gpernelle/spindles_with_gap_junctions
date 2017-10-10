__author__ = 'G. Pernelle'

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import time

## for data processing and plotting
# import pandas as pd
# import re, csv, os, datetime, random, os, sys, sh, math, socket, time
# from scipy.fftpack import fft
# from matplotlib import gridspec
# from matplotlib.pyplot import cm
# import matplotlib.colors as colors
# from matplotlib.mlab import psd
# from scipy.misc import comb
# from scipy import sparse, signal
# import seaborn as sns
# import importlib as imp

## paper colors
Icolor = '#3366cc'
Ecolor = '#FF6868'
gammaColor = "#66ccff"

## for nice plots
def update_mpl_settings():
    # Direct input
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath, lmodern}"]
    # Options
    fontsize = 22
    params = {'text.usetex' : True,
              'font.family' : 'lmodern',
              'text.latex.unicode': True,
              'text.color':'black',
              'xtick.labelsize': fontsize-2,
              'ytick.labelsize': fontsize-2,
              'axes.labelsize': fontsize,
              'axes.labelweight': 'bold',
              'axes.edgecolor': 'white',
              'axes.titlesize': fontsize,
              'axes.titleweight': 'bold',
              'pdf.fonttype' : 42,
              'ps.fonttype' : 42,
              'axes.grid':False,
              'axes.facecolor':'white',
              'lines.linewidth': 1,
              "figure.figsize": '5,4',
              }
    plt.rcParams.update(params)


update_mpl_settings()

pgf_with_pdflatex = {
    "pgf.texsystem": "lualatex",
    "pgf.preamble": [
        r'\usepackage{amsmath,lmodern}',
        r'\usepackage[scientific-notation=true]{siunitx}',
        ]
}

mpl.rcParams.update(pgf_with_pdflatex)