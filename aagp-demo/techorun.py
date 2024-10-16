#%%

'''

Package Installation

'''


import subprocess
import sys



def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
packages = [
    'matplotlib==3.7.1',
    'seaborn==0.12.2',
    'tqdm==4.65.0',
    'scipy==1.8.0',
    'scikit-learn==1.0.2',
    'xgboost==1.7.5',
    'GPy==1.10.0',
    'git+https://github.com/SheffieldML/pyDeepGP',
    'numpy==1.21.6',
    'pandas==2.0.1',
    'joblib==1.2.0',
    'ipykernel',
]
# for pack in packages:
#     install(pack)



# %pip install matplotlib==3.7.1
# %pip install seaborn==0.12.2
# %pip install tqdm==4.65.0
# %pip install scipy==1.8.0
# %pip install scikit-learn==1.0.2
# %pip install xgboost==1.7.5
# %pip install GPy==1.10.0
# %pip install git+https://github.com/SheffieldML/pyDeepGP
# %pip install numpy==1.21.6
# %pip install pandas==2.0.1
# %pip install joblib==1.2.0
# %pip install ipykernel



# =======================================================================
# =======================================================================
# =======================================================================
# =======================================================================
# =======================================================================







'''

Package Import

'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from joblib import Parallel, delayed

from modeling import model_master
from equations import get_function
from visualization import SETSTYLE, fast_plot
from simulation import run_doe
import SETTINGS
SETSTYLE('bmh', clear=True)




# =======================================================================
# =======================================================================
# =======================================================================
# =======================================================================
# =======================================================================






'''

Simulation Execution

'''

'''
'z.-1': 'Crack Stress',
         'z.1': 'Cosine',
         'z.2': 'Zackarov',
         'z.3': 'Schwefel',
         'z.4': 'Rastrigin',
         'z.5': 'Qing',
         'z.6': 'Townsend',
         'z.7': 'Adjiman',
         'z.8': 'Levy03',
         'z.9': 'Branin',
         'z.10': 'Keane',
         'z.11': 'Tripod',
         'z.12': 'Bird',
         'z.13': 'CosineValley',
         'z.14': 'Deceptive',
         'z.15': 'MatalaePeak',
         'z.16': 'Alpine',
         'z.17': 'Quartic',
         'z.18': 'DP',
         'z.19': 'Bohavchevsky',

'''


replicates = 20
parallel   = True
functions = [
    'z.16.2',
    'z.8.2',
    'z.5.3',
    'z.11.3',
    'z.7.6',
    'z.12.6',
    'z.1.10',
    'z.4.10',
    'z.10.15',
    'z.3.15',
]
for tf in functions:
    print('Running Test Function: ', tf)
    DF,XA,YA = run_doe(
                models          = SETTINGS.models,
                test_function   = tf,#SETTINGS.test_function,
                inits           = SETTINGS.inits,
                pops            = SETTINGS.pops,
                noise           = SETTINGS.noise,
                n_find          = SETTINGS.n_find,


                replicates      = replicates,
                parallel        = parallel,
                nJobs           = -4,
                )
    DF.to_csv('localridge-gbal %s.csv'%(tf.replace('.','-')))




# =======================================================================
# =======================================================================
# =======================================================================
# =======================================================================
# =======================================================================



color_idx = 0
colors = {
    ['gp','GP (ALM)'][color_idx]: '#30a2da',
    ['aagp','AAGP (Adj.Ada. ALM)'][color_idx]: '#fc4f30',
    ['localridge','LocalRidge (GBAL)'][color_idx]: '#e5ae38',
    ['localridge-gbal','LocalRidge (GBAL)'][color_idx]: '#e5ae38',
    ['xgboost','XGBoost (GBAL)'][color_idx]: '#6d904f',
    ['xgboost-gbal','XGBoost (GBAL)'][color_idx]: '#6d904f',
    ['lod','LOD (Lap.Reg. DoD)'][color_idx]: '#8b8b8b',
    ['lrk','LRK (Lap.Reg. ALM)'][color_idx]: '#e630e6',
    ['slrgp','SLRGP (Lap.Reg. ALC)'][color_idx]: '#30309c',
    ['deepgp','DeepGP (Deep ALM)'][color_idx]: '#423030',

    ['localridge*','LocalRidge (GBAL)'][color_idx]: 'lime',
    ['loess','LocalRidge (GBAL)'][color_idx]: 'orangered',
    ['loess*','LocalRidge (GBAL)'][color_idx]: 'firebrick',

    
    ['localridge-gbal','LocalRidge (GBAL)'][color_idx]: 'cyan',
    ['localridge*-gbal','LocalRidge (GBAL)'][color_idx]: 'dodgerblue',
    ['loess-gbal','LocalRidge (GBAL)'][color_idx]: 'magenta',
    ['loess*-gbal','LocalRidge (GBAL)'][color_idx]: 'purple',

    
    ['gp-gbal','GP (ALM)'][color_idx]: 'firebrick',
    ['gp-maximin','GP (ALM)'][color_idx]: 'lime',


 }

fig = plt.figure(figsize=(10,5),dpi=150)
ax = fig.add_subplot(2,2,1)
ax = sns.lineplot(
    ax=ax,
    data = DF,
    x    = 'Samples Added',
    y    = 'WMAPE',
    hue  = 'Model',
    errorbar = None, #('pi',100)
    palette = colors,
)
ax.set_title('WMAPE')
ax.locator_params(axis='y', nbins=4)


ax = fig.add_subplot(2,2,2)
ax = sns.lineplot(
    ax=ax,
    data = DF,
    x    = 'Samples Added',
    y    = 'RMSE',
    hue  = 'Model',
    errorbar = None, #('pi',100)
    palette = colors,
)
ax.set_title('RMSE')
ax.locator_params(axis='y', nbins=4)


ax = fig.add_subplot(2,2,3)
ax = sns.lineplot(
    ax=ax,
    data = DF,
    x    = 'Samples Added',
    y    = 'SMAPE',
    hue  = 'Model',
    errorbar = None, #('pi',100)
    palette = colors,
)
ax.set_title('SMAPE')
ax.locator_params(axis='y', nbins=4)

m  = 'COD'
ax = fig.add_subplot(2,2,4)
ax = sns.lineplot(
    ax=ax,
    data = DF,
    x    = 'Samples Added',
    y    = m,
    hue  = 'Model',
    errorbar = None, #('pi',100)
    palette = colors,
)
ax.set_title(m)
ax.locator_params(axis='y', nbins=4)
plt.suptitle('%s Reproduction'%(tf))

plt.tight_layout()