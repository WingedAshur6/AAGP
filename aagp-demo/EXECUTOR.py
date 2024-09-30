#%%
# [1] - import the required packages
# =============================================
import os
import SETTINGS
SETTINGS.install_packages()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from timeit import default_timer as defTime

from visualization import SETSTYLE
from simulation import run_doe
SETSTYLE('bmh', clear=True)
current_directory = os.path.dirname(os.path.abspath(__file__))

# [2] - set up and run the simulation
# ==============================================
print()
print('Running simulation in directory ---->', current_directory)
test_function = 'Qing (3D)' if 'z.5' in SETTINGS.test_function.lower() else 'Cosine (10D)'
print('Test Function  ---> ',test_function)
models_1 = [g for g in SETTINGS.models if not 'deepgp' in g.lower()]
models_2 = [g for g in SETTINGS.models if not g in models_1]
print('Group 1 models ---> ', models_1)
print('Group 2 models ---> ', models_2)
tstart = defTime()

# [3] - set up the simulation parameters
# ========================================
simulation_parameters = dict(
                test_function   = SETTINGS.test_function,
                inits           = SETTINGS.inits,
                pops            = SETTINGS.pops,
                noise           = SETTINGS.noise,
                n_find          = SETTINGS.n_find,
                replicates      = SETTINGS.replicates,
                parallel        = SETTINGS.parallel,
                nJobs           = SETTINGS.n_jobs,
)

# [4] - run group 1 in parallel
# ==============================================
DF_1 = pd.DataFrame()
if True:
    simulation_parameters['models'] = models_1
    DF_1,XA,YA = run_doe(**simulation_parameters)

tend1 = defTime()
tend = tend1-tstart
print('Group 1 completed. Time   : %0.2fs (%0.2fmins, %0.2fhrs)'%(tend,tend/60,tend/3600))

# [5] - run group 2 in series (memory issue)
# ==============================================
DF_2 = pd.DataFrame()
if True:
    simulation_parameters['models'] = models_2
    DF_2,XA,YA = run_doe(**simulation_parameters)
tend2 = defTime()
tend = tend2-tend1
tender = tend2 - tstart
print('Group 2 completed. Time   : %0.2fs (%0.2fmins, %0.2fhrs)'%(tend,tend/60,tend/3600))
print('All groups completed. Time: %0.2fs (%0.2fmins, %0.2fhrs)'%(tender,tender/60,tender/3600))


# [6] - Concatenate the results, visualize, and output
# ======================================================
DF = pd.concat((DF_1,DF_2),axis=0,ignore_index=True)
DF['Model'] = DF['Model'].replace(to_replace=SETTINGS.renames)
agg = DF.groupby(['Model','Samples Added']).agg({'NRMSE':'mean'}).reset_index()
fig = plt.figure(figsize=(8,5),dpi=350)
ax = fig.add_subplot(1,1,1)
sns.lineplot(
    data = agg,
    x = 'Samples Added',
    y = 'NRMSE',
    hue = 'Model',
    palette = SETTINGS.model_palette,
    linewidth=2
)
ax.set_title(test_function)
ax.legend(loc='upper right', bbox_to_anchor=(1.5,1), frameon=True, shadow=True)
plt.tight_layout()
plt.savefig(current_directory + '/Example Output.jpg',dpi=350)
print('Output JPG saved to ---->', current_directory + '/' + 'Example Output.jpg')