
import subprocess
import sys
import os
import EXAMPLE_FUNCTION

# if True:
def install_packages():
    def install(package):
        print('Installing package: ', package)
        if 'matplot' in package or 'scipy' in package or 'xgboost' in package or 'pandas' in package or 'scikit-learn' in package:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package.split('=')[0]])
        else:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except:
                subprocess.check_call([sys.executable, "-m", "pip", "install",'--use-pep517', package])

    packages = [
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'tqdm==4.65.0',
        'scipy==1.8.0', # 1.7.3 for python 3.7.16
        'scikit-learn==1.0.2',
        'xgboost==1.7.5', # 1.6.2 for python 3.7.16
        'GPy==1.10.0',
        'git+https://github.com/SheffieldML/pyDeepGP',
        'numpy==1.21.6',
        'pandas==2.0.1',
        'joblib==1.2.0',
        'ipykernel',
    ]
    print()
    print('================================================================')
    print('[ SYSTEM INFO ] - PYTHON VERSION: < %s > '%('.'.join([str(g) for g in sys.version_info[:3]])))
    print('================================================================')
    print()
    for g in packages:
        install(g)
    print()
    print()
    print()
    print()
    print()
    print()


test_function   = 'z.5.3' if EXAMPLE_FUNCTION.test_function == 0 else 'z.1.10'
deepGP_maxIters = 2000 if not test_function == 'z.5.3' else 200
inits           = [4,8,10]
pops            = [200,350,500]
n_find          = 30
noise           = 1 # percent value, do not convert to decimal.
parallel        = True
n_jobs          = int(min(os.cpu_count()-4, 32))
if n_jobs <=0:
    n_jobs = 1
models          = ['gp','aagp','localridge-gbal','xgboost-gbal','lod','lrk','slrgp','deepgp']
replicates      = 20

renames = {
    'gp':'GP (ALM)',
    'aagp':'AAGP (Adj.Ada. ALM)',
    'localridge-gbal':'LocalRidge (GBAL)',
    'xgboost-gbal':'XGBoost (GBAL)',
    'lod':'LOD (Lap.Reg. DoD)',
    'lrk':'LRK (Lap.Reg. ALM)',
    'slrgp':'SLRGP (Lap.Reg. ALC)',
    'deepgp':'DeepGP (Deep ALM)',
}

model_palette   = {
    'GP (ALM)': '#30a2da',
    'AAGP (Adj.Ada. ALM)': '#fc4f30',
    'LocalRidge (GBAL)': '#e5ae38',
    'XGBoost (GBAL)': '#6d904f',
    'XGBoost (GBAL)': '#6d904f',
    'LOD (Lap.Reg. DoD)': '#8b8b8b',
    'LRK (Lap.Reg. ALM)': '#e630e6',
    'SLRGP (Lap.Reg. ALC)': '#30309c',
    'DeepGP (Deep ALM)': '#423030',
 }

