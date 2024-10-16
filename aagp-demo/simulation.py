# [1] - python package import
# ==================================================
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
import warnings


# [2] - self-made package import
# ==================================================
from equations import get_function
from modeling import(
    model_master,
    calculate_regression_metrics
)
from auxiliaries import ProgressParallel


# [3] - functions
# ==================================================



def DOE(
        xa,ya,
        xe,ye,
        seed_idx,
        n_find = 30,
        model  = 'gp',
        acquisition = None,
        seed        = 0,
        verbose     = False,
):
    '''
    # Run DOE
    This function executes a DOE using the `model` and `acquisition` via `model_master`.
    '''
    import timeit
    defTime = lambda: timeit.default_timer()
    idx = seed_idx.copy()
    jdx = [g for g in range(xa.shape[0]) if not g in idx]

    xs  = xa[idx,:]
    ys  = ya[idx,:]
    xz  = xa[jdx,:]
    yz  = ya[jdx,:]



    _metrics = ['r2', 'mae', 'mse', 'rmse','nrmse', 'mape', 'mmape', 'wmape', 'smape','bias', 'adjusted_r2', 'cod', 'mbd', 'cv']
    biases = []
    rmses  = []
    wmapes = []
    smapes = []
    R2s = []
    fit_times = []

    metrics = {}
    for g in _metrics:
        metrics[g] = []

    for i in range(n_find+1):

            
        # [1] - fit the model
        # =================================
        xa = np.vstack((xs,xz))

        model_timer = defTime()
        mu,sig = model_master(
            x = xs,
            y = ys,
            xa= xa,
            model_name = model,
            acquisition = acquisition,
        )

        # [2] - calcualte the performance metrics
        # =======================================
        model_timer = defTime()-model_timer

        yp = np.ravel(mu(xe))
        yt = np.ravel(ye)

        for _m_ in _metrics:
            __m__ = calculate_regression_metrics(
                y_true = yt,
                y_pred = yp,
                p      = xs.shape[1],
                metric = _m_
            )
            metrics[_m_].append(__m__)

        # yp = np.ravel(mu(xe))
        # error= np.ravel(ye) - yp
        # # overEst = np.clip(error,0,None)
        # # underEst= np.clip(error,None,0)*-1

        # # overEst = np.where(error>0, np.abs(error),0)
        # # underEst= np.where(error<=0, np.abs(error),0)
        # # bias    = np.divide(overEst-underEst,overEst+underEst).mean()
        # # wmape   = np.abs(error).sum()/np.abs(ye).sum()
        # # rmse    = np.sqrt(np.square(error).mean())
        # # smape   = np.divide(
        # #                     np.ravel(np.abs(error)),
        # #                     (np.abs(np.ravel(yp)) + np.abs(np.ravel(ye)))/2
        # #                     ).mean()
        # # R2 = r2_score(np.ravel(ye), np.ravel(yp))

        # biases.append(bias)
        # wmapes.append(wmape)
        # rmses.append(rmse)
        # smapes.append(smape)
        # R2s.append(R2)


        fit_times.append(model_timer)

        # [3] - select the next optimal point to test
        # ==============================================
        vhat = sig(xz)
        sdx  = np.argmax(np.ravel(vhat))
        
        xOpt = xz[[sdx],:]
        yOpt = yz[[sdx],:]
        xs   = np.vstack((xs,xOpt))
        ys   = np.vstack((ys,yOpt))
        xz   = np.delete(xz,sdx,axis=0)
        yz   = np.delete(yz,sdx,axis=0)
        if verbose:
            _nrmse = '%0.3f'%(metrics['nrmse'][-1])
            _wmape = '%0.3f'%(metrics['wmape'][-1])
            print(f'{model} - {i}/{n_find} | nrmse: {_nrmse} | wmape: {_wmape}')
    
    # [3] - create the pandas dataframe to show the results
    # ======================================================
    # df = pd.DataFrame(
    #         {
    #             'WMAPE':wmapes,
    #             'SMAPE':wmapes,
    #             'R2':R2s,
    #             'RMSE':rmses,
    #             'Bias':biases,
    #             'Fit Time':fit_times
    #         }
    #     )
    df                  = pd.DataFrame()
    df['Samples Added'] = [g for g in range(n_find+1)]
    df['Model'] = model
    df['Acquisition'] = acquisition.upper() if not acquisition is None else 'Default'
    df['Initial Points'] = len(idx)/xa.shape[1]
    df['Total Population'] = xa.shape[0]
    df['No. Variables'] = xa.shape[1]
    df['Seed'] = seed
    df['Fit Time'] = fit_times
    for metric in list(metrics.keys()):
        m = metric
        if 'adjusted_r2' in m:
            m = 'a-r2'
        df[m.upper()] = metrics[metric]

    # df['mu'] = mu
    # df['sig']= sig
    return df

def prepare_doe(
        models= ['gp','aagp','lod','lrk','xgboost','localridge','deepgp'],
        inits = [4,8,10],
        pops  = [200,350,500],
        replicates = 20,
        noise = 1, # percent value, do not convert to decimal.
        test_function = 'z.15.2',
):
    seed_counter = []
    seed_indices = []
    seed_noises  = []
    modelouts    = []
    inits_counter= []
    pops_counter = []
    XA = []
    YA = []
    XE = []
    YE = []
    for model in models:
        for T in pops:
            xa, ya, xe, ye, f, b = get_function(
                                                fName = test_function,
                                                n = T,e=1000,lhs_iters=10,
                                            )
            np.random.seed(777)
            for I in inits:
                m     = I * xa.shape[1]
                seeds      = [np.random.choice(xa.shape[0],m,replace=False) for g in range(replicates)]
                seedNoises = (np.random.rand(ya.shape[0],replicates) * 2 - 1) * noise * (ya.max()-ya.min())/100
                counter = 0
                # print(seeds)
                # print(np.column_stack((ya,seedNoises[:,[0]])))
                # import sys
                # sys.exit('test')
                for g in range(replicates):
                    # seed = np.random.choice(xa.shape[0],m,replace=False)
                    # nois = (np.random.rand(ya.shape[0],1) * 2 - 1) * noise * (ya.max() - ya.min())/100


                    # print(model,T,I,seed)
                    seed_indices.append(seeds[g])
                    seed_noises.append(seedNoises[:,[g]] + ya)
                    YA.append(seedNoises[:,[g]] + ya)
                    XA.append(xa)
                    YE.append(ye)
                    XE.append(xe)
                    seed_counter.append(counter)
                    modelouts.append(model)
                    pops_counter.append(T)
                    inits_counter.append(I)
                    counter+=1

    # sys.exit('test'):
    # for i in range(len(seed_indices)):
    #     print('*******************************************')
    #     print('Total Population: %s'%(XA[i].shape[0]))
    #     print('Initial Points: %s'%(len(seed_indices[i])/XA[i].shape[1]))
    #     print(seed_indices[i])
    # import sys
    # sys.exit('simulation.py line 174 checking the seeds and points.')
    return modelouts,XA,YA,XE,YE,seed_indices, seed_counter, pops_counter,inits_counter

def run_doe(
    test_function = 'z.15.2',
    models      = ['gp','aagp','lod','lrk','xgboost','localridge','deepgp'],
    acquisition = None,
    inits       = [4,8,10],
    pops        = [200,350,500],
    replicates  = 1,
    n_find      = 30,
    noise       = 1, # percent value, do not convert to decimal.,
    parallel    = True,
    nJobs       = -1
):

    M,XA,YA,XE,YE,SI,SC,TC,IC      = prepare_doe(test_function=test_function,models=models,inits=inits, pops=pops, replicates=replicates, noise=noise)
    print('Prepping Models.')
    if acquisition is None:
        acquisition = []
        for g in models:
            if '-' in g:
                # print('Populating Acquisition: ', g.split('-'))
                acquisition.append(g.split('-')[-1])
            else:
                acquisition.append(None)


    APP             = lambda iii: DOE(
        xa          = XA[iii],
        ya          = YA[iii],
        xe          = XE[iii],
        ye          = YE[iii],
        model       = M[iii],
        acquisition = acquisition[ models.index(M[iii]) ],
        seed        = SC[iii],
        seed_idx    = SI[iii],

        n_find      = n_find,
    )

    print('Running Simulation. Replicates: ',replicates, '%s Jobs: '%('Parallel' if parallel else 'Series'), nJobs)
    with warnings.catch_warnings():
        np.seterr('ignore')
        warnings.simplefilter('ignore')
        if parallel:
            DF = pd.concat(
                # Parallel(n_jobs = nJobs)(delayed(APP)(i) for i in range(len(M))),
                ProgressParallel(n_jobs = nJobs, use_tqdm=False, timeout=None, backend='loky')(delayed(APP)(i) for i in range(len(M))),

                axis=0,
                ignore_index=True
            )
            
        else:
            DF = pd.concat(
                [APP(i) for i in range(len(M))], axis=0, ignore_index=True
            )
    DF['Test Function'] = test_function
    return DF,XA,YA


if __name__ == '__main__':
    models = ['gp','aagp','lod','lrk']
    inits  = [4,8,10]
    pops   = [200,350,500]
    replicates = 5
    noise      = 1 # percent value, do not convert to decimal.
    parallel   = True

    DF = run_doe(
        models          = models,
        test_function   = 'z.5.3',
        acquisition     = None,
        inits           = inits,
        pops            = pops,
        replicates      = replicates,
        noise           = 1,
        parallel        = parallel,
    )

    import seaborn as sns
    sns.lineplot(
    data = DF,
    x    = 'Samples Added',
    y    = 'WMAPE',
    hue  = 'Model'
    )