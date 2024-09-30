
#%%
# [1] - python package import
# =================================
import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import euclidean_distances
# from pairwise_distances import euclidean_distances
# import os

# [2] - self-made packages
# =================================
from algorithms import lhs_sampling

cutta = lambda lo, hi, dims: [[lo for g in range(dims)], [hi for g in range(dims)]]


def testFunctions(fName):
    from scipy.special import factorial
    "[ INTRO ] - This is to conglomerate as many functions as I can and give them multi-dimensional support between the boundaries [-10,10]"
    "[ NOTE ] - The inputs x MUST BE BETWEEN [-10,10] FOR THE FUNCTIONS TO WORK."

    ## [ LIST OF FUNCTIONS ]
    ##-------------------------------------
    # [5] - Qing          : http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html

    ##[ FUNCTION BOUNDARIES ]
    ##--------------------------------------
    # [5] - Qing          : x = [-2,2]


    '  [ LIST OF N-Dimensional FUNCTIONS ] '
    '-------------------------------------'
    # [z.1] - Cosine Mixture
    # [z.2] - Zackarov
    # [z.4] - Rastrigin
    # [z.5] - Qing
    # [z.8] - Levy03
    # [z.3]- Schwefel20
    # [z.13] - Trid

    cutta = lambda lo, hi, dims: [[lo for g in range(dims)], [hi for g in range(dims)]]

    def convertBounds(x, bounds):
        bLo, bHi = bounds
        xg = (x+10)/20
        # xg = np.matrix(np.copy(x))
        h = 0
        for g in range(x.shape[1]):
            if h == len(bLo):
                h = 0
            xgmin,xgmax = xg[:,g].min(),xg[:,g].min()

            # xg[:,g] = (xg[:,g] - xgmin)/(xgmax-xgmin)
            blo     = bLo[h]
            bhi     = bHi[h]
            xg[:,g] = xg[:,g] * (bhi - blo) + blo
            h += 1
        return(xg)
    
    def f1_cosine(xIn, bounds):

        x = convertBounds(xIn,bounds)
        y1= 0
        y2= 0
        for i in range(x.shape[1]):
            xi = x[:,i]
            y1 = y1 + np.cos(5.0*np.pi*xi)
            y2 = y2 - np.square(xi)
        y = 10 * (0.1 * y1 + y2)
        return y

    def f5_qing(xIn,bounds):
        x = convertBounds(xIn,bounds)
        y = 0
        for i in range(x.shape[1]):
            xi = x[:,i]
            y = y + np.square(np.square(xi)-(i+1))
        return y
    
    def f12_mishraBird(x, bounds): ## mishra's bird
        x = convertBounds(x, bounds)
        y = 0

        for i in range(x.shape[1]-1):
            j = i+1

            xi = x[:,i]
            xj = x[:,j]

            y1 = np.multiply(np.sin(xi), np.exp(np.square(1-np.cos(xj))))
            y2 = np.multiply(np.cos(xj), np.exp(np.square(1-np.sin(xi))))
            y3 = np.square(xi-xj)

            y  = y + y1 + y2 + y3
        return y


    fSplit = fName.split('.')
    dims   = int(fSplit[-1])
    f      = int(fSplit[1])


    funcBounds = {
            5: [f5_qing, [-2,2]],
            12: [f12_mishraBird, [-2*np.pi,2*np.pi]],
            1: [f1_cosine, [-1,1]],
            }
    
    func, bounds = funcBounds[f]
    boundsIn = cutta(bounds[0],bounds[1], dims)
    FUNC   = lambda input: func(np.matrix(input).reshape(-1,dims), boundsIn)
    bounds_ = cutta(-10,10, dims)


    return FUNC, bounds_


def get_case_study_name(fName, addDim = 0):

    fSplit = fName.split('.')
    f      = fSplit[0]+'.'+fSplit[1]

    fz = {
         'z.1': 'Cosine',
         }
    fz = {
         'z.5': 'Qing',
         }
    fz = {
         'z.12': 'Bird',
         }

    addDim = addDim * ((not 'c' in fName) or 'eyelink' in fz[f])
    try:
        nameOut = fz[f]+['-%sD'%(fName.split('.')[-1]) if addDim == True else ''][0]
    except:
        nameOut = fz[fName]+['-%sD'%(fName.split('.')[-2]) if addDim == True else ''][0]
    return nameOut






def get_function(fName, n=300, e=1000, lhs_iters = 250, structure_noise=0.075, scale=0.5):
    
    if fName == 'mlp_mnist':
        import pandas as pd
        DF = pd.read_csv('mlp_trainer_mnist_4layers.csv')
        return DF


    # [INTRO] - this grabs the functions we want from < testFunctions >.
    f, b = testFunctions(fName)
    p    = len(b[0])

    # print('generating training points.')
    xa   = lhs_sampling(n=n,p=p, seed=777, iterations=lhs_iters)

    # print('generating evaluation points.')
    xe   = lhs_sampling(n=e,p=p, seed=777, iterations=lhs_iters)

    bHi = b[1]
    bLo = b[0]
    for i in range(xa.shape[1]):
        xg = xa[:,i]
        xg = (xg - np.min(xg))/(xg.max()-xg.min()) * (bHi[i] - bLo[i]) + bLo[i]
        xa[:,i] = xg


        xg = xe[:,i]
        xg = (xg - np.min(xg))/(xg.max()-xg.min()) * (bHi[i] - bLo[i]) + bLo[i]
        xe[:,i] = xg

    ya   = f(xa)
    ye   = f(xe)

    return [np.asarray(g) for g in [xa, ya, xe, ye]] +  [f, b]
