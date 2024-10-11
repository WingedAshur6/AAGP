# [0] - package import
# =================================
import itertools
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as ED, manhattan_distances as MD

# [1] - Developed packages
# ==========================================
from auxiliaries import(
    VS
)



# [2] - functions
# =================================



def maxiMin_acquisition(xRef,xIn):
    return (euclidean_distances(xRef,xIn).min(0)).reshape(-1,1)

def lhs_sampling(n=10, p=2, seed=777, iterations=1000, sampling_mode=['distance','normsort'][0]):
    def lhs_sampler(n=n, p=p, seed=seed):
        # [INTRO] - uses uniform sampling to generate LHS samples.
        np.random.seed(seed)
        x = np.random.uniform(size=[n, p])
        for i in range(0, p):
            x[:, i] = (np.argsort(x[:, i]) + 0.5) / n
        return x

    minDist = -np.inf
    for i in range(iterations + 1):
        x = lhs_sampler(n=n, p=p, seed=seed + i)

        if 'dist' in sampling_mode:
            d = euclidean_distances(x, x)
            d = d[d>0].min()
            # d = d[d > 0]
            # d = d.min()
            if d > minDist:
                minDist = d
                xOut = x
        else:
            d = np.ravel(np.sqrt(np.square(x).sum(1)))
            d = d[np.argsort(d)]
            d = d[1]-d[0]
            if d > minDist:
                minDist = d
                xOut    = x
    return xOut


def FFD(dims, levels):
    '''
    # FFD
    generates a full factorial design of `dims` dimensions.
    - `levels` can either be a single INT or a LIST of `dims` INTS.
    '''
    if isinstance(levels, list):
        yeets = [np.linspace(0, dims, levels[g]).tolist()
                 for g in range(len(levels))]
    else:
        yeets = [list(range(dims)) for g in range(dims)]
    return np.array(list(itertools.product(
        *[g for g in yeets]))) / np.max(yeets)


def geoSpace(n=300,xa=None, m=10, x = None, bins = 10,p=None, verbose=False):
    '''
    # GeoSpace
    This function takes input points and selects `m` points from varying `bins` of increasing distance.
    '''
    if xa is None:

        if not x is None:
            p = x.shape[1]
        if p is None:
            print('ERROR: < p > variable for dimensions is not defined.')
        xa = lhs_sampling(n=n, p=p, iterations=5000, sampling_mode='normsort')
    xa_ = xa.copy()

    bins_ = int(np.copy(bins))

    failure = 0
    VS = lambda x1,x2: np.matrix(np.vstack((x1,x2)))
    da = euclidean_distances(xa,xa)
    da_= da # da[da>0]
    spaces = np.linspace(da_.min(), da_.max(), bins)

    if x is None:
        x = xa.min(0)
    x_ = x.copy()


    counter = 0
    while counter < m and failure <= bins:
        for i in range(len(spaces)-1):
            h0 = spaces[i]
            h1 = spaces[i+1]


            grabs = list(set(np.where((da>=h0) & (da<h1))[0].tolist()))
            xG    = xa[grabs,:]
            try:
                d     = euclidean_distances(x,xG).min(0)
                failure= 0
            except:
                # print('WARNING. NO POINTS IN THIS DISTANCE: \n%s-%s: %s/%s'%(h0,h1,i+1,len(spaces)-1))
                failure += 1
                continue
            idx   = np.argmax(d)
            x     = VS(x,xa[idx,:])
            xa    = np.delete(xa,idx,axis=0)
            da    = np.delete(da,idx,axis=0)
            da    = np.delete(da,idx,axis=1)
            counter += 1
            if verbose==True:
                print('%s/%s points found.'%(counter+1,m))
            # print(xs)
    if failure > bins:
        bins -= 1
        x,xa = geoSpace(xa_, m = m, x = x_, bins = bins)

    return x,xa

def lhs_select(xa, m=500, maxIters=500, verbose=True):
    '''
    # LHS Select
    This function takes an input dataset `xa` and selects `m` points from it that best maximize the latin hypercube criterion.
    '''
    idx_tril = np.tril_indices(m,k=-1)
    def ckm(xIn):
        d = euclidean_distances(xIn)[idx_tril]
        return d[d>0].min()

    idx = np.random.choice(xa.shape[0], m, replace=False)
    idx_out = idx
    xOut= xa[idx,:]
    s0  = ckm(xOut)
    for i in range(maxIters):
        np.random.seed(777+i)
        idx   = np.random.choice(xa.shape[0], m, replace=False)
        xTest = xa[idx,:]
        s1    = ckm(xTest)

        if s1 > s0:
            s0 = s1
            xOut = xTest
            idx_out = idx

            if verbose==True:
                print('Iteration: %s/%s | Score: %0.3f'%(i+1,maxIters, s1))

    jdx = [g for g in range(xa.shape[0]) if not g in idx_out]
    return xOut, xa[jdx,:], idx_out, jdx




def GET_LAPLACIAN_BOUNDS():
    '''
    # Get Laplacian Bounds
    This function is a simple grabber to return consistent laplacian regularization penalty coefficient upper and lower bounds for ALL applications.
    '''
    return [0,1e-2]



def euclidean_distances(x1,x2 = None):

    '''
    # Euclidean distances
    This computes the euclidean distances between two points, enforcing `np.array` for compatibility issues.
    '''

    if x2 is None:
        return ED(np.asarray(x1), np.asarray(x1))
    else:
        return ED(np.asarray(x1), np.asarray(x2))
    
def manhattan_distances(x1,x2 = None):

    '''
    # Manhattan distances
    This computes the manhattan distances between two points, enforcing `np.array` for compatibility issues.
    '''

    if x2 is None:
        return MD(np.asarray(x1), np.asarray(x1))
    else:
        return MD(np.asarray(x1), np.asarray(x2))


def SLOGDET(H, y, eps = 0.000000125):

    '''
    # SLOGDET
    This function computes the GP likelihood via spectral decomposition
    '''

    # [1] - try to perform SVD
    # ===============================
    success = True
    try:
        u,s,v = np.linalg.svd(H)


    # [2] - if it fails, then add a ridge parameter to the hessian to increase stability
    # ====================================================================================
    except:
        success = False
    
    if success == False:
        counter = 0
        while counter < 100:
            H = H + np.eye(H.shape[0]) * eps
            try:
                u,s,v = np.linalg.svd(H)
                success = True
                break
            except:
                counter += 1
        if counter == 100:
            print('[ FATAL ] - Algorithms.SLOGDET: SVD failed.')
            np.linalg.svd(H)
    
    # [3] - if it succeeds, then we can continue with the spectral decomposition.
    # ===========================================================================
    s[s==0] = 1e-20
    si      = np.diag(np.ravel(np.divide(1.0,s)))
    HI      = u @ si @ v
    sProd   = np.product(np.abs(s))

    m       = float(np.trace(y.T @ HI @ y))
    logDet  = 0
    if sProd > 0:
        try:
            logDet = np.product(np.sign(s)) * np.log(sProd)
        except:
            try:
                logDet = np.product(np.sign(s)) * np.log(sProd)
            except:
                logDet = np.product(np.sign(s)) * np.log(sProd)
    
    LL = m + logDet
    return LL


def trainTest_binSplitter(my_list, n=-1, x=None):

    '''
    Bin splitter for training and testing
    '''
    n=int(np.ceil(len(my_list))*0.75)
    nSplits = n
    if n == -1:
        m = len(my_list)
        if m < 20:
            nSplits = x.shape[0]
            if m >= 20 and m < 40:
                nSplits = 5
            if nSplits >= 40:
                nSplits = -10


    if nSplits > 0:
        xOut = [g.tolist() for g in np.array_split(my_list,int(np.abs(nSplits)))]
    elif not x is None:
        splits = geoSpace(xa=x,m=int(x.shape[0]/10), p=x.shape[1])[0]
        splitter=[]
        for i in range(splits.shape[0]):
            idx = np.argmin(np.ravel(np.abs(splits[i,:]-x).sum(1)))
            splitter.append(idx)
        splits = [splitter] + [[g for g in my_list if not g in splitter]]
        xOut=splits
    return xOut




def CL(K, normalize=False, walk=False, signless=False):
    '''
    # Compose Laplacian (CL)
    This function takes a symmetric kernel matrix `K` and generates the Laplacian matrix `L = S - K`, where `S` is `diag(K.sum(1))`.
    - `normalize`: normalize the laplacian (default is `False`)
    - `walk`: make random-walk laplacian (default is `False`)
    - `signless`: make signless laplacian (default is `False`)
    '''

    S = np.ravel(K.sum(1))
    SS= np.diag(S)
    if signless:
        L = SS + K
    else:
        L = SS - K
    
    if normalize or walk:
        SD = np.sqrt(S.copy())
        SD[SD==0] = 1

        D = np.diag(np.ravel(np.divide(1.0, SD)))
        if not walk:
            L = D @ L @ D
        else:
            L = D @ L
    return L


def densityGrabber(xIn, gIn=None, nSamples=250):
    '''
    # Density grabber
    This function fits a density to `xIn` and selects a point from `gIn` using `nSamples` that have the highest density with respect to `xIn`.
    '''

    if gIn is None:
        gIn = euclidean_distances(xIn,xIn)[np.tril_indices(xIn.shape[0], k=-1)]
    
    sets = np.linspace(gIn.min(), gIn.max(), nSamples).reshape(-1,1)
    gIn  = gIn.reshape(-1,1)

    grabs = np.exp(
        -1.0 * np.square(
            euclidean_distances(sets,gIn) / np.percentile(gIn,50)
        )
    ).sum(1)

    s = sets[np.argmax(grabs),0]
    return s



def SLV(a,b=None, safetyRidge = 0.0000125):
    '''
    # System of Equations Solver
    This function solves a system of equations `ax = b` or inverts a matrix `a` if `b=None`.
    '''
    if b is None:
        try:
            mi = np.linalg.pinv(a)
        except:
            mi = np.linalg.pinv(a + np.eye(a.shape[0]) * safetyRidge)

    else:
        try:
            mi = np.linalg.solve(a,b)
        except:
            mi = np.linalg.pinv(a) * b

    return mi




def subsampler(x_,sampling=0.25, verbose=True):
    if sampling < 1 and sampling > 0:
        sampling = int(x_.shape[0] * sampling)
    idxs = []
    x = x_.copy()
    idx = np.argmin(np.sqrt(np.square(x).sum(1)))
    xout = x[idx,:].reshape(1,-1)
    # x = np.delete(x,idx,axis=0)
    idxs.append(idx)
    # for i in TQDM(range(sampling-1),disable=not verbose, desc='Subsampling %s points from %s'%(sampling, x_.shape[0])):
    for i in range(sampling-1):

        d = euclidean_distances(xout,x).min(0)
        idx = np.argmax(d)
        xout = np.vstack((xout,x[idx,:].reshape(1,-1)))
        idxs.append(idx)
        # x = np.delete(x,idx,axis=0)

    return xout,idxs

def GBAL_acq(xRef, xIn, dFunc = [euclidean_distances, manhattan_distances], simple=False):
    dFunc = dFunc[int(simple)]

    scores = []
    ranger = range(xIn.shape[0])
    if simple==True:
        d0 = dFunc(xRef, xIn).min(0)
        scores = d0

    else:
        d00 = dFunc(xRef,xIn)#.min(0)
        for i in ranger:
            j  = [g for g in ranger if not g == i]

            d0 = d00[:,j].min(0)
            d1 = dFunc(VS(xRef,xIn[[i],:]), xIn[j,:]).min(0)

            d  = d0.sum()-d1.sum()
            scores.append(d)
    scores = np.matrix(scores).reshape(-1,1)
    return scores

def mutual_euclid(xRef, xIn, xall, simple=True):
    # this function does mutual information based on euclidean distances and measured/unmeasured sets.

    scores = xIn[:,0].reshape(-1,1) * 0
    for i in range(xIn.shape[0]):
        zRef = np.delete(xIn.copy(), i,axis=0)
        xscore = GBAL_acq(np.vstack((xRef,xIn[i,:].reshape(1,-1))),xall, simple=simple).mean()
        zscore = GBAL_acq(zRef,xall, simple=simple).mean()

        scores[i,:] = float(zscore/xscore)
    return scores.reshape(-1,1)

def GREEDY_acq(xRef, mu, xIn, dFunc = euclidean_distances, improveGreed = False):
    yRef   = mu(xRef)
    yIn    = mu(xIn)
    mode   = ['GSy', 'iGS'][int(improveGreed)]


    if mode == 'iGS':
        d0 = np.multiply(dFunc(xRef, xIn), dFunc(yRef, yIn)).min(0)
    else:
        d0 = dFunc(yRef, yIn).min(0)


    scores = d0

    scores = np.matrix(scores).reshape(-1,1)
    return scores



def get_kernel_data_bounds(x,y,xa=None, semi_variance=False, low_noise=True):
    '''
    # Get Kernel Data Bounds
    This function grabs data-driven boundaries for kernel based models.
    '''

    # [1] - grab the lower triangle indices
    # ======================================
    trils = np.tril_indices(x.shape[0],k=-1)
    if xa is None:
        trila = trils.copy()
        xa    = x.copy()
    else:
        trila = np.tril_indices(xa.shape[0], k=-1)
    

    # [2] - compute the pairwise differences in y
    # =============================================
    v = euclidean_distances(y)
    if semi_variance:
        v = np.square(v)
    vt = v[trils]

    # [3] - grab the distances for each dimension of < x > and < xa >
    # =================================================================
    # '''
    da = np.zeros((xa.shape[0],xa.shape[0],xa.shape[1]))
    ds = np.zeros((x.shape[0],x.shape[0],x.shape[1]))
    for i in range(da.shape[2]):
        # da[:,:,i] = euclidean_distances(xa[:,[i]])
        # ds[:,:,i] = euclidean_distances(x[:,[i]])
        da[:,:,i] = euclidean_distances(xa)
        ds[:,:,i] = euclidean_distances(x)
        
    dsx = np.sqrt(np.square(ds).sum(2))
    dax = np.sqrt(np.square(da).sum(2))
    # '''

    '''
    dsx = euclidean_distances(x,x)
    dax = euclidean_distances(xa,xa)
    '''
    dst = dsx[trils]
    dat = dax[trila]


    # [4] - grab the minimum and maximum for lengthscale, signal variance, and noise variance.
    # =========================================================================================
    # '''
    rLo = [ds[:,:,i][trils] for i in range(ds.shape[2])]
    rLo = np.mean([g[g>0].min() for g in rLo])
    # '''
    '''
    rLo = dsx[dsx>0].min()
    '''
    rHi = da.max()*3

    sLo = vt[vt>0].min()
    sHi = vt.max()

    nLo = 1e-9
    nHi = 1e-3
    if not low_noise:
        nHi = densityGrabber(gIn = vt)
    rBounds = [rLo,rHi]
    sBounds = [sLo,sHi]
    nBounds = [nLo,nHi]

    return trils, trila, dsx,v, dax, dst, vt, dat, sBounds,nBounds,rBounds,rBounds


def matern_52(x1, x2, g = None, r = 1, s = 1, ARD = False):

    if g is None:

        if ARD == True:

            g = 0
            for i in range(x1.shape[1]):

                x1i,x2i = x1[:,i], x2[:,i]
                g       = g + np.square(euclidean_distances(x1i,x2i) / r[i])
            g = np.sqrt(g)

        if ARD == False:

            g = euclidean_distances(x1,x2) / float(np.ravel(r)[0])
    g = np.sqrt(5) * g
    K = np.multiply(1 + g + np.square(g) / 3, np.exp(-1.0 * g))

    return K

def gaussian(x1,x2, g = None, r=1, s=1, ARD = False):

    if g is None:

        if ARD == True:

            g = 0
            for i in range(x1.shape[1]):

                x1i,x2i = x1[:,i], x2[:,i]
                g       = g + np.square(euclidean_distances(x1i,x2i) / r[i])

        if ARD == False:
            g = np.square(euclidean_distances(x1,x2) / float(np.ravel(r)[0]))

    K = np.exp(-0.5 * g)

    return K

def euclidean(x1,x2,g=None,r=1,s=1,ARD=False):
    # just a basic euclidean kernel.
    if g is None:
        g = euclidean_distances(x1,x2)
    g[g==0] = 1
    K = np.divide(1,np.square(g))
    return K


def skewGaussian(x1,x2, r=1, s=1, k=0, ARD=False):
    r = np.ravel(r).tolist()
    k = np.ravel(k).tolist()
    if ARD==False:
        r=r[0]
        k=k[0]
    # [REFERENCE] - Quanto's answer in: https://math.stackexchange.com/questions/3334869/formula-for-skewed-bell-curve
    # [FORMULA]   - exp(-0.5 * (g/r)^2) * [1 - kg/(2r^2)]
    # if g is None:

    if ARD: # we are just going to treat every distance as g, so no need to worry about partial derivatives.
        g1 = 0
        g2 = 0
        for i in range(x1.shape[1]):
            d  = euclidean_distances(x1[:,i],x2[:,i])
            g1 = g1 + np.square(d/r[i])
            g2 = g2 + np.square(k[i] * d/(2*np.square(r[i])))
        g1 = np.sqrt(g1)
        g2 = 1-np.sqrt(g2)
    else:
        g1 = euclidean_distances(x1,x2)/r
        g2 = 1-k*g1/(2*r)
    g1 = np.exp(-0.5 * np.square(g1))
    # K = s * np.clip(np.multiply(g1, g2),0,None)
    K =  np.multiply(g1, g2)
    return K



def myKernel(x1,x2, g = None, r =1, s = 1, k=0, ARD = False, kernel = ['matern','gaussian','skew','euclidean'][0]):
    r = np.ravel(r).tolist()
    ARD = ARD and len(r) == x1.shape[1]


    # [CONFIRMED] - See: syandana_eDist_timeTester.py for verification that this can achieve greater calculation speed for ARD.
    if not 'euclidean' in kernel and not (x1 is None or x2 is None):
        if ARD == False and len(r) == 1:
            x1 = x1/r[0]
            x2 = x2/r[0]
        else:# elif len(r)==x1.shape[1]:#(ARD == False and len(r) == x1.shape[1]) or ARD==True:
            x1 = np.divide(x1,r)
            x2 = np.divide(x2,r)
            
        ARD = False
        r=1

    if 'mat' in kernel:
        K = matern_52(x1,x2, g=g, r=r, s=1, ARD=ARD)
    if 'gau' in kernel:
        K = gaussian(x1,x2, g=g, r=r, s=1, ARD=ARD)
    if 'euc' in kernel:
        K = euclidean(x1,x2, g=g, r=1, s=1, ARD=ARD)
    if 'skew' in kernel:
        K = skewGaussian(x1,x2, r=r, s=1, k=k, ARD=ARD)

    if s <= 0:
        K = 1 - K
    return np.abs(s) * K