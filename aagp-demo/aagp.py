#%%
# [1] - package import
# =========================
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
# from matplotlib import cm


# [2] - self-made package import
# ===============================
from algorithms import(
    euclidean_distances,
    get_kernel_data_bounds as CKDG,
    myKernel,
)

from optimization import(
    class_execute_optimizer as CEO
)

from visualization import(
    dawn_cmap
)


def euclidean_correction(xRef, xTarget, dRef=None, dTarget=None):

    '''
    # Euclidean Correction
    Generates a coefficient to ensure that `xRef` and `xTarget` are on the same euclidean scale.
    '''
    ff = lambda xin: euclidean_distances(xin)[np.tril_indices(xin.shape[0],k=-1)]

    # [1] - get the reference frame
    if dRef is None:
        uSig= ff(xRef).max()
    else:
        uSig = dRef

    # [2] - get the target frame
    if dTarget is None:
        xSig= ff(xTarget).max()
    else:
        xSig = dTarget

    if uSig==0:
        uSig=1
    if xSig==0:
        xSig=1


    # [3] - make the transforming values.
    coeff= np.divide(xSig,uSig)

    return coeff

def calculate_ALE(x, y,):
    '''
    # Analytical Likelihood calculation (ALE)
    This function calculates the optimal lengthscale given data-driven parameters for signal and noise variance using geostatistics.
    '''

    # [1] - get the pairwise distances
    # ==================================
    d = euclidean_distances(x,x)
    v = euclidean_distances(y,y)
    idx = np.where(v>0)

    dt = d[idx]
    vt = v[idx]

    idx = np.argsort(dt)
    vt  = vt[idx]
    dt  = dt[idx]

    # [2] - set the data-driven values
    # ====================================
    valid = lambda xIn: np.mean(xIn[np.isfinite(xIn)])
    n     = vt.min()
    vg    = vt-n
    s     =  np.mean(vg)/2 + np.percentile(vg,50)/2

    # [3] - calculate the optimal lengthscale
    # ========================================
    c = -1
    G = 2 * s / (2 * (n + s) - vg) + 2j * np.pi * c
    R = np.divide(dt,G)
    R = np.sqrt(np.square(R.real) + np.square(R.imag))
    R = valid(R)
    
    return s,n,R





class AdjacencyVectorizer:

    def __init__(self, x, y, xa=None, train=True, plot_ALE=False):

        self.x     = x
        self.y     = y
        self.xa    = xa if not xa is None else x
        self.train = train
        self.prepare_training_data( plot_ALE=plot_ALE )
    
    def create_kernel_matrix(self, x1,x2,g=None, r = 1):
        # print(g)
        # if g is None:
        #     r = np.ravel([float(np.abs(r))] * x1.shape[1])
        #     x1= np.divide(x1,r)
        #     x2= np.divide(x2,r)
        #     g = euclidean_distances(x1,x2)
        
        # K = np.exp(-0.5 * np.square(g))
        K = myKernel(x1,x2,g=g,r=r,s=1,ARD=False,kernel='gaussian')
        # print(K)
        return K

    
    def prepare_training_data(self, plot_ALE=False):

        # this function will prepare the training data for speed optimization when running training.
        x,y,xa = self.x, self.y, self.xa
        d = euclidean_distances(x,x)
        da= euclidean_distances(xa,xa)
        v = euclidean_distances(y,y)

        tril_s = np.tril_indices(x.shape[0],k=-1)
        tril_a = np.tril_indices(xa.shape[0],k=-1)

        dt, vt = [g[tril_s] for g in [d,v]]
        dat    = da[tril_a]


        sIdx = []
        for i in range(x.shape[0]):
            xi = x[i,:]
            delta = np.ravel(np.abs(xi-xa).sum(1))
            idx = np.argmin(delta)
            if delta[idx] == 0:
                sIdx.append(i)
        uIdx = [g for g in range(xa.shape[0]) if not g in sIdx]


        self.d = d
        self.da= da
        self.v = v

        self.dt  = dt
        self.dat = dat
        self.vt  = vt

        self.tril_s = tril_s
        self.tril_a = tril_a

        # [1] - grab the bounds for the training.
        # =======================================================================
        # trils, trila, dsx,v, dax, dst, vt, dat, sBounds,nBounds,rBounds,rBounds = CKDG(x,y,xa=xa)
        idxs, idxa, d, v, da, dt, vt, dat, sBounds, nBounds, rBounds, raBounds  = CKDG(x,y,xa=xa)
        self.rBounds = rBounds
        self.dt = dt
        self.dat= dat
        
        # [1.1] - get a correction coefficient and function (REQUIRED)
        # =======================================================================
        fc = euclidean_correction(xa,xTarget=xa,dTarget=dat.max())
        self.Z_X = lambda xIn: xIn * fc
        zxa = self.Z_X(xa)
        zxs = self.Z_X(x)
        self.zxs = zxs
        self.zxa = zxa


        # [2] - calculate the by-dimension training distances.
        # =======================================================================
        VECS = []
        VECSA= []
        for i in range(x.shape[1]):

            da_ = euclidean_distances(zxa[:,i].reshape(-1,1), zxa[:,i].reshape(-1,1))
            VECSA.append(da_)
            # d_ = euclidean_distances(zxs[:,i].reshape(-1,1), zxa[:,i].reshape(-1,1))
            VECS.append(da_[sIdx,:])


        
        self.VECS  = VECS
        self.VECSA = VECSA

        # [3] - Utilize analytical likelihood estimation (ALE)
        # =======================================================================
        rho_s, rho_n, rho_r    = calculate_ALE(x,y)
            
        self.rho_s = rho_s
        self.rho_n = rho_n
        self.rho_r = rho_r
    
    def vectorize(self, xIn,  r=1, mode=['train','test'][0], use_all = False):

        if mode=='train':
            V = self.VECS
            if use_all:
                V = self.VECSA
            
            J = np.column_stack([self.create_kernel_matrix(None,None, g=f/float(r), r=1).mean(1).reshape(-1,1) for f in V])

        else:
            zIn = self.Z_X(xIn)
            J = np.column_stack([self.create_kernel_matrix(zIn[:,i].reshape(-1,1), self.zxa[:,i].reshape(-1,1), r=r).mean(1).reshape(-1,1) for i in range(xIn.shape[1])])
        
        return J
    

    def loss_function(self, xIn, hypersIn, mode=['train','test'][0]):

        r = np.ravel(hypersIn).tolist()[0]
        j = self.vectorize(None, r=r, mode='train', use_all = not 'train' in mode)
        
        
        if mode=='test':
            # varxi = self.dat.max()/euclidean_distances(j,j).max()
            varxi = euclidean_correction(xRef = j, xTarget = self.xa, dTarget = self.dat.max())
            SV = lambda xIn: self.vectorize(xIn, r=r, mode='test') * varxi
            return SV
        
        else:
            # varxi = self.dt.max()/euclidean_distances(j,j).max()
            varxi = euclidean_correction(xRef = j, xTarget = self.x, dTarget = self.dt.max())
            J = j * varxi
            K = self.rho_s * (1 - self.create_kernel_matrix(J,J,r=self.rho_r)) + self.rho_n
            K = K[self.tril_s]

            m = np.log(K) + np.divide(self.vt, 2.0 * K)
            m = m[np.isfinite(m)].sum()
            return m
    
    def fit(self):

        optLo     = [self.dt[self.dt>0].min()]
        optHi     = [self.dat.max()]
        optBounds = [optLo,optHi]
        optFunc   = lambda hypersIn: self.loss_function(None, hypersIn, mode='train')
        retFunc   = lambda hypersIn: self.loss_function(None, hypersIn, mode='test')
        if self.train:
            # xOpt = lhs_optimization(optFunc, optBounds, n_points=30)
            _, xOpt,yOpt = CEO(
                optFunc = optFunc,
                optBounds = optBounds,
                addCorners = False,
                O = 'pso',

            )
            SV = retFunc(xOpt)
        else:
            SV = retFunc(np.ravel(np.matrix(optBounds).mean(0)))
        self.SV = SV
        
        return SV
    
    def plot(self, gsize=50, dpi = 100, figsize=(4,4), ax=None):
        
        # this will plot a 2D function of the kernel at the first measured point.
        x  = self.x
        xs = self.x[0,:].reshape(1,-1)
        rho_r = self.rho_r
        SV    = self.SV

        xa = self.xa

        bLo = np.ravel(xa.min(0))
        bHi = np.ravel(xa.max(0))
        g1,g2 = [np.linspace(bLo[g], bHi[g], gsize).reshape(-1,1) for g in range(2)]
        gg1,gg2 = np.meshgrid(g1,g2)

        ggx   = np.column_stack((gg1.reshape(-1,1), gg2.reshape(-1,1)))
        K     = self.rho_s * self.create_kernel_matrix(SV(ggx),SV(xs), r=rho_r).sum(1).reshape(-1,gsize)

        if ax is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax  = fig.add_subplot(1,1,1)
        ax.contourf(gg1,gg2, K, cmap=dawn_cmap(), levels=10, alpha=0.75)
        ax.scatter(xa[:,0],xa[:,1], marker='2', linewidth=0.75, facecolor='orangered', zorder=1, alpha=0.85)
        ax.scatter(x[1:,0], x[1:,1], marker='s', facecolor='cyan',edgecolor='blue',linewidth=1, zorder=2)
        ax.scatter(xs[0,0], xs[0,1], marker='*', facecolor='cyan',edgecolor='blue',linewidth=1, zorder=3,s=500)
        ax.axis('square')
        ax.set_title('Adjacency-Adaptive Covariance (at Starred Point)')















if __name__ == '__main__':
    from equations import get_function
    xa, ya, xe, ye, f, b = get_function(
        fName = 'z.8.2',
        n = 300,e=1000,lhs_iters=10,
    )

    m = 16
    np.random.seed(777)
    idx = np.random.choice(xa.shape[0], m, replace=False)
    x = xa[idx,:]
    y = ya[idx,:]
    AV = AdjacencyVectorizer(x,y,xa=xa)
    AV.fit()
    AV.plot()





    '''
    def calculate_ALE_(x,y,):
        # [INTRO] - this procedure is as follows: (1) find s,n,r using basic estimates, (2) find s,n,r using analytical estimates, (3), take average
        valid = lambda xIn: np.mean(xIn[np.isfinite(xIn)])
        d = np.ravel(euclidean_distances(x,x))
        v = np.ravel(euclidean_distances(y,y))
        idx = np.where(v>0)
        d = d[idx]
        v = v[idx]
        # idx = np.tril_indices(d.shape[0],k=-1)
        idx = np.argsort(d)
        d = d[idx]
        v = v[idx]
        idx = np.where(v>0)
        d = d[idx]
        v = v[idx]
        # d = d[idx]
        # v = v[idx]
        n = v.min() # <--- assuming NO interspatial variance between lags (variogram cloud)
        # print('variogram n: ',n)
        v = v-n
        valid_idx = np.where(v>0)[0]
        v = v[valid_idx]
        d = d[valid_idx]
        n = v[np.argsort(d)][:5].mean()
        # print('variogram n: ',n)
        dv = np.column_stack([np.ravel(g).reshape(-1,1) for g in [d,v]])
        gkde = gaussian_kde(dv.T)(dv.T)
        gkde /= gkde.max()

        idx_v= np.where(gkde>=0.25)[0]
        dv2  = dv[idx_v,:]
        idx_d= np.where(dv2[:,0]/dv[np.argmax(gkde),0]<=1)[0]
        r,s  = np.ravel(dv2[idx_d,:].mean(0))
        print('variogram n: ',n)
        print('variogram s: ',s)
        print('variogram r: ',r)
        return s,n,r
    '''


# %%
