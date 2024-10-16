# [0] - package imports
# =====================================
import numpy as np


# [1] - self-made package imports
# =====================================
from algorithms import (
    CL,
    GET_LAPLACIAN_BOUNDS,
    get_kernel_data_bounds,
    lhs_select,
    SLOGDET,
    SLV,
    myKernel,
    # euclidean_distances,
    # trainTest_binSplitter,
)

from aagp import(
    AdjacencyVectorizer
)

from auxiliaries import (
    VS
)

from optimization import(
    class_execute_optimizer as CEO,
    # SPOPT_EXE
)

class kernelModel:

    def __init__(
            self,
            x,y,xa = None,
            training_mode = ['gp','aagp','lod','lrk'][0],
            kernel = ['gaussian','matern'][1],
            ARD    = False,
    ):
        self.x = x
        self.y = y
        self.xa= x.copy() if xa is None else xa
        self.training_mode = training_mode

        if ARD:
            print('Warning. Automatic-Relevance is not supported in this release. Switching to Equal-Relevance.')
            ARD = False

        self.ARD           = ARD
        self.kernel        = kernel
        self.load_data()
    
    def load_data(self):
        '''
        # Load data
        This function calculates the data driven upper and lower bounds for training.
        '''
        idxs, idxa, d, v, da, dt, vt, dat, sBounds, nBounds, rBounds, raBounds = get_kernel_data_bounds(
            x = self.x,
            y = self.y,
            xa= self.xa,
        )

        self.idxs = idxs
        self.idxa = idxa
        self.d = d
        self.v = v
        self.da = da
        self.dt = dt
        self.vt = vt
        self.dat = dat
        self.sBounds = sBounds
        self.nBounds = nBounds
        self.rBounds = rBounds
        self.raBounds = raBounds
        self.I = np.eye(self.x.shape[0])
        self.EYE = self.I.copy()
    
    def create_kernel_matrix(self, x1,x2, g=None, r=1, s=1, ARD=False):
        return myKernel(x1,x2,g=g, r=r, s=s, ARD=ARD, kernel=self.kernel)

    def fit(self):

        training_mode = self.training_mode

        if training_mode == 'gp':
            return self.train_gp()

        if training_mode == 'aagp':
            return self.train_aagp()
        
        elif training_mode == 'lod':
            return self.train_lod()
        
        elif training_mode == 'lrk':
            return self.train_lrk()
        
    
    def train_gp(self):
        '''
        # Train GP
        This function will train a kernel via the GP likelihood and return the covariance function for use.
        '''
        
        def fit_gp(x,y,hypersIn, returnLike=True):
            Hin = np.ravel(hypersIn).tolist().copy()
            s   = Hin.pop(0)
            n   = Hin.pop(0)
            r   = Hin
            if len(r) == 1:
                r = r[0]
            ckm = lambda x1,x2: self.create_kernel_matrix(x1,x2, r=r, s=s,)

            if returnLike==False:
                return ckm

            else:
                K = ckm(x,x)
                H = K + self.I * n
                m = SLOGDET(H,y)
                return m

        # [1] - get the bounds and data
        # ===================================
        sBounds = self.sBounds
        rBounds = self.rBounds
        nBounds = self.nBounds
        x       = self.x
        y       = self.y
        xa      = self.xa

        # [2] - set the bounds
        # ===================================
        idx = 0
        bLo = [sBounds[idx],nBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
        idx = 1
        bHi = [sBounds[idx],nBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
        optBounds = [bLo,bHi]
        
        # [3] - run the optimization
        # ====================================
        optFunc = lambda hypersIn: fit_gp(x,y,hypersIn, returnLike=True)
        retFunc = lambda hypersIn: fit_gp(x,y,hypersIn, returnLike=False)
        O,xOpt_,yOpt = CEO(optFunc = optFunc, optBounds=optBounds, O='pso')
        CKM  = retFunc(xOpt_)
        xOpt = np.ravel(xOpt_).tolist()

        # [4] - return the optimal covariance.
        # ====================================
        sOpt = xOpt.pop(0)
        nOpt = xOpt.pop(0)
        rOpt = xOpt.copy()
        self.sOpt = sOpt
        self.nOpt = nOpt
        self.n    = nOpt
        self.ridge= nOpt
        self.rOpt = rOpt
        self.CKM  = CKM
        return CKM

    def train_aagp(self):
        '''
        # Train AAGP
        This function will train the Adjacency-Adaptive Gaussian Process (AAGP) model using max likelihood.
        '''
        def fit_aagp(x,y,a,V_, hypersIn, returnLike = True):
            Hin = np.ravel(hypersIn).tolist().copy()
            s = Hin[0]
            n = Hin[1]
            rx= [Hin[2]]*x.shape[1]
            ra= [Hin[3]]*a.shape[1]
            rxa = rx + ra
            ckm = lambda x1,x2: self.create_kernel_matrix(x1,x2,r=rxa,s=s)
            if returnLike==False:
                ckm = lambda x1,x2: self.create_kernel_matrix(
                    np.column_stack((x1, V_(x1))),
                    np.column_stack((x2, V_(x2))),
                    r=rxa,s=s
                    )
                return ckm

            else:
                xv= np.column_stack((x,a))
                K = ckm(xv,xv)
                H = K + self.I * n
                m = SLOGDET(H,y)
                return m
            


        # [1.1] - get the bounds and data
        # ===================================
        sBounds = self.sBounds
        rBounds = self.rBounds
        nBounds = self.nBounds
        x       = self.x
        y       = self.y
        xa      = self.xa

        
        # [1.2] - fit the adjacency vectorizer
        # ====================================
        AV = AdjacencyVectorizer(
            x = x,
            y = y,
            xa = xa,
            train = True,
            plot_ALE = False,
        )
        SV = AV.fit()
        a  = SV(x)

        # [2] - set the bounds
        # ===================================
        idx = 0
        bLo = [sBounds[idx],nBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD)) + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
        idx = 1
        bHi = [sBounds[idx],nBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD)) + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
        optBounds = [bLo,bHi]
        
        
        
        # [3] - run the optimization
        # ====================================
        optFunc = lambda hypersIn: fit_aagp(x,y,a,SV,hypersIn, returnLike=True)
        retFunc = lambda hypersIn: fit_aagp(x,y,a,SV,hypersIn, returnLike=False)
        O,xOpt,yOpt = CEO(optFunc = optFunc, optBounds=optBounds, O='pso')
        CKM  = retFunc(xOpt.copy())
        xOpt = np.ravel(xOpt).tolist()
        
        # [4] - return the optimal covariance.
        # ====================================
        sOpt = xOpt.pop(0)
        nOpt = xOpt.pop(0)
        rOpt = xOpt.copy()
        self.sOpt = sOpt
        self.nOpt = nOpt
        self.n    = nOpt
        self.ridge= nOpt
        self.rOpt = rOpt
        self.CKM  = CKM
        return CKM

    def train_lod(self):
        '''
        # Train LOD
        This function will train a kernel via subsampled GCV for LOD, and return the estimator and varinace estimator of the optimal model.
        '''


        def lod_gcv(x,y,z,r=1,s=1,n=1,l=0,returnLike=True):
            '''
            # Generalized Cross-Validation for LOD
            '''
            xa = VS(x,z)
            CKM = lambda x1,x2: self.create_kernel_matrix(x1,x2, r=r, s=s, ARD=self.ARD)
            Kxs = CKM(xa,x)
            Kxx = CKM(xa,xa)
            L   = CL(Kxx, normalize=True)
            nu  = x.shape[0]/(np.square(xa.shape[0]))

            if returnLike==True:
                Kss = Kxs[0:x.shape[0],:]
                H = Kss + n * np.matrix(np.eye(x.shape[0]))
                if l > 0:
                    nu= 1/x.shape[0]
                    L = nu * l * (L[0:x.shape[0],:])[:,0:x.shape[0]] @ Kss
                    H = H + L
                HI = SLV(H)
                e = y - Kss @ HI @ y
                v = np.diag(Kss @ HI @ Kss.T)
                m = np.log(np.square(e).mean()/np.square(v.mean()))
                return m

            H   = Kxs @ Kxs.T + n * Kxx
            if l > 0:
                H = H + nu * l * Kxx @ L @ Kxx
            HI  = np.matrix(np.linalg.pinv(H))
            B   = HI @ Kxs @ y
            mu  = lambda xIn: CKM(xIn,xa) @ B
            sig = lambda xIn: np.matrix(np.abs(np.diag(CKM(xIn,xa) @ HI @ CKM(xa,xIn)))).reshape(-1,1)
            return CKM, mu, sig

        def lod_trainer(x,y,z, hypers, returnLike=True, percentage = 0.25):
            '''
            # Laplacian Optimal Design (LOD) Trainer
            This function will train LOD using Generalized cross validation and return the optimal regression model.
            '''
            HIN = np.ravel(hypers).tolist().copy()
            s = HIN.pop(0)
            n = HIN.pop(0)
            l = HIN.pop(0)
            r = HIN.copy()
            if returnLike:
                
                E = []
                nsplits = int(z.shape[0]/(z.shape[0]*percentage))
                splits  = np.array_split(range(z.shape[0]), nsplits)
                for i in range(len(splits)):
                    e = lod_gcv(x,y,z[splits[i],:], r=r, s=s,n=n,l=l,returnLike=True)
                    E.append(e)
                E = np.sqrt(np.square(E).sum())
                return E
            
            else:
                return lod_gcv(x,y,z, r=r, s=s,n=n,l=l,returnLike=False)
        
        # [1] - get the bounds and data
        # ===================================
        sBounds = self.sBounds
        rBounds = self.rBounds
        nBounds = self.nBounds
        lBounds = GET_LAPLACIAN_BOUNDS()
        x       = self.x
        y       = self.y
        xa      = self.xa

        # [2] - set the bounds
        # ===================================
        idx = 0
        bLo = [sBounds[idx],nBounds[idx],lBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
        idx = 1
        bHi = [sBounds[idx],nBounds[idx],lBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
        optBounds = [bLo,bHi]

        # [3] - perform subsampling for faster execution
        # ================================================
        sIdx = []
        for i in range(x.shape[0]):
            xi = x[i,:]
            delta = np.ravel(np.abs(xi-xa).sum(1))
            idx = np.argmin(delta)
            if delta[idx] == 0:
                sIdx.append(i)

        uIdx            = [g for g in range(xa.shape[0]) if not g in sIdx]
        z               = xa[uIdx,:]
        zu              = xa[uIdx,:]
        zIn             = z.copy()
        zOut            = z.copy()
        z,zu,zdx, zudx  = lhs_select(xa=zu, m=int(zu.shape[0]/5), maxIters=100, verbose=False)
        zIn             = z.copy()


        optFunc = lambda hypersIn: lod_trainer(x,y,zIn,hypersIn,returnLike=True)
        retFunc = lambda hypersIn: lod_trainer(x,y,zOut,hypersIn,returnLike=False)

        # [4] - run the optimizer
        # ===================================
        O,xOpt,yOpt = CEO(optFunc = optFunc, optBounds=optBounds, O='pso')
        CKM,mu,sig  = retFunc(xOpt)
        xOpt = np.ravel(xOpt).tolist()

        # [4] - return the optimal covariance.
        # ====================================
        sOpt = xOpt.pop(0)
        nOpt = xOpt.pop(0)
        rOpt = xOpt.copy()
        self.sOpt = sOpt
        self.nOpt = nOpt
        self.n    = nOpt
        self.ridge= nOpt
        self.rOpt = rOpt
        self.CKM  = CKM
        self.lapRLS_mu = mu
        self.lapRLS_sig = sig
        return CKM,mu,sig
    

    def train_lrk(self):
        '''
        # Train LRK
        This function will train a kernel via Warped RKHS and return the optimal covariance function.
        '''


        def lrk_mle(x,y,z,r=1,s=1,n=1,l=0,L = None, returnLike=True):
            '''
            # GP MLE with Warped RKHS
            '''
            if z is None:
                xa = x.copy()
            else:
                xa = VS(x,z)
            ckm_0 = lambda x1,x2: self.create_kernel_matrix(x1=x1, x2=x2, r=r, s=s,ARD=self.ARD)
            ckm   = lambda x1,x2: self.create_kernel_matrix(x1=x1, x2=x2, r=r, s=s, ARD=self.ARD)

            if l > 0:
                Ka = ckm_0(xa,xa)
                if L is None:
                    La = CL(Ka, normalize=True)
                else:
                    La = L
                nu = l * x.shape[0]/np.square(xa.shape[0])
                LB = np.matrix(np.eye(Ka.shape[0])) + nu * La @ Ka
                LB = SLV(LB,La*nu)
                ckm = lambda x1,x2: ckm_0(x1,x2) - ckm_0(x1,xa) @ LB @ ckm_0(xa,x2)
            
            if returnLike:
                K = ckm(x,x)
                H = K + self.I * n
                return SLOGDET(H, y)
            else:
                return ckm
            



        def lrk_trainer(x,y,z, hypers, returnLike=True):
            '''
            # Trainer for GP + Warped RKHS
            This function will train LRK using GP Max Likelihood and Warmed RKHS.
            '''
            HIN = np.ravel(hypers).tolist().copy()
            s = HIN.pop(0)
            n = HIN.pop(0)
            l = HIN.pop(0)
            r = HIN.copy()
            if returnLike:
                e = lrk_mle(x,y,z, r=r, s=s,n=n,l=l,returnLike=True)
                return e
            else:
                return lrk_mle(x,y,z, r=r, s=s,n=n,l=l,returnLike=False)
        
        # [1] - get the bounds and data
        # ===================================
        sBounds = self.sBounds
        rBounds = self.rBounds
        nBounds = self.nBounds
        lBounds = GET_LAPLACIAN_BOUNDS()
        x       = self.x
        y       = self.y
        xa      = self.xa

        # [2] - set the bounds
        # ===================================
        idx = 0
        bLo = [sBounds[idx],nBounds[idx],lBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
        idx = 1
        bHi = [sBounds[idx],nBounds[idx],lBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
        optBounds = [bLo,bHi]

        # [3] - perform subsampling for faster execution
        # ================================================
        sIdx = []
        for i in range(x.shape[0]):
            xi = x[i,:]
            delta = np.ravel(np.abs(xi-xa).sum(1))
            idx = np.argmin(delta)
            if delta[idx] == 0:
                sIdx.append(i)

        uIdx            = [g for g in range(xa.shape[0]) if not g in sIdx]
        z               = xa[uIdx,:]
        zu              = xa[uIdx,:]
        zIn             = x.copy()#z.copy()
        zOut            = z.copy()
        z,zu,zdx, zudx  = lhs_select(xa=zu, m=int(zu.shape[0]/5), maxIters=100, verbose=False)
        # zIn             = z.copy()


        optFunc = lambda hypersIn: lrk_trainer(x,y,None,hypersIn,returnLike=True)
        retFunc = lambda hypersIn: lrk_trainer(x,y,zOut,hypersIn,returnLike=False)

        # [4] - run the optimizer
        # ===================================
        O,xOpt,yOpt = CEO(optFunc = optFunc, optBounds=optBounds, O='pso')
        CKM  = retFunc(xOpt)
        xOpt = np.ravel(xOpt).tolist()

        # [4] - return the optimal covariance.
        # ====================================
        sOpt = xOpt.pop(0)
        nOpt = xOpt.pop(0)
        rOpt = xOpt.copy()
        self.sOpt = sOpt
        self.nOpt = nOpt
        self.n    = nOpt
        self.ridge= nOpt
        self.rOpt = rOpt
        self.CKM  = CKM
        return CKM