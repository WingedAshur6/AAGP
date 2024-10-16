# [1] - package import
# ============================
import numpy as np
from sklearn.linear_model import ARDRegression, BayesianRidge
import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning, UndefinedMetricWarning
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import xgboost
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score

# [2] - self-made packages
# ============================
from kernel_training import kernelModel
from algorithms import (
    CL,
    euclidean_distances,
    # trainTest_binSplitter,
    GBAL_acq,
    GREEDY_acq,
    # mutual_euclid,
    subsampler,
    SLV,
    VS,
    myKernel,
    trainTest_binSplitter,
    maxiMin_acquisition
)

from deepGP import (
    multiGP
)

from optimization import (
    class_execute_optimizer as CEO,
    SPOPT_EXE
)

from SETTINGS import (
    deepGP_maxIters
)

KERNEL = 'matern'










['r2', 'mae', 'mse', 'rmse', 'nrmse','mape', 'mmape', 'wmape', 'smape', 'bias', 'adjusted_r2', 'f1', 'cod', 'mbd', 'cv']
def calculate_regression_metrics(y_true, y_pred, p, metric='r2'):
    """
    Calculate various regression performance metrics based on the input parameter 'metric'.

    Parameters:
    y_true (array-like): True (actual) values.
    y_pred (array-like): Predicted values.
    metric (str): The metric to calculate. Options: 'r2', 'mae', 'mse', 'rmse', 'mape', 'mmape', 'wmape', 'bias', 'adjusted_r2',
                 'f1', 'cod', 'mbd', 'cv'.

    Returns:
    float: The calculated regression performance metric.
    """
    if metric == 'r2':
        return r2_score(y_true, y_pred)
    elif metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'mse':
        return mean_squared_error(y_true, y_pred)
    elif metric == 'rmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == 'nrmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))/(y_true.max()-y_true.min())
    elif metric == 'mape':
        absolute_percentage_errors = np.abs((y_true - y_pred) / np.abs(y_true))
        return np.mean(absolute_percentage_errors)
    elif metric == 'mmape':
        eps = 1e-10
        absolute_percentage_errors = np.abs((y_true - y_pred) / np.abs(y_true) + eps)
        return np.mean(absolute_percentage_errors)
    elif metric == 'wmape':
        wmape = np.abs((y_true - y_pred)).sum() / np.abs(y_true).sum()
        return wmape
    
    elif metric == 'smape':
        yp = np.ravel(y_pred)
        ye = np.ravel(y_true)
        error = ye - yp
        smape   = np.divide(
                            np.ravel(np.abs(error)),
                            (np.abs(yp) + np.abs(ye))/2
                            ).mean()
        return smape
        
    elif metric == 'bias':
        error   = y_true - y_pred
        overEst = np.where(error>0, np.abs(error),0)
        underEst= np.where(error<=0, np.abs(error),0)
        return np.divide(overEst-underEst,overEst+underEst).mean()

    elif metric == 'adjusted_r2':
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        return adjusted_r2
    elif metric == 'f1':
        # Placeholder values for binary classification (not a typical regression metric)
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        return f1_score(y_true_binary, y_pred_binary)
    elif metric == 'cod':
        variance_residuals = np.var(y_true - y_pred)
        variance_y = np.var(y_true)
        return 1 - (variance_residuals / variance_y)
    elif metric == 'mbd':
        return np.mean(y_true - y_pred)
    elif metric == 'cv':
        std_residuals = np.std(y_true - y_pred)
        mean_y = np.mean(y_true)
        return (std_residuals / mean_y) * 100

    else:
        raise ValueError("[ %s CHOSEN ] - Invalid metric. Choose from 'r2', 'mae', 'mse', 'rmse', 'mape', 'mmape', 'wmape', 'bias', 'adjusted_r2', 'f1', 'cod', 'mbd', 'cv'."%(metric))









def fit_gp(x,y,xa=None,kernel = KERNEL, model_name = ['gp','slrgp','aagp','lrk'][0],acquisition=[None,'gbal','mm'][0]):

    

    def ALC_MSE_acquisition(sig, x, xaIn, xIn, do_fast = True, acq = ['alc','mmse','imse'][0]):
        # we will do ALC assuming xaIn is < xa >
        ranger = range(xIn.shape[0])
        scores = []
        s  = float(CKM(x[[0],:],x[[0],:]))

        ALC = 'alc' in acq
        IMSE= 'imse' in acq
        MMSE= 'mmse' in acq

        scores = []
        D      = x
        X      = xa
        Z      = xIn
        if do_fast==True:
            Kxd = CKM(X,D)
            Kxx = CKM(X,X)
            Kxz = CKM(X,Z)

            H0  = Kxd @ Kxd.T + n @ Kxx
            H0i = SLV(H0)
            v0  = np.diag(Kxx @ H0i @ Kxx.T).reshape(-1,1)
            for i in range(Z.shape[0]):
                idx = np.where(np.ravel((Z[i,:]-X).sum(1)==0))[0]
                jdx = [g for g in range(X.shape[0]) if not g in idx]
                # using sherman-woodbury matrix update.
                # ADD: [H + q.T * q]^-1 = Hi - [Hi*q.T*q*HI]/[1 + q*HI*q.T]
                # DEL: [H - q.T * q]^-1 = Hi + [Hi*q.T*q*HI]/[1 - q*HI*q.T]
                q = Kxz[:,i]
                H1i = H0i + H0i @ q @ q.T @ H0i/float(s- q.T @ H0i @ q)
                v1  = np.diag(Kxx * H1i * Kxx.T).reshape(-1,1)

                vRed= (v0-v1)[jdx,:].sum()
                scores.append(vRed)

        else:
            I2 = np.matrix(np.eye(D.shape[0]+1))
            v0 = sig(X)
            Kpp= CKM(X,X)
            for i in range(Z.shape[0]):
                Zi    = Z[i,:]
                idx   = np.where(np.ravel((Zi-X).sum(1)) == 0)[0]
                jdx = [g for g in range(X.shape[0]) if not g in idx]

                D2 = VS(D,Zi)
                H2 = CKM(D2,D2) + n * I2
                Kps= CKM(X,D2)

                v2 = np.diag(Kpp - Kps @ SLV(H2,Kps.T)+n).reshape(-1,1)

                if ALC or IMSE:
                    vRed = (v0[jdx,:]-v2[jdx,:]).sum()
                else:
                    vRed = v0[jdx,:].max()-v2[jdx,:].max()
                scores.append(vRed)

        return np.matrix(scores).reshape(-1,1)


    def MI_acquisition(sig, x, xaIn, xIn, do_fast=True,):

        # this is mutual information.
        s = float(CKM(x[[0],:],x[[0],:]))
        scores = []
        if do_fast==True:
            Kxs = CKM(xaIn,x)
            Kxr = CKM(xaIn,xaIn)
            Kxp = CKM(xaIn, xIn)
            Kxx = CKM(xaIn,xaIn)

            H_s = Kxs @ Kxs.T + n * Kxx
            H_r = Kxr @ Kxr.T + n * Kxx

            H_si= SLV(H_s)
            H_ri= SLV(H_r)

            for i in range(xIn.shape[0]):
                H_ri2 = np.matrix(np.copy(H_ri))
                Kxp_i = Kxp[:,i]
                idx   = np.where(np.ravel((xIn[i,:]-xaIn).sum(1)) == 0)[0]
                if len(idx) > 0:

                    # using sherman-woodbury matrix update.
                    # ADD: [H + q.T * q]^-1 = Hi - [Hi*q.T*q*HI]/[1 + q*HI*q.T]
                    # DEL: [H - q.T * q]^-1 = Hi + [Hi*q.T*q*HI]/[1 - q*HI*q.T]

                    for j in idx:
                        Kxp_j = Kxx[:,j]
                        H_ri2 = H_ri + (H_ri2 @ Kxp_j @ Kxp_j.T @ H_ri2)/(s + Kxp_j.T @ H_ri2 @ Kxp_j)

                minf = float(Kxp_i.T @ H_si @ Kxp_i)/float(Kxp_i.T @ H_ri2 @ Kxp_i)
                scores.append(minf)
        return np.matrix(scores).reshape(-1,1)


    def train_SLRGP(
                            x,
                            y,
                            xa,
                            mu0,
                            sig0,
                            H0,
                            HI0,
                            ckm,
                            clip_variance = False
                            ):
        # [1] - predict the outputs at ALL locations
        ya = mu0(xa)

        # [2] - calculate the Laplacian
        Ly = CL(np.square(euclidean_distances(ya,ya)), normalize=False)
        Lx = CL(np.square(euclidean_distances(xa,xa)), normalize=False)
        L  = np.divide(Ly,Lx) * n
        L[~np.isfinite(L)] = 0

        # [3] - iterate through the points.
        v0 = sig0(xa)
        if clip_variance:
            v0 = np.clip(v0,0,None)
        Kxs  = ckm(xa,x)
        Kxx  = ckm(xa,xa)
        reds = []

        lambdas = [10**g for g in [-6,-5,-4,-2,-2,-1,0,1,2,3]]
        for i in range(len(lambdas)):

            l  = lambdas[i]
            H1 = H0 + l * HI0 @ Kxs.T @ L @ Kxs

            v1 = np.diag(Kxx - Kxs @ SLV(H1, Kxs.T)).reshape(-1,1)
            idx= np.argmax(np.ravel(v1))

            x2 = xa[idx,:].reshape(1,-1)
            if i == 0:
                xstatic = x2
            if i > 0 and np.abs(x2-xstatic).sum() == 0:
                continue

            x2 = np.vstack((x,x2))
            K2 = ckm(x2,x2)
            Kx2= ckm(xa,x2)

            H2 = K2 + np.eye(K2.shape[0])*n
            v2 = np.diag(Kxx - Kx2 @ SLV(H2, Kx2.T)).reshape(-1,1) + n

            if clip_variance:
                v2 = np.clip(v2,0,None)

            v2 = np.delete(v2,idx,axis=0)
            v3 = np.delete(v0,idx,axis=0)
            vred = (v3-v2).mean()
            reds.append(float(vred))
        idx = np.ravel(np.argmax(reds))[0]

        lOpt = lambdas[idx]
        return L * lOpt

    def SLRGP_acquisition(
                            x,
                            y,
                            xa,
                            ckm,
                            L,
                            H0,
                            HI0,
                            xIn,
                            l=1e-9,
                            clip_variance = True
                            ):

        # [1] - create the first hessian
        Kxs = ckm(xa,x)
        Kps = ckm(xIn,x)
        Kpp = ckm(xIn,xIn)
        H1  = H0 + HI0 @ Kxs.T @ L @ Kxs

        # [2] - calculate the variance over the set of points.
        v1 = np.diag(Kpp - Kps @ SLV(H1, Kps.T)).reshape(-1,1) + n

        # [3] - iterate through all points.
        scores = xIn[:,0].reshape(-1,1)*0
        for i in range(xIn.shape[0]):
            x2 = np.vstack((x,xIn[i,:].reshape(1,-1)))
            Kss2 = ckm(x2,x2)
            Kps2 = ckm(xIn,x2)
            Kxs2 = ckm(xa,x2)
            H2  = Kss2 + np.eye(Kss2.shape[0]) * n
            H2  = H2 + SLV(H2, Kxs2.T @ L @ Kxs2)

            v2 = np.diag(Kpp - Kps2 @ SLV(H2, Kps2.T) + n).reshape(-1,1)

            grab = np.column_stack((v1,v2))
            if clip_variance:
                grab = np.clip(grab,0,None)

            red = np.delete(grab, i, axis=0)
            red = np.mean(-red[:,1] + red[:,0])
            scores[i,:] = float(red)
        return scores



    # [1] - Train the GP
    # =======================
    tmode = 'gp'
    if 'aagp' in model_name:
        tmode = 'aagp'
    if 'lrk' in model_name:
        tmode = 'lrk'
    K = kernelModel(
        x = x,
        y = y,
        xa= xa,
        training_mode = tmode,
        kernel = kernel
    )
    CKM = K.fit()
    n   = K.nOpt


    # [2] - fit the model
    # ========================
    K = CKM(x,x)
    H = K + n * np.matrix(np.eye(x.shape[0]))
    HI= SLV(H)
    B = HI * y
    mu= lambda xIn: CKM(xIn,x) @ B
    sig=lambda xIn: np.matrix(np.diag(CKM(xIn,xIn) - CKM(xIn,x) @ HI @ CKM(x,xIn))).reshape(-1,1) + n


    # [3] - get the acquisition
    # ============================
    acq = acquisition
    if not acq is None:
        if True in [g in acq for g in ['alc','mmsee','imse']] and not 'slrgp' in acq:
            sig0 = sig
            sig=None
            sig  = lambda xIn: ALC_MSE_acquisition(sig0, x, xa, xIn, do_fast=False, acq=acq)

        if True in [g in acq for g in ['minf','minfo','mutual']]:
            sig0 = sig
            sig = None
            sig = lambda xIn: MI_acquisition(sig, x, xa, xIn, do_fast=True)

        if True in [g in acq for g in ['gbal']]:
            sig = lambda xIn: GBAL_acq(x, xIn)

        if True in [g in acq for g in ['igs','gsy']]:
            sig = lambda xIn: GREEDY_acq(x, mu, xIn, improveGreed = 'igs' in acq)
        if True in [g in model_name for g in ['slrgp']] or 'alc' in model_name or 'alc' in acquisition:
            '''[ NOTE ] - L already has the parameters applied to it!'''
            clipper = True
            L = train_SLRGP(x,y,xa,mu,sig,H,HI, CKM, clip_variance = clipper)
            sig = lambda xIn: SLRGP_acquisition(x,y,xa,CKM,L,H,HI,xIn, l=1, clip_variance=clipper)

    return mu,sig

def fit_lod(x,y,xa=None,kernel = KERNEL, model_name = ['lod'][0],acquisition=[None,'gbal','mm'][0]):

    # [1] - Train the GP
    # =======================
    K = kernelModel(
        x = x,
        y = y,
        xa= xa,
        training_mode = 'lod',
        kernel = kernel
    )
    CKM,mu,sig = K.fit()

    # [2] - augment the acquisition if applicable
    # ============================================
    acq = acquisition
    if not acq is None:
        if True in [g in acq for g in ['gbal']]:
            sig = lambda xIn: GBAL_acq(x, xIn)

        if True in [g in acq for g in ['igs','gsy']]:
            sig = lambda xIn: GREEDY_acq(x, mu, xIn, improveGreed = 'igs' in acq)

    return mu,sig
    

def fit_loess_regressor(x,y, train=True):

    # [INTRO] - this will fit the LOESS regression model

    def CRV(xIn, order=1):
        return PolynomialFeatures(order).fit_transform(xIn)

    def predictor(D,x, y, xIn, r=1,n=1e-3):
        K = myKernel(x,xIn, r=r, s=1, ARD=False, kernel='gaussian')
        I = np.eye(D.shape[1])
        yOut = (xIn.sum(1)*0).reshape(-1,1)
        DIn = CRV(xIn)
        for i in range(xIn.shape[0]):
            J = np.diag(np.ravel(K[:,i]))
            H = D.T @ J @ D + n * I
            HI= SLV(H)
            B = HI @ D.T @ J @ y
            yOut[i,:] = DIn[i,:] @ B
        return yOut

    def train_loess(D,x,y, hypersIn, splits):
        hin = np.ravel(hypersIn).tolist().copy()
        r=hin[0]
        n=hin[1]
        mse = []
        for i in range(len(splits)):
            rdx = splits[i]
            tdx = [g for g in range(x.shape[0]) if not g in rdx]
            Dt  = D[tdx,:]
            xt  = x[tdx,:]
            yt  = y[tdx,:]
            xr  = x[rdx,:]
            yr  = y[rdx,:]
            yp  = predictor(Dt,xt,yt, xr, r=r, n=n)
            error = np.square(yp-yr).mean()**0.5
            mse.append(error)
        return np.mean(mse)
    d           = euclidean_distances(x)
    dt          = d[d>0]
    D           = CRV(x)
    # splits      = trainTest_binSplitter(range(x.shape[0]), n=-1, x=x)
    
    ''' Good results, takes long time '''
    # splits      = np.array_split(range(x.shape[0]), int(x.shape[0]/2))

    ''' Is faster, results are ???? '''

    # original settings (loess was not using these anyway because it wasing being trained)
    # splits      = np.array_split(range(x.shape[0]), 4)
    # optBounds   = [[dt.min(),0],[dt.max()*3, 10]]

    # techno_loess_again
    splits      = np.array_split(range(x.shape[0]), 2)
    optBounds   = [[np.percentile(dt,10),1e-9],[dt.max()*3, 10]]

    # techno_loess_again2
    splits      = np.array_split(range(x.shape[0]), 2)
    optBounds   = [[dt[dt>0].min(),1e-9],[dt.max()*3, 10]]


    optProblem  = lambda hypersIn: train_loess(D,x,y, hypersIn, splits)
    xOpt        = [np.percentile(dt,5), 1e-3]
    OPTIMIZER   = None
    if train==True:
        OPTIMIZER, xOpt, yOpt = CEO(optFunc=optProblem, optBounds=optBounds, O='pso',ns=10, ni=3)
    rOpt,nOpt = np.ravel(xOpt).tolist()
    mu        = lambda xIn: predictor(D,x,y,xIn, r=rOpt,n=nOpt)
    return mu


def fit_localRidge(x,y, xa=None, train=True, use_cv=True, maxiter = 4, model_name='loess', acquisition=[None,'gbal','mm'][0]):


    M = AUTOLOESS2(x,y,train=not '*' in model_name, ard_regression='ard' in model_name)
    try:
        OPTIMIZER = M.OPTIMIZER
    except:
        OPTIMIZER = None
    mu = M.mu
    sig = M.sig

    
    acq = acquisition
    if not acq is None:
        if True in [g in acq for g in ['gbal']]:
            sig = lambda xIn: GBAL_acq(x, xIn)

        if True in [g in acq for g in ['igs','gsy']]:
            sig = lambda xIn: GREEDY_acq(x, mu, xIn, improveGreed = 'igs' in acq)

    return mu,sig



class AUTOLOESS2:
    def __init__(
            self,
            x,
            y,
            n_neighbors = -1,
            verbose=False,
            train = False,
            ard_regression = False,
    ):
        self.x = x
        self.y = y
        if n_neighbors is None:
            n_neighbors = int(np.sqrt(x.shape[0])*np.abs(np.log(np.log(x.shape[0]))))
        else:
            if n_neighbors < 2:
                n_neighbors = int(np.sqrt(x.shape[0])*np.abs(np.log(np.log(x.shape[0]))))
        self.ard_regression = ard_regression
        self.n_neighbors = n_neighbors
        self.verbose=verbose
        self.mu = lambda xIn: self.predict(xIn)[:,0].reshape(-1,1)
        self.sig = lambda xIn: self.predict(xIn)[:,1].reshape(-1,1)
        if train:
            self.train()

    def train(self):
        def cv(x,y,ranger,idx,n_neighbors):
            
            mse = []
            for rdx in idx:
                tdx = [g for g in ranger if not g in rdx]

                xr  = x[rdx,:].reshape(-1,x.shape[1])
                yr  = y[rdx,:].reshape(-1,1)

                xt  = x[tdx,:].reshape(-1,x.shape[1])
                yt  = y[tdx,:].reshape(-1,1)

                # get he predictions of the data held-out.
                yp  = self.predict(xr,x=xt,y=yt, n_neighbors=n_neighbors)[:,0]
                err = np.square(np.ravel(yr) - np.ravel(yp)).mean()
                mse.append(err)
            return np.mean(mse)
        


        # this function will train the model using several holdout sets.
        x = self.x
        y = self.y
        ranger = range(x.shape[0])
        splits = np.array_split(ranger, int(x.shape[0]))

        neighbor_groups = [1] + [2,] + [int(g) for g in [ np.log(x.shape[0]), np.sqrt(x.shape[0]), np.log(x.shape[0])*np.sqrt(x.shape[0]), x.shape[0] ]]
        # neighbor_groups = [2,] + [int(g) for g in [x.shape[0] * f for f in [0.25,0.5,0.75,0.95]]]
        neighbor_groups = [g for g in neighbor_groups if g >= 1]
        scores = [cv(x,y, ranger, splits, g) for g in neighbor_groups]
        nopt = neighbor_groups[np.argmin(scores)]


        self.mu = lambda xIn: self.predict(xIn, n_neighbors=nopt)[:,0].reshape(-1,1)
        self.sig = lambda xIn: self.predict(xIn, n_neighbors=nopt)[:,1].reshape(-1,1)



    def predict(self,xp, x=None,y=None, n_neighbors = None):

        if x is None or y is None:
            x = self.x
            y = self.y
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            

        # create the distance matrix
        d = euclidean_distances(xp,x)

        # create the output vector
        yout = np.zeros((xp.shape[0],2))

        # iterate
        # for i in TQDM(range(xp.shape[0]),disable = not self.verbose, desc='Fitting Local Bayesian Ridge'):
        for i in range(xp.shape[0]):
            # grab the nearest neighbors
            idx = np.argsort(np.ravel(d[i,:]))[0:n_neighbors]

            xn = x[idx,:]
            yn = np.ravel(y[idx,:])
            MM = StandardScaler()
            MM.fit(xn)
            xn = MM.transform(xn)
            # fit the model
            if self.ard_regression:
                # M = ARDRegression(fit_intercept=True, n_iter=300)
                M = ARDRegression(fit_intercept=True, n_iter=300, lambda_2=100)
            else:
                # M = BayesianRidge(fit_intercept=True, n_iter=300)
                M = BayesianRidge(fit_intercept=True, n_iter=300, lambda_2=100)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=DataConversionWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings("ignore", category=UserWarning)

                try:
                    M.fit(xn,yn.reshape(-1,1))
                except:
                    M.fit(xn,yn)

            # get the response surface estimate
            xi = MM.transform(xp[i,:].reshape(-1,x.shape[1]))
            yhat = M.predict(xi)

            # get the hessian
            H = M.sigma_
            HI= SLV(H)

            # get the variance
            try:
                vhat = float(xi @ HI @ xi.T)
            except:
                print(M.coef_)
                H = M.sigma_
                H = np.ravel(np.ravel(H).tolist() + [H[-1,-1]]*(xi.shape[1]-H.shape[1])).reshape(1,-1)
                H = H.T @ H
                HI= SLV(H)
                vhat = float(xi @ HI @ xi.T)


            # cast
            yout[i,0] = float(yhat)
            yout[i,1] = float(vhat)
        # yout[:,1] = v0 - yout[:,1]
        return yout






def fit_xgb_regressor(
                x,
                y,
                train   = True,
                maxiter = 50,
                holdout = False,
                nsplits = -1,
                name    = 'xgb',
                ):
    # this function will train an xgboost model and optimize the following parameters:
    # [1] - n_estimators
    # [2] - max_depth
    # [3] - colsample_bynode
    # [4] - colsample_bytree
    # [5] - colsample_bylevel
    # [6] - lambda
    # [7] - alpha
    # as we can see it is a very high dimensional feature space.. good luck!

    def trainer(x, y, ranger, idx, hypers,return_model = False):
        hin = np.ravel(hypers).tolist()
        # n_est, maxdepth, colnode, coltree, collevel, lam, alph = hin
        # colnode,coltree,collevel = [np.clip(g,0,1) for g in [colnode,coltree,collevel]]
        # n_est       = np.max([1,n_est])
        # maxdepth    = np.max([1,maxdepth])
        # lam         = np.max([0,lam])
        # alph        = np.max([0,alph])

        # n_est, maxdepth,alph = hin
        n_est,alph = hin
        maxdepth    = 15
        n_est       = np.max([1,n_est])
        maxdepth    = np.max([1,maxdepth])
        alph        = np.max([0,alph])

        # [1] - call the regressor
        m = xgboost.XGBRegressor(
                        # [ optimization inputs ]
                        n_estimators        = int(n_est),
                        max_depth           = int(maxdepth),
                        # colsample_bynode    = colnode,
                        # colsample_bytree    = coltree,
                        # colsample_bylevel   = collevel,
                        # reg_lambda          = lam,
                        reg_alpha           = alph,

                        # [ additional inputs ]
                        objective           = 'reg:squarederror',
                        # subsample           = 1,
                        # base_score          = 0.5,
                        # importance_type     = 'gain',
                        n_jobs              = 1,
                        nthread             = 1,
                        random_state        = 777,
                        seed                = 777,
                        )
        # [2] - if we want to just return the fitted model, then loets do so.
        if return_model:
            m.fit(np.asarray(x), np.asarray(y))
            mu = lambda xIn: m.predict(xIn).reshape(-1,1)
            return m,mu

        else:
            score = 0
            for i in range(len(idx)):
                rdx = idx[i]
                tdx = [g for g in ranger if not g in rdx]

                xt  = x[tdx,:].reshape(-1,x.shape[1])
                yt  = y[tdx,:].reshape(-1,1)
                xr  = x[rdx,:].reshape(-1,x.shape[1])
                yr  = y[rdx,:].reshape(-1,1)

                m.fit(xt,yt)

                err = np.square(yr - m.predict(xr)).mean() / len(idx)
                score += err
            # print(score)
            return score


    # [1] - grab the optimization bounds
    vv  = euclidean_distances(y)
    vv  = vv[vv>0].min()
    # bLo = [2, 1, 0,0,0,0,0]
    # bHi = [200, 10, 1, 1, 1, vv, vv]
    bLo = [2, 1, 0]
    bHi = [200, 10, vv]
    bLo = [2,0]
    bHi = [200,vv]
    optBounds = [bLo,bHi]

    # [2] - create the indices for training

    ranger = list(range(x.shape[0]))
    if nsplits == -1:
        nsplits = int(np.sqrt(x.shape[0]) + np.log(x.shape[0]))
    if holdout:
        xsub= np.divide(x-x.min(0), x.max(0)-x.min(0))
        ysub= (y-y.min())/(y.max()-y.min())
        idx = [subsampler(np.column_stack((xsub,ysub)), sampling = nsplits/x.shape[0], verbose=False)[1]]
        del xsub,ysub
    else:
        idx = np.array_split(ranger, nsplits)

    # [3] - run the optimization
    optFunc = lambda hypers: trainer(x,y, ranger, idx, hypers, return_model = False)
    retFunc = lambda hypers: trainer(x,y, ranger, idx, hypers, return_model = True)
    if train:
        O, xOpt, yOpt = SPOPT_EXE(
                optFunc = optFunc,
                optBounds = optBounds,
                # method = 'Nelder-Mead',
                method = 'L-BFGS-B',
                maxiter = maxiter,
                extra_eval=True,
                prefer_pso = False,
        )
        m,mu = retFunc(np.ravel(xOpt))
    else:
        nest = int(np.log(x.shape[0]) + np.sqrt(x.shape[0]))
        maxdepth = 6

        m,mu = retFunc([nest, maxdepth, 1, 1, 1, vv*1e-3, vv*1e-3])
    sig = lambda xIn: GBAL_acq(x, xIn, simple = not 'gbal' in name)
    return mu,sig




'''
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
=====================================================================
'''
















def model_master(x,y,xa=None, model_name = 'gp',acquisition=None, kernel = None):
    if kernel is None:
        kernel = KERNEL
    if '-' in model_name:
        acquisition = model_name.split('-')[-1]


    # [1] - GP-Based models
    # ===============================================
    if True in [f in model_name for f in ['gp','slrgp','aagp','lrk']]:
        mu,sig = fit_gp(x,y,xa=xa,model_name=model_name if not 'slrgp' in model_name else 'slrgp-alc', acquisition=acquisition if not 'slrgp' in model_name else 'alc', kernel=kernel)
    
    # [2] - LOD
    # ===============================================
    elif 'lod' in model_name:
        mu,sig = fit_lod(x,y,xa=xa,model_name=model_name, acquisition=None, kernel=kernel)
    
    # [3] - LocalRidge Regression
    # ===============================================
    elif 'localridge' in model_name:
        mu,sig = fit_localRidge(x,y,xa=xa,model_name=model_name)
        if 'gbal' in model_name:
            sig = lambda xIn: GBAL_acq(x,xIn)
        else:
            sig    = lambda xIn: maxiMin_acquisition(x,xIn)


    # [4] - LOESS Regression
    # ===============================================
    elif 'loess' in model_name:
        mu = fit_loess_regressor(x,y,train=not '*' in model_name)
        if 'gbal' in model_name:
            sig = lambda xIn: GBAL_acq(x,xIn)
        else:
            sig    = lambda xIn: maxiMin_acquisition(x,xIn)
    
    # [5] - XGBoost Regression
    # ===============================================
    elif 'xgb' in model_name:
        mu,sig = fit_xgb_regressor(x,y,train=True,holdout=True,name = model_name + '-gbal')
        sig    = lambda xIn: GBAL_acq(x,xIn)
    
    
    # [6] - DeepGP
    # ===============================================
    if 'deep' in model_name and 'gp' in model_name:
        # if not '[' in model_name:
        #     model_name += ' - [10_5]'
        # M = multiGP(kernel = kernel,
        #             name = model_name,
        #             )
        reg = multiGP(name = model_name, kernel = kernel, hypers = [0.0001,0.0001],)
        reg.fit(x, y, xa = xa, trainIters = deepGP_maxIters, nSeeds = 0, verbose = False, xe=None, hypers=None, layerDims=[10,5])
        mu = reg.mu
        sig= reg.sig

    acq = acquisition
    if not acq is None:
        if True in [g in acq for g in ['gbal','maximin']]:
            sig = lambda xIn: GBAL_acq(x, xIn,simple='maximin' in acquisition)

        if True in [g in acq for g in ['igs','gsy']]:
            sig = lambda xIn: GREEDY_acq(x, mu, xIn, improveGreed = 'igs' in acq)
    return mu,sig
