
# [1] - python package import
# ==============================
import numpy as np
import GPy, deepgp
# from IPython.display import display



# from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from algorithms import euclidean_distances#, manhattan_distances
import sys,os
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# [2] - self-made packages
# ============================================
from algorithms import get_kernel_data_bounds as CKDG
from auxiliaries import grabBrackets

class multiGP:

    def __init__(self, kernel = ['gaussian','matern'][1], name = 'gp', hypers = None):
        # import GPy,deepgp
        self.name = name
        self.kernel=kernel


    def get_normData(self, xIn, yIn):
        self.xBounds = [[xIn[:,g].min() for g in range(xIn.shape[1])],[xIn[:,g].max() for g in range(xIn.shape[1])]]
        self.yBounds = [[yIn[:,g].min() for g in range(yIn.shape[1])],[yIn[:,g].max() for g in range(yIn.shape[1])]]

    def normDown(self, zIn, mode = ['x','y'][0]):

        zscale = [self.xBounds if 'x' in mode else self.yBounds][0]
        zOut = zIn * 0
        for g in range(zOut.shape[1]):

            xg = zIn[:,g]
            xg = (xg - zscale[0][g])/(zscale[1][g]-zscale[0][g])
            zOut[:,g] = xg

        return zOut


    def normUp(self, zIn, mode = ['x','y'][0]):

        zscale = [self.xBounds if 'x' in mode else self.yBounds][0]
        zOut = zIn * 0
        for g in range(zOut.shape[1]):

            xg = zIn[:,g]
            xg = (xg)*(zscale[1][g]-zscale[0][g]) +zscale[0][g]
            zOut[:,g] = xg

        return zOut


    def fit(self, x, y, xa = None, layerDims = [], trainIters = 500, nSeeds = 0, verbose = False, xe=None, hypers=None):

        # [note] - (5)-deepgp!-[10]-{@._drr}
        # this means 5% inducing points, ARD training, 10 dimensional latent space, AAGP vectorizations.
        zIn = x.copy()
        inducers = 1
        if '(' in self.name:
            inducers = int(grabBrackets(self.name, key='('))/100



        xa = [x if xa is None else xa][0]

        self.x = x
        self.y = y
        self.xa = xa


        xIn= x.copy()
        xPaster = lambda xIn: xIn

        idxs, idxa, d, v, da, dt, vt, dat, sBounds, nBounds, rBounds, raBounds = CKDG(x, y, xa=xa)


        v  = np.square(y-y.T)
        vt = v[np.tril_indices(y.shape[0],k=-1)]
        d,da = [euclidean_distances(g,g) for g in [x,xa]]
        dt,dat = [g[np.tril_indices(g.shape[0],k=-1)] for g in [d,da]]

        rMin = raBounds[0]
        # rMin = dt[dt>0].min()
        rMax = raBounds[1]
        # sMin = np.percentile(vt[vt>0],25)
        # sMax = np.percentile(vt[vt>0],75)
        sMin = sBounds[0]
        sMax = sBounds[1]
        nMin = nBounds[0]
        nMax = nBounds[1]

        if 'deep' in self.name:
            rMin = (rMin - raBounds[0])/(raBounds[1]-raBounds[0])
            rMax = (rMax - raBounds[0])/(raBounds[1]-raBounds[0])
            sMin = (vt - sBounds[0])/(sBounds[1]-sBounds[0])
            sMin = sMin[sMin>0].min()
            sMax = (sMax - sBounds[0])/(sBounds[1]-sBounds[0])
            nMin = 0
            nMax = 1e-2

            # print(['%0.3f'%(g) for g in [rMin,rMax, sMin,sMax, nMin,nMax]])


        ARD = '!' in self.name
        vectorizer = '@' in self.name

        if vectorizer == True:

            '''
            Func = lambda xIn: xIn
            a  = adjacencyVectorization(vType = 'ard', functionOnly = 1, CFMODE=0)
            SV = a.analyze(x,y,xa=xa,f_x=Func)
            '''
            SV,SR = self.fit_surrogate(x,y,xa=xa)
            x_sv = SV(x)
            # xIn  = np.matrix(np.column_stack((x,x_sv)))
            # xPaster = lambda xIn: np.matrix(np.column_stack((xIn,SV(xIn))))

            xIn = x_sv
            xPaster = lambda xIn: np.matrix(SV(xIn))

            zIn  = xIn.copy()
            self.supReg = SR
        # if inducers < 1:
        #     zIn,_ = geoSpace(xIn.copy(),m=int(np.ceil(inducers * x.shape[0])))


        if '[' in self.name and 'deep' in self.name:

            # layerDims = [int(g) for g in self.grabBrackets(self.name,key='[').split('_')]
            layerDims = grabBrackets(self.name,key='[').split('_')
            # print(layerDims)
            ldims = []
            for h in layerDims:

                if not '+' in h:
                    g = int(h)
                    if g == 0:
                        g = xIn.shape[1]
                    if g < 0:
                        g = int(np.ceil(xIn.shape[1]/np.abs(g)))
                if '+' in h:
                    g = int(h) * xIn.shape[1]


                ldims.append(g)
            layerDims = ldims


        ZPOINTS = int(np.max([2,int(inducers * x.shape[0])]))
        kBase = [GPy.kern.Matern52 if 'mat' in self.kernel else GPy.kern.RBF][0](xIn.shape[1], ARD=ARD, )# + GPy.kern.Bias(xIn.shape[1])
        if 'deep' in self.name:

            kPuts = []
            for i in range(len(layerDims)):
                k = [GPy.kern.Matern52 if 'mat' in self.kernel else GPy.kern.RBF][0](layerDims[i], ARD=ARD, )# + GPy.kern.Bias(layerDims[i])
                kPuts.append(k)
            kPuts.append(kBase)
            hLayers = len(kPuts)
            
            dimensions = [y.shape[1]] + layerDims + [xIn.shape[1]]

            self.get_normData(xIn,y)
            xIn = self.normDown(xIn, mode='x')
            



            # MODEL = deepgp.DeepGP(dimensions, y.A, xIn.A, kernels = kPuts, num_inducing = int(np.ceil(inducers*x.shape[0])), back_constraint = False,inits = 'None',normalize_Y=True)
            MODEL = deepgp.DeepGP(dimensions, y, xIn, kernels = kPuts, num_inducing = ZPOINTS, back_constraint = False,inits = 'None',normalize_Y=False, shuffle = False)

        else:
            # MODEL = GPy.models.SparseGPRegression(X = xIn.A, Y = y.A, kernel = kBase, num_inducing = int(np.ceil(inducers*x.shape[0])))
            MODEL = GPy.models.SparseGPRegression(X = xIn, Y = y, kernel = kBase, num_inducing =ZPOINTS )


        # [1] - constrain the bounds.

        if verbose == False:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

        MODEL['.*lengthscale'].constrain_bounded(rMin,rMax)
        MODEL['.*variance'].constrain_bounded(sMin,sMax)
        # MODEL['.*variance'].constrain_fixed(1)
        MODEL['.*Gaussian_noise'].constrain_bounded(nMin,nMax)
        # MODEL['.*inducing_inputs'].constrain_fixed()
        # MODEL['.*bias'].constrain_bounded(nMin,nMax)

        # >>> m.parameter_names()
        # ['obslayer.inducing inputs', 'obslayer.Mat52.variance', 'obslayer.Mat52.lengthscale', 'obslayer.Gaussian_noise.variance', 'obslayer.Kuu_var', 'obslayer.latent space.mean', 'obslayer.latent space.variance']
        # >>> m['.*Kuu_var']
        # ←[1mdeepgp.obslayer.Kuu_var←[0;0m:
        # Param([0.00094017, 0.00054844, 0.0005889 , 0.00038871, 0.00064148,
        #        0.00055227, 0.00061474, 0.00094146])


        if verbose == False:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        # [2] - optimize

        MODEL.optimize(max_iters=trainIters, messages=verbose,optimizer = 'lbfgs')
        # MODEL.optimize(max_iters=trainIters, messages=verbose,optimizer='scg')
        # MODEL.optimize(max_iters=trainIters, messages=verbose,optimizer='tnc')
        if nSeeds > 0:
            MODEL.optimize_restarts(num_restarts = nSeeds, verbose = verbose)

        if verbose == False:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


        mu = lambda xIn: MODEL.predict(xPaster(xIn))[0].reshape(-1,1)
        sig= lambda xIn: MODEL.predict(xPaster(xIn))[1].reshape(-1,1)

        if 'deep' in self.name:
            mu = lambda xIn: np.matrix(MODEL.predict(self.normDown(xPaster(xIn), mode='x'))[0]).reshape(-1,1)
            sig= lambda xIn: np.matrix(MODEL.predict(self.normDown(xPaster(xIn), mode='x'))[1]).reshape(-1,1)

        self.mu = mu
        self.sig= sig
        self.MODEL=MODEL
        self.xIn = xIn
        self.zIn = zIn

    def predict(self, xp, type_ = 'response'):

        predictions = {
                        'response': self.mu,
                        'variance': self.sig,
                        }
        pred = predictions[type_]
        return pred(xp)

    # def display(self):
    #     display(self.MODEL)


