#%%

# [1] - package imports
# =========================
import numpy as np
from scipy.optimize import minimize as scipopt




from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(777)))


# [2] - self-made imports
# =========================================
from algorithms import lhs_sampling, FFD

from auxiliaries import VS#,CS,VSS,CSS

# [3] - introduce auxiliary codes
# ================================
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=np.VisibleDeprecationWarning)


# [4] - Function Evaluator
# ===================================
def functionEvaluator(x, optFuncIn, logScale = True):

    if len(x.shape)==1:
        x = x.reshape(1,-1)
    # this function evaluates the function depending on the number of inputs.
    resp  = np.zeros((x.shape[0],1)) + np.inf
    nArgs = optFuncIn.__code__.co_argcount

    for i in range(resp.shape[0]):
        xi = x.reshape(resp.shape[0],-1)[[i],:]
        if nArgs == 1:

            try:
                r = optFuncIn(xi)
            except:
                try:
                    r = optFuncIn([xi[0,g] for g in range(xi.shape[1])])
                except:
                    if xi.shape[1] == 1:
                        r = optFuncIn(float(xi))
                    else:
                        r = optFuncIn(xi.A)

        else:
            r = optFuncIn(*[xi[0,g] for g in range(xi.shape[1])])
        resp[i,:] = r
        pass
    if logScale == True:
        if resp.min()> 0:
            resp = np.log(resp)
        if resp.min()== 0:
            resp = np.log(resp + 1)
    return resp



# [5] - Particle Swarm Optimization
# ==================================

class PSO:
    def __init__(
                    self,
                    optFunc,
                    optBounds,
                    nSamps=20,
                    nIters=10,
                    traductive=True,
                    addParticleCorners=False,
                    r1=[0.5,0.125,0.25][0],
                    r2=[0.5,0.125,0.25][0],
                    c1=2.05,
                    c2=2.05,
                    w=0.72984,
                    haltTol=1e-3,
                    logScale=False
                ):
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        self.__optFunc__ = optFunc
        self.w = w
        self.optBounds = optBounds
        self.optFunc = lambda xIn: functionEvaluator(
                                                    self.clipBounds(xIn),
                                                    optFunc,
                                                    logScale=logScale
                                                    )
        self.traductive = traductive
        self.nSamps = int(nSamps + (2**len(optBounds[0]))*int(addParticleCorners))
        self.nIters = nIters
        self.addParticleCorners = addParticleCorners


    def clipBounds(self, xIn):
        # xIn = xIn_.reshape(-1,len(self.optBounds[0]))
        optBounds = self.optBounds
        bLo, bHi = optBounds
        for g in range(xIn.shape[1]):
            xg = xIn[:, g]
            xg[xg < bLo[g]] = bLo[g]
            xg[xg > bHi[g]] = bHi[g]
            xIn[:, g] = xg
        # print(xIn.max(),xIn.min())
        return xIn

    def create_particles(self):
        # [INTRO] - this function will create the particles required.
        AC = self.addParticleCorners
        x  = lhs_sampling(n = self.nSamps, p=len(self.optBounds[0]), iterations=10)

        if AC:
            x2 = FFD(dims=len(self.optBounds[0]), levels=2)
            x  = VS(x2,x)
        bLo,bHi = self.optBounds
        for i in range(x.shape[1]):
            xi = x[:,i]
            xi = xi * (bHi[i]-bLo[i]) + bLo[i]
            x[:,i] = xi
        return x

    def optimize(self,):
        # [REFERENCE] - D:\Research\Research Notes\Heuristic Optimization\Particle swarm algorithm for solving systems of nonlinear equations.pdf

        # [1] - lets make the particles right now.

        x = self.create_particles()
        v = np.divide(x,np.sqrt(np.square(x).sum(1)).reshape(-1,1))
        v = v*1e-3
        w = 1
        x_tracks = np.zeros((x.shape[0],x.shape[1],self.nIters))
        y_tracks = np.ones((x.shape[0], 1, self.nIters))*np.inf
        c1 = 1
        c2 = 1

        # [2] - now we iterate.
        for i in range(self.nIters):
            y               = self.optFunc(x)
            x_tracks[:,:,i] = x
            y_tracks[:,:,i] = y

            # this will first get the best IN ALL iterations, then find the best OF ALL iterations.
            person_best_idx = np.argmin(y_tracks,axis=2)
            personal_bests  = [y_tracks[i,:,person_best_idx[i]] for i in range(x.shape[0])]
            p = np.concatenate([x_tracks[i,:,person_best_idx[i]] for i in range(x.shape[0])], axis=0)
            # p=np.matrix(p)


            # then, we use equation 2.4 to update the worst competitor.
            # person_worst_idx= np.argmax(np.ravel(personal_bests))
            # personal_worst  = y_tracks[person_worst_idx,0,person_best_idx[person_worst_idx]].reshape(-1,1)
            # pw     = x_tracks[person_worst_idx,:,person_best_idx[person_worst_idx]]
            # pw_unit= np.sqrt(np.square(pw).sum(1))
            # pw_unit[pw_unit==0] = 1
            # pw_unit= np.divide(pw,pw_unit)
            # eps    = 1e-8

            # central differencing
            '''
            dpw_dx = self.optFunc(pw+eps*pw_unit) - self.optFunc(pw-eps*pw_unit)
            boundDiff = np.matrix(self.optBounds).max(0) - np.matrix(self.optBounds).min(0)
            boundDiff[boundDiff==0] = 1
            dpw_dx = np.divide(dpw_dx, 2*eps * boundDiff)
            '''

            # forward differencing
            # dpw_dx = self.optFunc(pw+eps*pw_unit) - personal_worst
            # boundDiff = np.matrix(self.optBounds).max(0) - np.matrix(self.optBounds).min(0)
            # boundDiff[boundDiff==0] = 1
            # dpw_dx = np.divide(dpw_dx, eps * boundDiff)


            global_best_idx = np.argmin(np.ravel(personal_bests))
            global_best     = y_tracks[global_best_idx,0,person_best_idx[global_best_idx]]
            g               = x_tracks[global_best_idx,:,person_best_idx[global_best_idx]]
            if i == self.nIters-1:
                break


            # print([g.shape for g in [p,x,g]])
            # [3] - then, we follow equation 2.1 and 2.2 from the above paper.
            # [3] - according to the authors, the PSO algorithm converges fast but slows down.

            if self.traductive:
                w  = 0.4 * (i+1 - self.nIters) / np.square(self.nIters) + 0.4
                c1 = -3 * (i+1)/self.nIters+3.5
                c2 = 3 * (i+1)/self.nIters+0.5

            else:
                pass
                # r1 = np.matrix(np.diag([np.random.rand() for g in range(2)]))
                # r2 = np.matrix(np.diag([np.random.rand() for g in range(2)]))

            np.random.seed(777+i)
            r1 = np.matrix(np.diag([np.random.uniform() for g in range(x.shape[0])]))
            r2 = np.matrix(np.diag([np.random.uniform() for g in range(x.shape[0])]))
            v = w * v + c1 * r1 @ (p-x) + c2 * r2 @ (g-x)
            x = x + v

        self.xOpt = g
        self.yOpt = global_best
        self.x_track = x_tracks
        self.y_track = y_tracks

        return np.matrix(g), np.matrix(global_best)
    









def class_execute_optimizer(optFunc, optBounds, O='pso', addCorners=False, ns=150, ni=5):
    # [intro] - This function executes the optimizer in a self-contained way so we dont have to write lines of code every time jesusChrist


    OPTIMIZER = PSO(optFunc, optBounds, addParticleCorners=addCorners, nSamps = ns, nIters=ni)
    xOpt, yOpt = OPTIMIZER.optimize()
    xOpt = np.ravel(xOpt)
    return OPTIMIZER, xOpt, yOpt



def SPOPT_EXE(optFunc, optBounds, method = ['Powell','Nelder-Mead','L-BFGS-B'][0], maxiter = None, maxfeval = None, extra_eval = True, prefer_pso = False):


    if prefer_pso:
        ns = int(10 * int(np.sqrt(len(optBounds[1])) + np.log(len(optBounds[0])+1)))
        OPTIMIZER, xOpt,yOpt = class_execute_optimizer(
                                                        None,
                                                        optFunc     = optFunc,
                                                        optBounds   = optBounds,
                                                        ns          = ns,
                                                        ni          = maxiter,
                                                        )
        return OPTIMIZER, xOpt,yOpt
    else:
        bLo = optBounds[0]
        bHi = optBounds[1]
        bounds = [[bLo[g], bHi[g]] for g in range(len(bLo))]
        x0  = np.ravel(np.array(optBounds).mean(0))
        x1  = np.ravel(np.array(optBounds).min(0))
        x2  = np.ravel(np.array(optBounds).max(0))

        options = None
        if not maxiter is None:
            options = {'maxiter':maxiter}
        if not maxfeval is None:
            options['maxfeval'] = maxfeval

        OPTIMIZER = scipopt(
                                fun     = optFunc,
                                x0      = np.ravel(x0),
                                method  = method,
                                bounds  = bounds,
                                options = options,

                            )
        if extra_eval:
            O = [OPTIMIZER]
            for g in [x1,x2]:
                o = scipopt(
                                    fun     = optFunc,
                                    x0      = np.ravel(g),
                                    method  = method,
                                    bounds  = bounds,
                                    options = options
                                )
                O.append(o)
            yhats = [g.fun for g in O]
            idx = np.argmin(yhats)
            OPTIMIZER = O[idx]

        return OPTIMIZER, OPTIMIZER.x, OPTIMIZER.fun








if __name__ =='__main__':
    from equations import get_function
    import timeit

    f = 'z.8.2'
    # f = 'q.0.2'
    xa, ya, xe, ye, f, b=get_function(fName=f, n=300, e=1000, lhs_iters=10)
    tStart = timeit.default_timer()
    Func = f
    optBounds = b
    pso = PSO(Func, optBounds, addParticleCorners=False, traductive=True, nSamps=150)
    xOpt, yOpt = pso.optimize()
    print('Optimization time: %0.4f'%(timeit.default_timer()-tStart))
    print('Optimal Location: ',xOpt)
    print('Optimal Loss    : ',yOpt)
    # pso.plot(addEval=False)
# %%
