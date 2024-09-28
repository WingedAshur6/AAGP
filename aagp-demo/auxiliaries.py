import numpy as np
from joblib import Parallel
from tqdm import tqdm as TQDM

def VS(x1,x2):
    return np.vstack((x1,x2))
def CS(x1,x2):
    return np.column_stack((x1,x2))
def CSS(xList):
    return np.concatenate(xList, axis=1)
def VSS(xList):
    return np.concatenate(xList, axis=0)

def grabBrackets(stringIn, key = '('):

    endKey = {'(':')', '{':'}', '[':']','<':'>','/':'/'}[key]
    s1 = stringIn.index(key)+1
    s2 = stringIn.index(endKey)
    if not '/' in endKey:
        # print(stringIn[s1:s2])
        return stringIn[s1:s2]
    else:
        s2 = stringIn[s1:].index(endKey)+1
        return (stringIn[s1:])[:s2]


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with TQDM(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()