import numpy as np
from functools import wraps
import time
import hashlib

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        t = time.strftime("%H:%M:%S", time.gmtime(te-ts))
        print(f"func:{f.__name__} took: {t}")
        return result
    return wrap

def hash_array(arr):
    return hashlib.sha224(arr).hexdigest()

def camb_clfile(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls, tensCls or ScalCls types) returned as a dict of numpy arrays.
    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.
    """
    with open(fname) as f:
        firstline = next(f)
    keys = [i.lower() for i in firstline.split(' ') if i.isalpha()][1:]
    cols = np.loadtxt(fname).transpose()

    ell = np.int_(cols[0])
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)

    cls = {k : np.zeros(lmax + 1, dtype=float) for k in keys}

    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)

    w = lambda ell :ell * (ell + 1) / (2. * np.pi)
    wpp = lambda ell : ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
    wptpe = lambda ell :np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi) 
    for i, k in enumerate(keys):
        if k == 'pp':
            we = wpp(ell)
        elif 'p' in k and ('e' in k or 't' in k):
            we = wptpe(ell)
        else:
            we = w(ell)
        cls[k][ell[idc]] = cols[i + 1][idc] / we[idc]
    return cls

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret