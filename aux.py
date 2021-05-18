# miscellaneous useful functions and classes
import h5py
import numpy as np
import os

cc = np.concatenate

Dataset = h5py._hl.dataset.Dataset
Group = h5py._hl.group.Group
Reference = h5py.h5r.Reference


class Objects(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v


def c_tile(x, n):
    """Create tiled matrix where each of n cols is x."""
    return np.tile(x.flatten()[:, None], (1, n))


def r_tile(x, n):
    """Create tiled matrix where each of n rows is x."""
    return np.tile(x.flatten()[None, :], (n, 1))


def get_idx(z, z_0, dz, l):
    """
    Return closest valid integer index of continuous value.
    
    :param z: continuous value
    :param z_0: min continuous value (counter start point)
    :param dz: 
    """
    try:
        z[0]
    except:
        z = np.array([z]).astype(float)
        
    int_repr = np.round((z-z_0)/dz).astype(int)
    return np.clip(int_repr, 0, l-1)


def get_seg(x, min_gap):
    # get segments, and start/stop bounds
    included = (x).astype(int)
    change = np.diff(cc([[0], included, [0]]))
    start = (change == 1).nonzero()[0]
    stop = (change == -1).nonzero()[0]
    
    mask = np.ones((len(start), 2), dtype=bool)
    
    for cseg in range(len(stop)-1):
        if (start[cseg+1] - stop[cseg]) < min_gap:
            mask[cseg+1, 0] = False
            mask[cseg, 1] = False
            
    start = start[mask[:, 0]]
    stop = stop[mask[:, 1]]
    
    seg = [x[start_:stop_] for start_, stop_ in zip(start, stop)]
    bds = np.array(list(zip(start, stop)))  # M x 2 array of start, stop idxs
    
    return seg, bds


def mv_avg(t, x, wdw):
    # return symmetric moving average of x with wdw s
    x_avg = np.nan * np.zeros(x.shape)
    for it, t_ in enumerate(t):
        mt = ((t_- wdw/2) <= t) & (t < (t_ + wdw/2))
        x_avg[it] = np.nanmean(x[mt])
    return x_avg


def loadmat_h5(file_name):
    '''Loadmat equivalent for -v7.3 or greater .mat files, which break scipy.io.loadmat'''
    
    def deref_s(s, f, verbose=False):  # dereference struct
        keys = [k for k in s.keys() if k != '#refs#']
        
        if verbose:
            print(f'\nStruct, keys = {keys}')

        d = {}

        for k in keys:
            v = s[k]

            if isinstance(v, Group):  # struct
                d[k] = deref_s(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and isinstance(np.array(v).flat[0], Reference):  # cell
                d[k] = deref_c(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.dtype == 'uint16':
                d[k] = ''.join(np.array(v).view('S2').flatten().astype(str))
                if verbose:
                    print(f'String, chars = {d[k]}')
            elif isinstance(v, Dataset):  # numerical array
                d[k] = np.array(v).T
                if verbose:
                    print(f'Numerical array, shape = {d[k].shape}')

        return d

    def deref_c(c, f, verbose=False):  # dereference cell
        n_v = c.size
        shape = c.shape

        if verbose:
            print(f'\nCell, shape = {shape}')

        a = np.zeros(n_v, dtype='O')

        for i in range(n_v):
            v = f['#refs#'][np.array(c).flat[i]]

            if isinstance(v, Group):  # struct
                a[i] = deref_s(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and isinstance(np.array(v).flat[0], Reference):  # cell
                a[i] = deref_c(v, f, verbose=verbose)
            elif isinstance(v, Dataset) and v.dtype == 'uint16':
                a[i] = ''.join(np.array(v).view('S2').flatten().astype(str))
                if verbose:
                    print(f'String, chars = {a[i]}')
            elif isinstance(v, Dataset):  # numerical array
                a[i] = np.array(v).T
                if verbose:
                    print(f'Numerical array, shape = {a[i].shape}')

        return a.reshape(shape).T
    
    with h5py.File(file_name, 'r+') as f:
        d = deref_s(f, f)
        
    return d
