# -*-coding:utf-8-*-

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


class H5Dict:
    """h5df parser"""
    def __init__(self, path, mode='a'):
        if isinstance(path, h5py.Group):
            self.data = path
        else:
            if h5py is None:
                raise ImportError('`keras model from hdf5'
                                  ' requires h5py library.')
            self.data = h5py.File(path, mode=mode)

    def __getitem__(self, attr):
        val = None
        if attr in self.data.attrs:
            val = self.data.attrs[attr]
            if type(val).__module__ == np.__name__:
                if val.dtype.type == np.string_:
                    val = val.tolist()
        elif attr in self.data:
            val = self.data[attr]
            if isinstance(val, h5py.Dataset):
                val = np.asarray(val)
            else:
                val = H5Dict(val)
        return val

    def close(self):
        self.data.close()
