'''
Module for Nuclear Power Surrogate Network's
Data Generation.

This module will have the following objectives:
1. Provide class to store + read data sets
2. Provide methods to scale data sets
3. Provide methods to generate data sets for training

Structure requirements for dirName (directory containing
CSVs and *one* .npy file for the entire dataset):
    1. There should be only one .npy file in dirName
    2. The .npy file should contain control blade heights
       where each row is a single permutation, and each column
       represents the height for each control blade.
       E.g. if there are 200 permutations for 5 control blades,
       the shape of the .npy file array will be (200, 5).
    3. For each height permutation, there should be a
       corresponding .csv file containing the power distribution.
       The naming convention for the .csv file is explained:
         There should be a single integer in the name corresponding
         to the permutation row # in the .npy file.  E.g. for .npy
         row #20, the csv file can be named "pwr-20.csv".
         Any other convention would require rework of
         PowerReader._readFiles
       The contents of the .csv file is explained next:
         The first column of the csv file will contain labels
         for each row (ignored, but useful for user).
         The first row will contain height, that will be checked
         against the .npy file data for a match, within tolerance.
         The remaining rows and columns will contain the power
         distribution. Each column will represent a particular
         fuel element location, and each row will represent the
         axial node.

To generate a .npy file for a latin hypercube sampled normal
distribution, use .lhs_gen.gen_samples()
'''

import re
from os import listdir
from os.path import isfile, join

import numpy as np
from sklearn.preprocessing import MinMaxScaler as MMS


class PowerRecord:
    """
    Class to store single power record.
    Attributes:
        idn: Id # of PowerRecord (row of .npy file)
        inx: 1-D array of control device height/position
        iny: 2-D resulting reactor power
    """
    def __init__(self, idn, inx, iny):
        self.idn = idn
        self.inx = inx
        self.iny = iny


class PowerReader:
    """
    Class to read power distributions.
    """
    def __init__(self, dirName, n_x, n_y, rmCol=None):
        '''
        dirName: directory name containing csvs
        n_x: 1D array of input control blade heights
        n_y: 2D array of size (nelem, nnode) where
            nelem: number of fuel elements
            nnode: number of nodes per element
            if using rmCol, nnode should add len(rmCol)
        rmCol: remove any csv column (needed for dummy locs)
        '''
        assert type(n_x) is int or type(n_x) is np.int32, \
            "n_x not an int"
        assert type(n_y) is tuple, "n_y not a tuple"
        if rmCol is not None:
            assert type(rmCol) is tuple, "rmCol not a tuple"
        assert len(n_y) == 2, "len(n_y) != 2"
        assert type(n_y[0]) is int, "n_y[0] not an int"
        assert type(n_y[1]) is int, "n_y[1] not an int"
        self.n_x = n_x
        self.n_y = n_y
        self.hls, self.pwrl = self._readFiles(dirName, n_x, n_y)
        if rmCol is not None:
            self._remove_null_pwr(rmCol)
        # Keep track of scaling & initiate
        self.is_x_scaled = False
        self.is_y_scaled = False
        self._initiate_scalers(rmCol)

    @staticmethod
    def _readFiles(dirName, n_x, n_y):
        '''
        inputs:
            same defs as __init__
        returns:
            hls: height list generated
            pwrl: a list of PowerRecord objects
        '''
        # Function to read .csv files
        def __csvread(fn):
            sbh = np.loadtxt(fn, delimiter=',', max_rows=1,
                             usecols=range(1, 1+n_x))
            pwr = np.loadtxt(fn, delimiter=',', max_rows=n_y[0],
                             skiprows=1, usecols=range(1, 1+n_y[1]))
            return sbh, pwr

        # For strange error when using listdir with np.str_
        if type(dirName) == np.str_:
            dirName = str(dirName)
        # Look for saved .npy file (contains heights)
        hfn = [join(dirName, f)
               for f in listdir(dirName)
               if isfile(join(dirName, f)) and f.endswith('.npy')]
        assert len(hfn) == 1, "Dataset list error."
        hls = np.load(hfn[0])

        # Get all .csv files in directory
        fns = [f
               for f in listdir(dirName)
               if isfile(join(dirName, f)) and f.endswith('.csv')]

        # Read all .csv files using PowerRecord class
        pwrl = []
        for fn in fns:
            sbh, pwr = __csvread(join(dirName, fn))
            idn = int(re.findall('\d+', fn)[-1])
            #  Check if sbh read is equal to .npy file
            assert np.allclose(hls[idn], sbh), "sbh discrepancy"
            # SBH reshaped to be [sample, nfeatures] compliant
            pwrl.append(PowerRecord(idn, sbh.reshape(-1, 1), pwr))

        # Final output
        return hls, pwrl

    def _remove_null_pwr(self, rmCol):
        '''
        Removes columns listed in rmCol
        '''
        for rec in self.pwrl:
            assert ~np.any(rec.iny[:, rmCol]), "Removing non-zero cols?"
            rec.iny = np.delete(rec.iny, rmCol, axis=1)

    def _initiate_scalers(self, rmCol):
        '''
        Scaling for control device: linear minmax to [0.1, 0.9]
        Scaling for power: logarithmic element-wise minmax to [0.1, 0.9]
        '''
        # Treat rmCol
        if rmCol is None:
            nrmCol = 0
        else:
            nrmCol = len(rmCol)
        # Find limits
        minh = np.inf
        maxh = 0
        minp = np.full(self.n_y[1]-nrmCol, np.inf)
        maxp = np.zeros(self.n_y[1]-nrmCol)
        for rec in self.pwrl:
            # Min and max of all sbh (flattened array)
            minh = min(minh, np.amin(rec.inx))
            maxh = max(maxh, np.amax(rec.inx))
            # Min and max of each column (e.g. fuel element for MITR)
            minp = np.amin(np.vstack((minp, np.amin(rec.iny, axis=0))), axis=0)
            maxp = np.amax(np.vstack((maxp, np.amax(rec.iny, axis=0))), axis=0)
        # Save limits
        self.minh = minh
        self.maxh = maxh
        self.minp = minp
        self.maxp = maxp
        # Create MMS objects
        self.heightMMS = MMS((0.1, 0.9))
        self.heightMMS.fit([[minh], [maxh]])
        self.powerMMS = MMS((0.1, 0.9))
        self.powerMMS.fit(np.log([minp, maxp]))

    def scale_powers(self):
        if not self.is_y_scaled:
            for rec in self.pwrl:
                rec.iny = self.powerMMS.transform(np.log(rec.iny))
            self.is_y_scaled = True
        else:
            print('Power already scaled!')

    def descale_powers(self):
        if self.is_y_scaled:
            for rec in self.pwrl:
                rec.iny = np.exp(self.powerMMS.inverse_transform(rec.iny))
            self.is_y_scaled = False
        else:
            print('Power already descaled!')

    def descale_model_powers(self, iny):
        '''
        For nn-predicted data.
        Inputs:
            iny: Scaled power distribution
        Returns
            iny: Descaled power distribution
        '''
        assert type(iny) is np.ndarray, "Input is not np.ndarray!"
        for i in range(iny.shape[0]):
            iny[i] = np.exp(self.powerMMS.inverse_transform(iny[i]))
        return iny

    def scale_heights(self):
        if not self.is_x_scaled:
            for rec in self.pwrl:
                rec.inx = self.heightMMS.transform(rec.inx)
            self.is_x_scaled = True
        else:
            print('Heights already scaled!')

    def descale_heights(self):
        if self.is_x_scaled:
            for rec in self.pwrl:
                rec.inx = self.heightMMS.inverse_transform(rec.inx)
            self.is_x_scaled = False
        else:
            print('Heights already descaled!')

    def descale_model_heights(self, inx):
        '''
        For nn-predicted data.
        Inputs:
            inx: Scaled height
        Returns:
            inx: Descaled height
        '''
        assert type(inx) is np.ndarray, "Input is not np.ndarray!"
        for i in range(inx.shape[0]):
            inx[i] = np.squeeze(
                self.heightMMS.inverse_transform(inx[i].reshape(-1, 1)))
        return inx

    def load_data(self, tr_ratio=0.8, seededBool=True):
        '''
        Data loader for generating datasets.
        Inputs:
            tr_ratio: percent of training data [0,1]
            seededBool: random state fixed
        Returns:
            tuple containing xtrain, ytrain, xtest, ytest
        '''
        print('Loading data:')
        print('Is x scaled? {}'.format(self.is_x_scaled))
        print('Is y scaled? {}'.format(self.is_y_scaled))
        xtrainl = []
        ytrainl = []
        xtestl = []
        ytestl = []
        nd = len(self.pwrl)
        if seededBool:
            p = np.random.RandomState(seed=42).permutation(nd)
        else:
            p = np.random.permutation(nd)
        for i in p[:int(tr_ratio*nd)]:
            xtrainl.append(self.pwrl[i].inx)
            ytrainl.append(self.pwrl[i].iny)
        for i in p[int(tr_ratio*nd):]:
            xtestl.append(self.pwrl[i].inx)
            ytestl.append(self.pwrl[i].iny)
        return (np.array(xtrainl), np.array(ytrainl),
                np.array(xtestl), np.array(ytestl))


class DataLoader():
    """
    Class to handle data from PowerReader
    """
    def __init__(self, dirnm, n_x, n_y, rmCol=None):
        '''
        dirnm: directory name containing csvs
        n_x: 1D array of input control blade heights
        n_y: 2D array of size (nelem, nnode) where
            nelem: number of fuel elements
            nnode: number of nodes per element
            if using rmCol, nnode should add len(rmCol)
        rmCol: remove any csv column (needed for dummy locs)
        '''
        self.dirnm = dirnm
        self.n_x = n_x
        self.n_y = n_y
        self.rmCol = rmCol

    def get_data_settings(self):
        '''
        Return data info as a dict for saving
        '''
        data_info = {
            'dirnm': self.dirnm,
            'n_x': self.n_x,
            'n_y': self.n_y,
            'rmCol': self.rmCol
        }
        return data_info

    def load_data(self, c_r=None):
        '''
        Main loader for data
        Inputs:
            c_r: is the data reduction ratio
        Returns:
            x_train, y_train, x_test, y_test
        '''
        tdat = PowerReader(self.dirnm, self.n_x, self.n_y, self.rmCol)
        tdat.scale_heights()
        tdat.scale_powers()
        x_train, y_train, x_test, y_test = tdat.load_data()
        x_train = np.squeeze(x_train)
        x_test = np.squeeze(x_test)
        if c_r is not None and c_r > 0 and c_r < 1:
            x_train, y_train = self._data_reducer(c_r, x_train, y_train)
        print(x_train.shape)
        print(y_train.shape)
        return x_train, y_train, x_test, y_test

    def _data_reducer(self, c_r, x_in, y_in):
        '''
        Reduce the number of samples
        For determining how sensitive loss is to sample size
        Inputs:
            c_r: constant from 0->1, how much to reduce by in %/100
            x_in: original x input array
            y_in: original y output array
        Returns:
            x_out, y_out
        '''
        samp_sz = x_in.shape[0]
        # Random permutation to use for sampling x, y inputs
        p = np.random.RandomState(seed=42).permutation(samp_sz)
        p_ch = p[0:int(c_r*x_in.shape[0])]
        x_out = x_in[p_ch]
        y_out = y_in[p_ch]
        return x_out, y_out


def append_to_hdf(fname, **kwargs):
    '''
    A function to append dataset settings to hdf5 file.
    Useful to save NN model + setting sin one file.
    Inputs:
        fname: file name of hdf5 keras model
        kwargs: dict containing vars to store
    '''
    from tables import open_file
    h5file = open_file(fname, mode='a')
    array = h5file.create_array('/', 'NN_Settings', [])
    for key, value in kwargs.items():
        setattr(array.attrs, key, value)
    h5file.close()


def read_from_hdf(fname, input_dict):
    '''
    Read data stored from append_to_hdf
    Inputs:
        fname: file name of hdf5 keras model
        input_dict: empty dict with keys to fetch info
    Returns:
        input_dict: filled dict
    '''
    from tables import open_file
    h5file = open_file(fname, mode='r')
    array = h5file.root.NN_Settings
    for key in input_dict:
        input_dict[key] = getattr(array.attrs, key)
        print('Read from hdf5: {}'.format(key))
    h5file.close()
    return input_dict
