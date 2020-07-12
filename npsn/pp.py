'''
Module for Nuclear Power Surrogate Network's
Post Processing.

This module will have the following objectives:
1. Evaluate a keras model against train and test datasets
2. Generate csv files for quantitative analysis of performance
'''

import os
import time

import keras
import numpy as np
import scipy.io as scio

from .dg import PowerReader
from .dg import read_from_hdf


def map_error(x, xhat, errTyp='mape'):
    '''
    inputs:
        x: ground truth
        xhat: prediction
        -> shape of x, xhat = (nbatch,nz,nelem)
        errTyp: selection of error functions defined locally
    returns:
        noerr: 1D node-wise error
        elerr: 1D element-wise error
        coerr: 2D core-wise error
    '''
    def rmse(x, xhat, ax=None):
        if ax is None:
            out = np.sqrt(np.square(x-xhat))
        else:
            out = np.sqrt((np.square(x-xhat)).mean(axis=ax))
        return out

    def nrmse(x, xhat, ax=None):
        if ax is None:
            numer = np.sqrt(np.square(x-xhat))
            out = np.divide(numer, x)
        else:
            denom = x.mean(axis=ax)
            numer = np.sqrt((np.square(x-xhat)).mean(axis=ax))
            out = np.divide(numer, denom)
        return out

    def mape(x, xhat, ax=None):
        if ax is None:
            out = ((np.abs((xhat-x)/x))*100)
        else:
            out = ((np.abs((xhat-x)/x))*100).mean(axis=ax)
        return out

    def mape_std(x, xhat, ax=None):
        if ax is None:
            out = ((np.abs((xhat-x)/x))*100)
        else:
            out = np.square(((np.abs((xhat-x)/x))*100).std(axis=ax))
        return out

    assert x.shape == xhat.shape, "Shape mismatch."
    assert len(x.shape) == 3, "Too many dims."
    errFn = locals()[errTyp]

    # element-wise average error
    elerr = np.zeros(x.shape[-1])
    # node-wise average error
    noerr = np.zeros(x.shape[1])
    # core-wise average error
    coerr = np.zeros(x.shape[1:])
    # fill error
    if errTyp != 'mape_std':
        for i in range(x.shape[0]):  # loop over batches
            elerr += errFn(x[i], xhat[i], ax=0)
            noerr += errFn(x[i], xhat[i], ax=1)
            coerr += errFn(x[i], xhat[i], ax=None)
        noerr = noerr/x.shape[0]
        elerr = elerr/x.shape[0]
        coerr = coerr/x.shape[0]
    else:
        coerr_ls = []
        for i in range(x.shape[0]):  # loop over batches
            elerr += errFn(x[i], xhat[i], ax=0)
            noerr += errFn(x[i], xhat[i], ax=1)
            coerr_ls.append(errFn(x[i], xhat[i], ax=None))
        noerr = np.sqrt(noerr)
        elerr = np.sqrt(elerr)
        coerr = np.array(coerr_ls).std(axis=0)
    return noerr, elerr, coerr


def eval_model(PRJNM, inpdict, model):
    '''
    Evaluates keras model using map_error function and
    writes csv files into csv/ directory
    inputs:
        PRJNM: base name of inp model
        inpdict: keys to be loaded for data
        model: keras model to be evaluated
    '''
    def _load_data(ind):
        tdat = PowerReader(ind['dirnm'], int(ind['n_x']),
                           ind['n_y'], ind['rmCol'])
        tdat.scale_heights()
        tdat.scale_powers()
        return tdat

    tdat = _load_data(inpdict)
    # using default load_data tr_ratio=0.8
    xtr, ytr, xte, yte = tdat.load_data()
    xtr = np.squeeze(xtr)
    xte = np.squeeze(xte)

    ytrds = tdat.descale_model_powers(ytr)
    yptr = model.predict_on_batch(xtr)
    yptrds = tdat.descale_model_powers(yptr)

    yteds = tdat.descale_model_powers(yte)
    ypte = model.predict_on_batch(xte)
    ypteds = tdat.descale_model_powers(ypte)

    # Calculate time per batch
    start_time = time.time()
    _ = model.predict_on_batch(xtr)
    end_time = time.time()
    eval_time = (end_time-start_time)/xtr.shape[0]
    print('Time taken per evaluation: {} seconds'.format(eval_time))

    # Path checking
    csvpath = os.path.join(os.getcwd(), 'csvs')
    if not os.path.isdir(csvpath):
        os.mkdir(csvpath)
    path = os.path.join(csvpath, PRJNM+'_')

    errType = ['mape', 'mape_std']
    for errSel in errType:
        fn = path+errSel
        ntr, etr, ctr = map_error(ytrds, yptrds, errTyp=errSel)
        nte, ete, cte = map_error(yteds, ypteds, errTyp=errSel)
        np.savetxt(fn+"_ntr.csv", ntr, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_nte.csv", nte, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_etr.csv", etr, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_ete.csv", ete, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_ctr.csv", ctr, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_cte.csv", cte, delimiter=",", fmt='%.8e')


def parse_trials(PRJNM, trial):
    '''
    Function to parse trials object that results from a
    hyperas execution. Trials object contains information
    about each hyperparameter permutation and its result.
    Inputs:
        PRJNM: base name of inp model
        trial: pass trial object type = hyperopt.Trial
    Returns:
        None (but prints out a .mat file)
    '''
    # Path checking
    matpath = os.path.join(os.getcwd(), 'mats')
    if not os.path.isdir(matpath):
        os.mkdir(matpath)
    path = os.path.join(matpath, PRJNM+'_')

    fn = path+'hyppars'
    output_dict = {
        'labels': np.array(list(trial.vals.keys())),
        'values': np.array(list(trial.vals.values())).T,
        'losses': np.array(trial.losses())
    }
    scio.savemat(fn+"_values.mat", output_dict)
    return None


def show_map(PRJNM):
    '''
    Simple function to visualize error map
    Inputs:
        PRJNM: base name of inp model
    '''
    import matplotlib.pyplot as plt
    fn = 'csvs/'+PRJNM+'_'+'mape_cte.csv'
    err = np.genfromtxt(fn, delimiter=',')
    plt.imshow(err)
    plt.colorbar()
    plt.show()
    return None


def post_processor(PRJNM):
    '''
    Postprocesses model 'PRJNM.hdf' and creates output
    in cwd/csvs with error statistics
    Inputs:
        PRJNM: saved name of trained model
    '''
    fn = PRJNM+'.hdf5'
    fpath = os.path.join(os.getcwd(), fn)
    model = keras.models.load_model(fpath)

    # read model settings from hdf5 file
    inpdict = {'dirnm': None, 'n_x': None, 'n_y': None,
               'rmCol': None}
    inpdict = read_from_hdf(fpath, inpdict)

    # calculate the error metrics
    eval_model(PRJNM, inpdict, model)
