'''
Module for Nuclear Power Surrogate Network's
Post Processing.

This module will have the following objectives:
1. Evaluate a keras model against train and test datasets
2. Generate csv files for quantitative analysis of performance
'''

import os
import time
import numpy as np

from .dg import PowerReader
from .mg import ModelGenerator


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

    def mae(x, xhat, ax=None):
        if ax is None:
            out = np.abs(xhat-x)
        else:
            out = np.abs(xhat-x).mean(axis=ax)
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
    # Currently written to handle batch of 2D arrays only.
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
        noerr = np.sqrt(noerr/x.shape[0])
        elerr = np.sqrt(elerr/x.shape[0])
        coerr = np.array(coerr_ls).std(axis=0)
    return noerr, elerr, coerr


def model_error(prj_nm, inp_dict, model_fn):
    '''
    Evaluates model using map_error and
    writes csv files into csv/ directory
    inputs:
        prj_nm: String, base name of model
        inp_dict: Dict, loaded for data
        model_fn: Function, model fn to be evaluated
    '''
    # Setup object to load data
    tdat = PowerReader(inp_dict['dirnm'], int(inp_dict['n_x']),
                       inp_dict['n_y'], inp_dict['rmCol'])
    tdat.scale_heights()
    tdat.scale_powers()

    # Using default tr_ratio=0.8
    xtr, ytr, xte, yte = tdat.load_data()
    xtr = np.squeeze(xtr)
    xte = np.squeeze(xte)

    ytrds = tdat.descale_model_powers(ytr)
    yptr = model_fn(xtr)
    yptrds = tdat.descale_model_powers(yptr)

    yteds = tdat.descale_model_powers(yte)
    ypte = model_fn(xte)
    ypteds = tdat.descale_model_powers(ypte)

    # Calculate time per batch
    start_time = time.time()
    _ = model_fn(xtr)
    end_time = time.time()
    eval_time = (end_time-start_time)/xtr.shape[0]
    print('Time taken per evaluation: {} seconds'.format(eval_time))

    # Path checking
    csvpath = os.path.join(os.getcwd(), 'csvs')
    if not os.path.isdir(csvpath):
        os.mkdir(csvpath)
    path = os.path.join(csvpath, prj_nm+'_')

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


def post_processor(prj_nm, model_nm):
    '''
    Postprocesses file_nm and creates output
    in cwd/csvs with error statistics
    Inputs:
        prj_nm: String, name of saved model
        model_nm: String, type of regression model used
    '''
    file_nm = prj_nm + '.' + model_nm

    # Inputs needed to setup new BaseModel class
    inp_dict = {'dirnm': None, 'n_x': None, 'n_y': None,
                'rmCol': None}

    # Instantiate model and access to prediction
    model = ModelGenerator(model_nm)
    inp_dict = model.load_model(file_nm, inp_dict)
    model_regressor = model.eval_model()

    # Calculate the error metrics
    model_error(prj_nm, inp_dict, model_regressor)


# Utility functions ###########


def show_map(prj_nm):
    '''
    Simple function to visualize error map
    Inputs:
        prj_nm: base name of inp model
    '''
    import matplotlib.pyplot as plt
    fn = 'csvs/'+prj_nm+'_'+'mape_cte.csv'
    err = np.genfromtxt(fn, delimiter=',')
    plt.imshow(err)
    plt.colorbar()
    plt.show()
    return None


def gpr_var_post_process(prj_nm, model_nm):
    '''
    Utility function to get variance results from GPR.
    Inputs:
        prj_nm: String, name of saved model
        model_nm: String, type of regression model used
    '''
    if model_nm != 'GPR':
        raise(Exception('Wrong model type {}.'.format(model_nm)))

    file_nm = prj_nm + '.' + model_nm

    # Inputs needed to setup new BaseModel class
    inp_dict = {'dirnm': None, 'n_x': None, 'n_y': None,
                'rmCol': None}

    # Instantiate model and access to prediction
    model = ModelGenerator(model_nm)
    inp_dict = model.load_model(file_nm, inp_dict)
    model_fn = model.eval_model(seekingVar=True)

    # Setup object to load data
    tdat = PowerReader(inp_dict['dirnm'], int(inp_dict['n_x']),
                       inp_dict['n_y'], inp_dict['rmCol'])
    tdat.scale_heights()
    tdat.scale_powers()

    ## x,y data
    # Using default tr_ratio=0.8
    xtr, ytr, xte, yte = tdat.load_data()
    xtr = np.squeeze(xtr)
    xte = np.squeeze(xte)

    # Fetching GPR variance for test & training data
    yptr = model_fn(xtr)
    ypte = model_fn(xte)
    # Normalize to power
    tr_std_n = tr_std/pow_tr
    te_std_n = te_std/pow_te

    # Path checking
    csvpath = os.path.join(os.getcwd(), 'csvs')
    if not os.path.isdir(csvpath):
        os.mkdir(csvpath)
    # Added _var_ to separate from normal error post-processing
    path = os.path.join(csvpath, prj_nm+'_var_')

    # mae will provide a simple average of variances
    errType = ['mae']
    for errSel in errType:
        fn = path+errSel
        # "*0" added to only provide average of yptrds/ypteds
        ntr, etr, ctr = map_error(tr_std_n, tr_std_n*0, errTyp=errSel)
        nte, ete, cte = map_error(te_std_n, te_std_n*0, errTyp=errSel)
        np.savetxt(fn+"_ntr.csv", ntr, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_nte.csv", nte, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_etr.csv", etr, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_ete.csv", ete, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_ctr.csv", ctr, delimiter=",", fmt='%.8e')
        np.savetxt(fn+"_cte.csv", cte, delimiter=",", fmt='%.8e')
