'''
Model Generator Module

This module will have the following objectives:
1. Abstract training & optimization into single user function (train)
2. Allow easy manipulation of search space & base architecture
'''

import os
import numpy as np
import scipy.io as scio
from hyperopt import fmin, tpe

from .models.base import TrainingHistory
from .dg import DataLoader

from .models.ann import ANN
from .models.gbr import GBR
from .models.gpr import GPR
from .models.svr import SVR


def ModelGenerator(model_nm, **kwargs):
    '''
    Generates model to be trained.
    Inputs:
        model_nm: String, type of algorithm
    Optional kwargs:
        data_info: Dict, metadata of training set
        data: Tuple, x_train, y_train, x_test, y_test
    Returns:
        Instantiated daughter class of BaseModel
        If no kwargs, returns empty instantiated object.
    '''

    tr_hist = TrainingHistory()

    models = {
        'ANN': ANN,
        'GBR': GBR,
        'GPR': GPR,
        'SVR': SVR
    }

    if kwargs:
        try:
            data_info = kwargs['data_info']
            data = kwargs['data']
        except KeyError as kerr:
            print('Incorrect KeyError: {}'.format(kerr))
        else:
            generated_model = models[model_nm](data_info, *data, tr_hist)
    else:
        generated_model = models[model_nm]()

    return generated_model


def parse_trials(prj_nm, trial):
    '''
    Function to parse trials object that results from a
    hyperopt execution. Trials object contains information
    about each hyperparameter permutation and its result.
    Inputs:
        prj_nm: String, base model name
        trial: hyperopt.Trial, trial object
    Returns:
        None (but prints out a .mat file)
    '''
    # Path checking
    matpath = os.path.join(os.getcwd(), 'mats')
    if not os.path.isdir(matpath):
        os.mkdir(matpath)
    path = os.path.join(matpath, prj_nm+'_')
    file_nm = path+'hyppars_values.mat'

    output_dict = {
        'labels': np.array(list(trial.vals.keys())),
        'values': np.array(list(trial.vals.values()),
                           dtype=object).T,
        'losses': np.array(trial.losses())
    }
    scio.savemat(file_nm, output_dict)
    return None


def train(prj_nm, model_nm, datadir,
          n_x, n_y, rmCol=None, npy_check=False,
          guessBool=True, max_evals=1):
    '''
    Training driver of surrogate model with optimization
    Inputs:
        prj_nm: String, name to save model and trials
        model_nm: String, type of regression model used
        datadir: String, path to read csv files from
        n_x: Int, 1D array of input control blade heights
        n_y: Tuple(Int), 2D array of size (nelem, nnode) where
            nelem: number of fuel elements
            nnode: number of nodes per element
        rmCol: Tuple(Int), 1D array to remove any csv column
        npy_check: Bool, if .npy file with height list in dataset exists
        guessBool: Bool, if including initial guess for hyperparameters
        max_evals: Int, if optimizing, >1, else == 1
    Returns:
        None (but saves the best trained model)
    '''
    # Instantiate DataLoader
    data_args = (prj_nm, datadir, n_x, n_y)
    data_loader = DataLoader(*data_args, rmCol=rmCol,
                             npy_check=npy_check)

    # Instantiate BaseModel
    model = ModelGenerator(
        model_nm,
        data_info=data_loader.get_data_info(),
        data=data_loader.load_data())

    # Create trial object
    trials = model.gen_trials(doGuess=guessBool)

    # Define hyperparameter search space
    hpss = model.hpss_space()

    # Start optimization
    fmin(model.train_model, space=hpss, algo=tpe.suggest,
         max_evals=max_evals, trials=trials)

    # Save best model
    model.save_model()

    # Print best model info
    model.tr_hist.best_model_info()

    # save the .mat file of all trials, if optimizing
    parse_trials(prj_nm, trials)
