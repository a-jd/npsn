'''
Model Generator Module

This module will have the following objectives:
1. Abstract training & optimization into single user function (train)
2. Allow easy manipulation of search space & base architecture
'''

from hyperopt import fmin, tpe

from .models.base import TrainingHistory
from .dg import DataLoader
from .pp import parse_trials

from .models.ann import ANN
from .models.svr import SVR


def ModelGenerator(model_nm, data_info, data):
    '''
    Generates model to be trained.
    Inputs:
        model_nm: string, type of algorithm
        data_info: dict, information of training data
        data: tuple, x_train, y_train, x_test, y_test
    Returns:
        Instantiated daughter class of BaseModel
    '''

    tr_hist = TrainingHistory(model_nm)

    models = {
        'ann': ANN,
        'svr': SVR
    }

    return models[model_nm](data_info, *data, tr_hist)


def train(prj_nm, model_nm, datadir,
          n_x, n_y, rmCol=None, npy_check=False,
          guessBool=True, max_evals=1):
    '''
    Training driver of neuralnetwork with optimization
    Inputs:
        prj_nm: string, name to save model and trials
        model_nm: string, type of regression model used
        datadir: string, path to read csv files from
        n_x: 1D array of input control blade heights
        n_y: 2D array of size (nelem, nnode) where
            nelem: number of fuel elements
            nnode: number of nodes per element
        rmCol: 1D array to remove any csv column
        npy_check: if .npy file with height list in dataset exists
        guessBool: if including initial guess for hyperparameters
        max_evals: if optimizing, >1, else == 1
    '''
    # Instantiate DataLoader
    data_args = (prj_nm, datadir, n_x, n_y)
    data_loader = DataLoader(*data_args, rmCol=rmCol,
                             npy_check=npy_check)

    # Instantiate BaseModel
    model = ModelGenerator(model_nm,
                           data_loader.get_data_info,
                           data_loader.load_data())

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
