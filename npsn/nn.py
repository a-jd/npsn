'''
Module for Nuclear Power Surrogate Network's
Neural Network Training & Optimization.

This module will have the following objectives:
1. Abstract training & optimization into single user function (train)
2. Allow easy manipulation of search space & base architecture
'''

import keras
import numpy as np
import os

from hyperopt import fmin, Trials, STATUS_OK, tpe
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.hp import choice

from .dg import DataLoader
from .dg import append_to_hdf
from .pp import parse_trials


class BaseModel():
    """
    Class to instantiate NN model creation/training/testing
    """
    def __init__(self, x_train, y_train, x_test, y_test):
        '''
        Inputs:
            x/y train/test data
        '''
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.in_shape = x_train.shape[1:]
        self.out_shape = y_train.shape[1:]

    def train_model(self, params):
        '''
        Input a dict, params, containing:
            nint_dense: number of intermediate dense layers
            sint_dense: shape of intermediate dense layers
            aint_dense: activation of intermediate dense layers
            train_loss: loss type used for training
            optim_type: optimizer type used for training
            batch_size: batch size used per epoch
        Returns:
            Dict containing info on permutation
        '''
        nint_dense = params['nint_dense']
        sint_dense = params['sint_dense']
        aint_dense = params['aint_dense']
        train_loss = params['train_loss']
        optim_type = params['optim_type']
        batch_size = params['batch_size']

        # Create Network #
        # Input layer
        inp = keras.layers.Input(self.in_shape)
        # First dense layer
        d1 = keras.layers.Dense(np.prod(self.out_shape),
                                activation=aint_dense)(inp)
        # Intermediate dense layer
        for ninterm in range(nint_dense):
            d1 = keras.layers.Dense(sint_dense,
                                    activation=aint_dense)(d1)
        # Final dense layer
        d1 = keras.layers.Dense(np.prod(self.out_shape))(d1)
        # Output layer
        outp = keras.layers.Reshape(self.out_shape)(d1)

        # Compile and train #
        model = keras.Model(inp, outp)
        model.compile(optimizer=optim_type, loss=train_loss,
                      metrics=['mape', 'mse'])
        model.summary()
        history = model.fit(x=self.x_train, y=self.y_train, epochs=1000,
                            batch_size=batch_size,
                            validation_data=(self.x_test, self.y_test),
                            verbose=0)

        # Hyperopt loss for each permutation #
        # Taking val MSE from last 10 epochs
        loss_array = history.history['val_mse'][-10:]
        hyp_loss = np.mean(loss_array)
        return {'loss': hyp_loss, 'status': STATUS_OK, 'keras_model': model}


def generate_trial(isDef=False):
    '''
    Generate an initial Trial object.
    Inputs:
        isDef: Bool, if true initial guess defined
    Returns:
        empty or pre-defined trials object
    '''
    # Modify pts to have a different initial guess
    # NOTE: Hyperopt reads dict keys and values separately,
    # and organizes keys in alphabetical order. While values order
    # remains the same. Thus, place keys in alphabetical order.
    if isDef:
        pts = [{
                'aint_dense': 1,
                'batch_size': 0,
                'nint_dense': 2,
                'optim_type': 0,
                'sint_dense': 0,
                'train_loss': 0
                }]
        new_trials = generate_trials_to_calculate(pts)
    else:
        new_trials = Trials()
    return new_trials


def save_model(mfn, model, data_info):
    '''
    Save keras hdf5 model and append DataLoader settings
    Inputs:
        model: keras model to be saved
        mfn: file name to save to
        data_info: dict containing DataLoader settings
    '''
    model.save(mfn)
    dirnm = data_info['dirnm']
    n_x = data_info['n_x']
    n_y = data_info['n_y']
    rmCol = data_info['rmCol']
    append_to_hdf(mfn,
                  dirnm=dirnm,
                  n_x=n_x, n_y=n_y, rmCol=rmCol)


def get_best_model(trials):
    '''
    Go through trials and return best trained keras model
    Inputs:
        trials: trials type object from hyperopt training
    Returns:
        best_model: model with lowest loss
    '''
    minidx = np.argmin([tr['loss'] for tr in trials.results])
    return trials.results[minidx]['keras_model']


def train(PRJNM, datadir,
          n_x, n_y, rmCol=None, npy_check=False,
          guessBool=True, max_evals=1):
    '''
    Training driver of neuralnetwork with optimization
    Inputs:
        PRJNM: name to save .hdf5 and trail data
        datadir: path to read csv files from
        n_x: 1D array of input control blade heights
        n_y: 2D array of size (nelem, nnode) where
            nelem: number of fuel elements
            nnode: number of nodes per element
        rmCol: 1D array to remove any csv column
        npy_check: if .npy file with height list in dataset exists
        guessBool: if including initial guess for hyperparameters
        max_evals: if optimizing, >1, else == 1
    '''
    # Instantiate trial object
    new_trials = generate_trial(guessBool)

    # Instantiate DataLoader
    data_args = (datadir, n_x, n_y)
    data_loader = DataLoader(*data_args, rmCol=rmCol,
                             npy_check=npy_check)

    # Instantiate BaseModel
    model = BaseModel(*data_loader.load_data())

    # Define hyperparameter search space
    hpss = {
        'nint_dense': choice('nint_dense', list(range(6))),
        'sint_dense': choice('sint_dense', np.arange(1, 5)*16*24),
        'aint_dense': choice('aint_dense',
                             ['elu', 'relu', 'sigmoid',
                              'softsign', 'softplus', 'tanh']),
        'train_loss': choice('train_loss',
                             ['logcosh', 'mse', 'mape', 'msle']),
        'optim_type': choice('optim_type',
                             ['adam', 'rmsprop', 'sgd']),
        'batch_size': choice('batch_size', [4, 8, 16, 32])
    }

    # Start optimization
    fmin(model.train_model, space=hpss, algo=tpe.suggest,
         max_evals=max_evals, trials=new_trials)

    # Save best model
    best_model = get_best_model(new_trials)
    modelpath = os.path.join(os.getcwd(), PRJNM+'.hdf5')
    save_model(modelpath, best_model,
               data_loader.get_data_settings())
    print('The best model was:')
    best_model.summary()

    # save the .mat file of all trials, if optimizing
    parse_trials(PRJNM, new_trials)
