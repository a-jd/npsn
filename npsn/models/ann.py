"""
NPSN Aritifical Neural Network Class
"""

import os
import numpy as np
from .base import BaseModel

# Import for NN
import keras

# hyperopt imports
from hyperopt import STATUS_OK
from hyperopt.hp import choice


class ANN(BaseModel):
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
            Dict containing info on combination
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

    def hpss_space(self):
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
        return hpss

    def gen_trials(self, doGuess=False):
        if doGuess:
            self.hps_guess = [{'aint_dense': 1,
                               'batch_size': 0,
                               'nint_dense': 2,
                               'optim_type': 0,
                               'sint_dense': 0,
                               'train_loss': 0}]
        return super().gen_trials()

    def save_model(self):
        '''
        Save keras hdf5 model and append DataLoader settings
        '''
        model = self.TrainingHistory.best_model
        if model is None:
            raise(Exception('Model not trained.'))

        # Unpack data_info
        data_info = self.data_info
        prj_nm = data_info['prj_nm']
        dirnm = data_info['dirnm']
        n_x = data_info['n_x']
        n_y = data_info['n_y']
        rmCol = data_info['rmCol']

        # Save hdf file
        modelpath = os.path.join(os.getcwd(), prj_nm+'.hdf5')
        model.save(modelpath)

        # Append data to hdf file
        self.append_to_hdf(modelpath, dirnm=dirnm,
                           n_x=n_x, n_y=n_y, rmCol=rmCol)

    @staticmethod
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

    @staticmethod
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
