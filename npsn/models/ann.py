"""
NPSN Aritifical Neural Network Class
"""

import os
import numpy as np

# Base model
from .base import BaseModel

# Import for ANN
import keras

# hyperopt imports
from hyperopt import STATUS_OK
from hyperopt.hp import choice


class ANN(BaseModel):
    def __init__(self, *args):
        self.model_nm = 'ANN'
        if len(args) == 6:
            super().__init__(*args)
        else:
            print("Empty {} initialized".format(self.model_nm))
            self.loaded_model = None
        self.file_ext = '.' + self.model_nm

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
        inp = keras.layers.Input(self.x_shape)
        # First dense layer
        d1 = keras.layers.Dense(np.prod(self.y_shape),
                                activation=aint_dense)(inp)
        # Intermediate dense layer
        for ninterm in range(nint_dense):
            d1 = keras.layers.Dense(sint_dense,
                                    activation=aint_dense)(d1)
        # Final dense layer
        d1 = keras.layers.Dense(np.prod(self.y_shape))(d1)
        # Output layer
        outp = keras.layers.Reshape(self.y_shape)(d1)

        # Compile and train #
        model = keras.Model(inp, outp)
        model.compile(optimizer=optim_type, loss=train_loss,
                      metrics=['mape', 'mse'])
        model.summary()
        history = model.fit(x=self.x_train, y=self.y_train, epochs=1000,
                            batch_size=batch_size,
                            validation_data=(self.x_test, self.y_test),
                            verbose=0)

        # Hyperopt loss for each combination
        # Taking val MSE from last 10 epochs
        loss_array = history.history['val_mse'][-10:]
        hyp_loss = np.mean(loss_array)
        self.tr_hist.update_history(params, hyp_loss, model)

        return {'loss': hyp_loss, 'status': STATUS_OK}

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
        model = self.tr_hist.best_model
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
        modelpath = os.path.join(os.getcwd(), prj_nm+self.file_ext)
        model.save(modelpath)

        # Append data to hdf file
        self.append_to_hdf(modelpath, dirnm=dirnm,
                           n_x=n_x, n_y=n_y, rmCol=rmCol)

    def load_model(self, file_nm, inpdict):
        '''
        Load file_nm
        Inputs:
            file_nm: String, name of saved file
            inpdict: Dict, empty containing keys to be read
        Returns:
            inpdict: Dict, filled for DataLoader
        '''
        if file_nm[-len(self.file_ext):] != self.file_ext:
            raise(Exception('Wrong file_nm {}'.format(file_nm)))
        fpath = os.path.join(os.getcwd(), file_nm)
        try:
            self.loaded_model = keras.models.load_model(fpath)
        except Exception:
            print("Error loading {} model.".format(self.model_nm))
        else:
            print("{} loaded.".format(file_nm))
        # Settings used to train the loaded model
        inpdict = self.read_from_hdf(fpath, inpdict)
        return inpdict

    def eval_model(self):
        '''
        Provides access to evaluate inputs.
        Returns:
            A function that can be used to eval loaded ANN
        '''
        if self.loaded_model is None:
            raise(Exception('Model not loaded.'))
        return self.loaded_model.predict

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
    def read_from_hdf(fpath, input_dict):
        '''
        Read data stored from append_to_hdf
        Inputs:
            fpath: String, full path name of hdf5 keras model
            input_dict: Dict, empty with keys to fetch info
        Returns:
            input_dict: Dict, filled
        '''
        from tables import open_file
        h5file = open_file(fpath, mode='r')
        array = h5file.root.NN_Settings
        for key in input_dict:
            input_dict[key] = getattr(array.attrs, key)
            print('Read from hdf5: {}'.format(key))
        h5file.close()
        return input_dict
