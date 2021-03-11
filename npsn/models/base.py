import numpy as np
from hyperopt import Trials
from hyperopt.fmin import generate_trials_to_calculate


class BaseModel():
    """
    Super class to instantiate model creation and training
    """
    def __init__(self, data_info, x_train, y_train, x_test, y_test, tr_hist):
        '''
        Super class will hold all the training/test data
        No additional pre/post-processing will occur
        Scalers should be called externally.
        Inputs:
            data_info: Dict, contains training data settings
                       Ref .dg.DataLoader.get_data_settings
            x/y train/test data
            tr_hist: TrainingHistory object to store optimization
                     data
        '''
        self.data_info = data_info
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.ntrain = x_train.shape[0]
        self.ntest = x_test.shape[0]
        self.x_shape = x_train.shape[1:]
        self.y_shape = y_train.shape[1:]
        self.tr_hist = tr_hist
        self.hps_guess = None
        self.loaded_model = None

    def train_model(self, params):
        raise(NotImplementedError)

    def hpss_space(self):
        raise(NotImplementedError)

    def gen_trials(self):
        '''
        Generate an initial Trial object.
        Redefine this function if you want custom guesses.
        Returns:
            trials: empty or pre-defined trials object
        '''
        # Modify pts to have a different initial guess
        # NOTE: Hyperopt reads dict keys and values separately,
        # and organizes keys in alphabetical order. While values order
        # remains the same. Thus, place keys in alphabetical order.
        if self.hps_guess is not None:
            trials = generate_trials_to_calculate(self.hps_guess)
        else:
            trials = Trials()
        return trials

    def save_model(self):
        raise(NotImplementedError)

    def load_model(self, file_nm, inpdict):
        raise(NotImplementedError)

    def eval_model(self):
        raise(NotImplementedError)

    def flat_y(self):
        '''
        To handle flattening of training data from
        (nbatch,nz,nelem) -> (nbatch,nz*nelem)
        Returns:
            y_train_flat, y_test_flat
        '''
        # Train matrix
        shape = (self.ntrain, np.prod(self.y_shape))
        y_train_flat = np.reshape(self.y_train, shape)
        # Test matrix
        shape = (self.ntest, np.prod(self.y_shape))
        y_test_flat = np.reshape(self.y_test, shape)
        return y_train_flat, y_test_flat

    def un_flat_y(self, y_flat):
        '''
        To handle unflattening of predicted data from
        (nbatch,nz*nelem) -> (nbatch,nz,nelem)
        Inputs:
            y_flat: Array, shape(nbatch,nz*nelem)
        Returns:
            y_reshaped: Array, shape (nbatch,nz,nelem)
        '''
        nbatch = y_flat.shape[0]
        n_y = self.data_info['n_y']
        if self.data_info['rmCol'] is None:
            nRmCol = 0
        else:
            nRmCol = len(self.data_info['rmCol'])
        n_y = (n_y[0], n_y[1]-nRmCol)

        # Reshape
        shape = (nbatch, n_y[0], n_y[1])
        return np.reshape(y_flat, shape)


class TrainingHistory:
    def __init__(self):
        """
        Class to keep a record of the training history
        """
        # Initiate tracking variables
        self.hps = []  # Hyperparameters list
        self.loss = []  # Loss list
        self.min_loss = np.inf
        self.min_loss_idx = -1
        self.best_model = None

    def update_history(self, hps, loss, model):
        self.hps.append(hps)
        self.loss.append(loss)
        if loss <= self.min_loss:
            self.min_loss = loss
            self.min_loss_idx = len(self.loss)
            self.best_model = model

    def best_model_info(self):
        if self.best_model is None:
            print("Not yet trained!")
        else:
            print("Best comb # {}".format(self.min_loss_idx))
            print("Loss was {}".format(self.min_loss))
            print("Hyper-parameters were:")
            print(self.hps[self.min_loss_idx-1])
