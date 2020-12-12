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
        self.in_shape = x_train.shape[1:]
        self.out_shape = y_train.shape[1:]
        self.tr_hist = tr_hist
        self.hps_guess = None

    def train_model(self, params):
        raise(NotImplementedError)

    def hpss_space(self):
        raise(NotImplementedError)

    def gen_trials(self):
        '''
        Generate an initial Trial object.
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

    def load_model(self):
        raise(NotImplementedError)


class TrainingHistory:
    def __init__(self, model_nm):
        """
        Class to keep a record of the training history
        Inputs:
            model_nm: String, type of regression model used
        """
        self.model_nm = model_nm
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
            self.min_loss_idx = self.loss.count
            self.best_model = model

