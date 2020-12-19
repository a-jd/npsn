'''
NPSN Gradient Boosting Regressor Class
'''

import os
from joblib import dump, load

# Base model
from .base import BaseModel

# Import for GBR
from sklearn.multioutput import MultiOutputRegressor as MOR
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.ensemble import GradientBoostingRegressor as skGBR

# hyperopt imports
from hyperopt import STATUS_OK
from hyperopt.hp import choice, quniform, uniform


class GBR(BaseModel):
    def __init__(self, *args):
        self.model_nm = 'GBR'
        if len(args) == 6:
            super().__init__(*args)
        else:
            print("Empty {} initialized".format(self.model_nm))
            self.loaded_model = None
        self.file_ext = '.' + self.model_nm

    def train_model(self, params):
        '''
        Input a dict, params, containing:
            loss_type: String, 'ls', 'lad', or 'huber'
            learning_rate: Float, ~0.1
            n_estimators: Int, boosting stages, ~100
            criterion: String, split quality, 'friedman_mse', 'mse', 'mae'
            max_depth: Int, depth of regressors, ~3
            max_features: String, method, 'auto', 'sqrt', 'log2'
        Returns:
            Dict containing info on combination
        '''
        loss_type = params['loss']
        learning_rate = params['learning_rate']
        n_estimators = int(params['n_estimators'])
        criterion = params['criterion']
        max_depth = int(params['max_depth'])
        max_features = params['max_features']

        model = MOR(skGBR(loss=loss_type, learning_rate=learning_rate,
                          n_estimators=n_estimators, criterion=criterion,
                          max_depth=max_depth, max_features=max_features))

        # Print current combination
        print('Current GBR combination: {}'.format(params))

        # Flat versions of y (power/flux distribution)
        y_tr_fl, y_te_fl = self.flat_y()

        # Fit
        model.fit(self.x_train, y_tr_fl)

        # Hyperopt loss for each combination
        y_predict = model.predict(self.x_test)
        hyp_loss = sklmse(y_te_fl, y_predict)
        self.tr_hist.update_history(params, hyp_loss, model)

        return {'loss': hyp_loss, 'status': STATUS_OK}

    def hpss_space(self):
        hpss = {
            'loss': choice('loss', ['ls', 'lad', 'huber']),
            'learning_rate': uniform('lr', 0.05, 0.4),
            'n_estimators': quniform('nest', 50, 200, 1),
            'criterion': choice('cri', ['friedman_mse', 'mse', 'mae']),
            'max_depth': quniform('md', 2, 6, 1),
            'max_features': choice('mf', ['auto', 'sqrt', 'log2'])
        }
        return hpss

    def gen_trials(self, doGuess=False):
        return super().gen_trials()

    def save_model(self):
        '''
        Save GBR model and DataLoader settings
        '''
        # Get best GBR model
        model = self.tr_hist.best_model
        if model is None:
            raise(Exception('Model not trained.'))
        pickle_dict = {
            'model': model,
            'data_info': self.data_info
        }

        # Save with joblib
        prj_nm = self.data_info['prj_nm']
        modelpath = os.path.join(os.getcwd(), prj_nm+self.file_ext)
        dump(pickle_dict, modelpath)

    def load_model(self, file_nm, inp_dict):
        '''
        Load file_nm
        Inputs:
            file_nm: String, name of saved file
            inp_dict: Dict, empty containing keys to be read
        Returns:
            inp_dict: Dict, filled for DataLoader
        '''
        if file_nm[-len(self.file_ext):] != self.file_ext:
            raise(Exception('Wrong file_nm {}'.format(file_nm)))
        fpath = os.path.join(os.getcwd(), file_nm)
        try:
            loaded_dict = load(fpath)
        except Exception:
            print("Error loading {} model.".format(self.model_nm))
        else:
            self.loaded_model = loaded_dict['model']
            self.data_info = loaded_dict['data_info']
            print("{} loaded.".format(file_nm))
        return self.data_info

    def eval_model(self):
        '''
        Provides access to evaluate inputs.
        Returns:
            predict: Function, used to eval loaded model
        '''
        if self.loaded_model is None:
            raise(Exception('Model not loaded.'))

        # GBR requires reshaping output
        def predict(x_in):
            y_out = self.loaded_model.predict(x_in)
            return self.un_flat_y(y_out)

        return predict
