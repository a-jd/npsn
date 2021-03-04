'''
NPSN Support Vector Regression Class
'''

import os
from joblib import dump, load

# Base model
from .base import BaseModel

# Import for SVR
from sklearn.multioutput import MultiOutputRegressor as MOR
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.svm import NuSVR

# hyperopt imports
from hyperopt import STATUS_OK
from hyperopt.hp import choice, quniform, uniform


class SVR(BaseModel):
    def __init__(self, *args):
        self.model_nm = 'SVR'
        if len(args) == 6:
            super().__init__(*args)
        else:
            print("Empty {} initialized".format(self.model_nm))
            self.loaded_model = None
        self.file_ext = '.' + self.model_nm

    def train_model(self, params):
        '''
        Input a dict, params, containing:
            nu: Float, fraction of support vectors (0,1]
            C: Float, penalty parameter of error (~1.0)
            kernel: String, 'linear', 'poly', 'rbf', sigmoid'
            degree: Int, degree of polynomial for poly
            gamma: String, 'scale'/'auto' for 'rbf', 'poly', 'sigmoid'
        Returns:
            Dict containing info on combination
        '''
        kernel = params['kernel']
        nu = params['nu']
        C = params['C']

        # Instantiate SVR
        if kernel in ['linear']:
            model = MOR(NuSVR(C=C, nu=nu, kernel=kernel))
        elif kernel in ['rbf', 'sigmoid']:
            gamma = params['gamma']
            model = MOR(NuSVR(C=C, nu=nu, kernel=kernel,
                              gamma=gamma))
        elif kernel in ['poly']:
            gamma = params['gamma']
            degree = params['degree']
            model = MOR(NuSVR(C=C, nu=nu, kernel=kernel,
                              degree=degree, gamma=gamma))

        # Print current combination
        print('Current SVR combination: {}'.format(params))

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
        hpss = choice('kernel_type', [
            {
                'kernel': 'linear',
                'nu': uniform('nu_lin', 1e-5, 1),
                'C': uniform('C_lin', 0.5, 10.0),
            },
            {
                'kernel': 'poly',
                'nu': uniform('nu_poly', 1e-5, 1),
                'C': uniform('C_poly', 0.5, 10.0),
                'degree': quniform('degree_poly', 2, 5, 1),
                'gamma': choice('gamma_poly', ['scale', 'auto'])
            },
            {
                'kernel': 'rbf',
                'nu': uniform('nu_rbf', 1e-5, 1),
                'C': uniform('C_rbf', 0.5, 10.0),
                'gamma': choice('gamma_rbf', ['scale', 'auto'])
            },
            {
                'kernel': 'sigmoid',
                'nu': uniform('nu_sigmoid', 1e-5, 1),
                'C': uniform('C_sigmoid', 0.5, 10.0),
                'gamma': choice('gamma_sigmoid', ['scale', 'auto'])
            },
        ])
        return hpss

    def gen_trials(self, doGuess=False):
        return super().gen_trials()

    def save_model(self):
        '''
        Save SVR model and DataLoader settings
        '''
        # Get best SVR model
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

        # SVR requires reshaping output
        def predict(x_in):
            y_out = self.loaded_model.predict(x_in)
            return self.un_flat_y(y_out)

        return predict
