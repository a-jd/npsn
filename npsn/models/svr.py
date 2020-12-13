'''
NPSN Support Vector Regression Class
'''

from .base import BaseModel

# Import for SVR
from sklearn.multioutput import MultiOutputRegressor as MOR
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.svm import NuSVR

# hyperopt imports
from hyperopt import STATUS_OK
from hyperopt.hp import choice, quniform, uniform


class SVR(BaseModel):
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
                'C': uniform('C_lin', 0.5, 1.5),
            },
            {
                'kernel': 'poly',
                'nu': uniform('nu_poly', 1e-5, 1),
                'C': uniform('C_poly', 0.5, 1.5),
                'degree': quniform('degree_poly', 2, 5, 1),
                'gamma': choice('gamma_poly', ['scale', 'auto'])
            },
            {
                'kernel': 'rbf',
                'nu': uniform('nu_rbf', 1e-5, 1),
                'C': uniform('C_rbf', 0.5, 1.5),
                'gamma': choice('gamma_rbf', ['scale', 'auto'])
            },
            {
                'kernel': 'sigmoid',
                'nu': uniform('nu_sigmoid', 1e-5, 1),
                'C': uniform('C_sigmoid', 0.5, 1.5),
                'gamma': choice('gamma_sigmoid', ['scale', 'auto'])
            },
        ])
        return hpss

    def gen_trials(self, doGuess=False):
        return super().gen_trials()

    def save_model(self):
        pass

    def load_model(self):
        pass
