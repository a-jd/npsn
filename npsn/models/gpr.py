'''
NPSN Gaussian Process Regression Class
'''

import os

# Base model
from .base import BaseModel

# Import for GPR
from sklearn.metrics import mean_squared_error as sklmse
import numpy as np
import gpflow as gpf
from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter
import tensorflow as tf

# hyperopt imports
from hyperopt import STATUS_OK
from hyperopt.hp import choice, quniform, uniform

# misc imports
from joblib import dump, load

# kernel aliases
k_lin = gpf.kernels.Linear
k_exp = gpf.kernels.Exponential
k_m52 = gpf.kernels.Matern52
kc_si = gpf.kernels.SeparateIndependent


class GPR(BaseModel):
    def __init__(self, *args):
        self.model_nm = 'GPR'
        if len(args) == 6:
            super().__init__(*args)
        else:
            print("Empty {} initialized".format(self.model_nm))
            self.loaded_model = None
        self.file_ext = '.' + self.model_nm

    def train_model(self, params):
        '''
        Input a dict, params, containing:
            kernel: String, Covariance function
                    {Linear, Exponential, Matern52,
                     Linear+Exponential, Linear*Exponential,
                     Linear+Matern52, Linear*Matern52,
                     Exponential+Matern52, Exponential*Matern52,
                     Dim_Linear+Exponential, Dim_Linear*Exponential,
                     Dim_Linear+Matern52, Dim_Linear*Matern52,
                     Dim_Exponential+Matern52, Dim_Exponential*Matern52,
                     }
            nipts: Integer, number of inducing points (kernel definition)
        Returns:
            Dict containing info on combination
        '''
        kernel = params['kernel']
        nipts = params['nipts']

        # Flat versions of y (power/flux distribution)
        y_tr_fl, y_te_fl = self.flat_y()

        # Define kernel used
        y_flshp = np.prod(self.y_shape)
        if kernel == 'Linear':
            ker = kc_si([k_lin() for _ in range(y_flshp)])
        elif kernel == 'Exponential':
            ker = kc_si([k_exp() for _ in range(y_flshp)])
        elif kernel == 'Matern52':
            ker = kc_si([k_m52() for _ in range(y_flshp)])
        elif kernel == 'Linear+Exponential':
            kl = [k_lin() + k_exp() for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Linear*Exponential':
            kl = [k_lin() * k_exp() for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Dim_Linear+Exponential':
            kl = [k_lin(active_dims=[0]) + k_exp(active_dims=[1])
                  for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Dim_Linear*Exponential':
            kl = [k_lin(active_dims=[0]) * k_exp(active_dims=[1])
                  for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Linear+Matern52':
            kl = [k_lin() + k_m52() for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Linear*Matern52':
            kl = [k_lin() * k_m52() for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Dim_Linear+Matern52':
            kl = [k_lin(active_dims=[0]) + k_m52(active_dims=[1])
                  for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Dim_Linear*Matern52':
            kl = [k_lin(active_dims=[0]) * k_m52(active_dims=[1])
                  for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Exponential+Matern52':
            kl = [k_exp() + k_m52() for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Exponential*Matern52':
            kl = [k_exp() * k_m52() for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Dim_Exponential+Matern52':
            kl = [k_exp(active_dims=[0]) + k_m52(active_dims=[1])
                  for _ in range(y_flshp)]
            ker = kc_si(kl)
        elif kernel == 'Dim_Exponential*Matern52':
            kl = [k_exp(active_dims=[0]) * k_m52(active_dims=[1])
                  for _ in range(y_flshp)]
            ker = kc_si(kl)

        # Define inducing points
        # Inducing points = poitns at which kernel is trained/defined
        # Helps reduce computational requirement
        # Must be weighed against reduction in variance vs. pts
        # Hard coded upper and lower bounds for scaled control
        # rod heights (0.1, 0.9). See PowerReader._initiate_scalers
        # in npsn/dg.py
        single_ipts = np.linspace(0.1, 0.9, nipts)[:, None]
        ipts = np.repeat(single_ipts, self.x_shape, axis=1)

        ivars = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(ipts)
        )

        # Define gp model
        model = gpf.models.SVGP(
            ker, gpf.likelihoods.Gaussian(),
            inducing_variable=ivars,
            num_latent_gps=y_flshp
        )

        # Optimize
        opt = gpf.optimizers.Scipy()
        opt.minimize(
            model.training_loss_closure((self.x_train, y_tr_fl)),
            variables=model.trainable_variables,
            method='L-BFGS-B',
            options={"disp": True, "maxiter": ci_niter(2000)},
        )

        # Hyperopt loss for each combination
        y_predict, y_predict_var = model.predict_y(self.x_test)
        hyp_loss = sklmse(y_te_fl, y_predict)
        self.tr_hist.update_history(params, hyp_loss, model)

        return {'loss': hyp_loss, 'status': STATUS_OK}

    def hpss_space(self):
        kernel_ls = [
            'Linear', 'Exponential', 'Matern52',
            'Linear+Exponential', 'Linear*Exponential',
            'Dim_Linear+Exponential', 'Dim_Linear*Exponential',
            'Linear+Matern52', 'Linear*Matern52',
            'Dim_Linear+Matern52', 'Dim_Linear*Matern52',
            'Exponential+Matern52', 'Exponential*Matern52',
            'Dim_Exponential+Matern52', 'Dim_Exponential*Matern52',
        ]
        hpss = {
            'kernel': choice('kernel', kernel_ls),
            'nipts': choice('nipts', [21, 45, 75, 101])
        }
        return hpss

    def gen_trials(self, doGuess=False):
        return super().gen_trials()

    def save_model(self):
        '''
        Save GPR model and DataLoader settings
        GPR requires using tf.saved_model.save()
        '''
        # Get best GPR model
        model = self.tr_hist.best_model
        if model is None:
            raise(Exception('Model not trained.'))

        # Save GPR model
        tfdt = tf.float64
        tfts = tf.TensorSpec(shape=[None, self.x_shape[0]], dtype=tfdt)
        model.predict_f_compiled = \
            tf.function(model.predict_f, input_signature=[tfts])
        prj_nm = self.data_info['prj_nm']
        modelpath = os.path.join(os.getcwd(), prj_nm+self.file_ext)
        tf.saved_model.save(model, modelpath)

        # Save data_info with joblib
        datapath = os.path.join(modelpath, ".datainfo")
        dump(self.data_info, datapath)

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

        # Load GPR saved model
        fpath = os.path.join(os.getcwd(), file_nm)
        try:
            loaded_model = tf.saved_model.load(fpath)
        except Exception:
            print("Error loading {} model.".format(self.model_nm))
        else:
            self.loaded_model = loaded_model

        # Load data_info
        datapath = os.path.join(fpath, ".datainfo")
        try:
            data_info = load(datapath)
        except Exception:
            print("Error loading {} datainfo.".format(self.model_nm))
        else:
            self.data_info = data_info
            print("{} loaded.".format(file_nm))

        return data_info

    def eval_model(self, seekingVar=False):
        '''
        Provides access to evaluate inputs.
        Inputs:
            seekingVar: Bool, if variance instead of mean sought
        Returns:
            predict: Function, used to eval loaded model
        '''
        if self.loaded_model is None:
            raise(Exception('Model not loaded.'))

        # GPR requires reshaping output
        def predict(x_in):
            y_out, y_var_out = self.loaded_model.predict_f_compiled(x_in)
            if not seekingVar:
                y_final = y_out
            else:
                y_final = y_var_out
            return self.un_flat_y(y_final.numpy())

        return predict
