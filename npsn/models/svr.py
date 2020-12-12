'''
NPSN Support Vector Regression Class
'''

import numpy as np
from .mg import BaseModel

from sklearn.multioutput import MultiOutputRegressor as MOR
from sklearn.svm import NuSVR


class svr(BaseModel):
    def train_model(self, params):
        '''
        Input a dict, params, containing:
            nu: Float, fraction of support vectors (0,1]
            C: Float, penalty parameter of error (~1.0)
            kernel: String, 'linear', 'poly', 'rbf', sigmoid'
            degree: Int, degree of polynomial for poly
        Returns:
            Dict containing info on combination
        '''
        nu     = params['nu']
        C      = params['C']
        kernel = params['kernel']
        degree = params['degree']

        # Instantiate SVR
        mor = MOR(NuSVR(C=C, nu=nu, kernel=kernel, degree=degree))

        # Forward reshape of y (power/flux distribution)
        pre_shape = self.y_train.shape[0]
        post_shape = np.prod(self.in_shape)
        y_tr_fre = np.reshape(self.y_train, (pre_shape, post_shape))

        # Fit
        mor.fit(x_train, y_tr_fre)

    def hpss_space(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
