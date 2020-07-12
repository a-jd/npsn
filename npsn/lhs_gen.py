import numpy as np
import datetime
import os

from pyDOE import lhs


def gen_samples(nbanks, nsamples, dz, center, svBool=False):
    """
    Utility function to geferate LHS distributed samples.
    Samples are inputs for generating training data.
    Inputs:
        nbanks: number of control blades/banks
        nsamples: number of samples
        dz: uniform dist >half< width
        center: center of uniform dist
        svBool: if saving samples as external file (for later use)
    Returns:
        smp: list of samples
    Example usage:
       Generate 10 samples, for 6 banks about
       24 cm center +/- 2 cm
       gensamples(6, 10, 2., 24.)
    """
    smp = lhs(nbanks, samples=nsamples)*2*dz+(center-dz)
    if svBool:
        now = datetime.datetime.now()
        date = str(now.year) + "_" + str(now.month) + "_" + str(now.day)
        fn = 'saved-test-matrix_{}.npy'.format(date)
        cpath = os.path.join(os.getcwd(), fn)
        if not os.path.isfile(cpath):
            np.save(cpath, smp)
        else:
            raise(FileExistsError('Backup existing before proceeding.'))
    return smp


if __name__ == "__main__":
    # Sample plot
    smp = gen_samples(6, 200, 2., 24.)
    from matplotlib import pyplot as plt
    plt.scatter(range(100), smp[:, 0])
    plt.title('Sample for 24  center and dz = +/-2')
    plt.show()
