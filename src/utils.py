import numpy as np
import jax.numpy as jnp
import pandas as pd


def make_df(samples, log_probs, param_names):
    df = pd.DataFrame(samples, columns=param_names).assign(log_posterior=log_probs)
    return df


def load_shear_experiment():
    param_names = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]
    alpha       = np.array([0.3156, 0.831, -1.0]) # prior parameters (DES prior) # Om, s8, w0
    fiducial_dv = np.loadtxt("data/DES_shear-shear_a1.0_b0.5_data_vector")[:, 1]
    covariance  = np.loadtxt("data/covariance_cosmic_shear_PMEpaper.dat")
    Cinv        = np.linalg.inv(covariance)
    precision   = np.linalg.inv(np.matrix(covariance))
    derivatives = np.loadtxt("data/derivatives.dat").T
    F           = np.linalg.multi_dot([derivatives, precision, derivatives.T])
    Finv        = np.linalg.inv(F)
    lower       = np.array([0.05, 0.45, -1.40])
    upper       = np.array([0.55, 1.00, -0.10])
    return (param_names,) + tuple(
        jnp.asarray(_) for _ in (
            alpha, 
            fiducial_dv,
            covariance,
            Cinv,
            precision,
            derivatives,
            F, 
            Finv, 
            lower, 
            upper
        )
    )