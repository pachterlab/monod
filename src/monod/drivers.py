"""
This script provides one-parameter convenience functions for estimating biological parameters
for a single grid point. These functions are intended to take in a high-quality estimate 
of the technical noise parameters, and evaluate the biological parameters for many
samples in a parallelized fashion.
"""

# from .preprocess import * #dispensable
from .extract_data import extract_data
from .cme_toolbox import CMEModel
from .inference import InferenceParameters, GradientInference
from .analysis import load_search_data


def parfun_noisefree(x):
    ds, phys_lb, phys_ub, gradient_params = x
    fitmodel = CMEModel("Bursty", "None")
    inference_parameters = InferenceParameters(
        phys_lb,
        phys_ub,
        [1, 1],
        [1, 1],
        [1, 1],
        ds,
        fitmodel,
        use_lengths=False,
        gradient_params=gradient_params,
    )
    search_data = load_search_data(ds + "/raw.sd")
    full_result_string = inference_parameters.fit_all_grid_points(
        num_cores=1, search_data=search_data
    )
    return full_result_string


def parfun_poiss(x):
    ds, phys_lb, phys_ub, samp_lb, samp_ub, gradient_params = x
    fitmodel = CMEModel("Bursty", "Poisson")
    inference_parameters = InferenceParameters(
        phys_lb,
        phys_ub,
        samp_lb,
        samp_ub,
        [1, 1],
        ds,
        fitmodel,
        use_lengths=True,
        gradient_params=gradient_params,
    )
    search_data = load_search_data(ds + "/raw.sd")
    full_result_string = inference_parameters.fit_all_grid_points(
        num_cores=1, search_data=search_data
    )
    return full_result_string


def parfun_cir(x):
    ds, phys_lb, phys_ub, samp_lb, samp_ub = x
    fitmodel = CMEModel("CIR", "Poisson")
    inference_parameters = InferenceParameters(
        phys_lb,
        phys_ub,
        samp_lb,
        samp_ub,
        [1, 1],
        ds,
        fitmodel,
        use_lengths=True,
        gradient_params=gradient_params,
    )
    search_data = load_search_data(ds + "/raw.sd")
    full_result_string = inference_parameters.fit_all_grid_points(
        num_cores=1, search_data=search_data
    )
    return full_result_string


def parfun_delay(x):
    ds, phys_lb, phys_ub, samp_lb, samp_ub = x
    fitmodel = CMEModel("Delay", "Poisson")
    inference_parameters = InferenceParameters(
        phys_lb,
        phys_ub,
        samp_lb,
        samp_ub,
        [1, 1],
        ds,
        fitmodel,
        use_lengths=True,
        gradient_params=gradient_params,
    )
    search_data = load_search_data(ds + "/raw.sd")
    full_result_string = inference_parameters.fit_all_grid_points(
        num_cores=1, search_data=search_data
    )
    return full_result_string
