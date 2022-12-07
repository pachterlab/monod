"""
This script provides convenience functions for evaluating RNA distributions. 
"""


import numpy as np

import scipy
from scipy import integrate

from scipy.fft import irfftn

# from .nn_toolbox import basic_ml_bivariate, ml_microstate_logP


class CMEModel:
    """Stores and evaluates biological and technical variation models.

    This class is used as a general wrapper around modular, generating function-based
    solutions for chemical master equation models.

    We use "sampling," "sequencing," and "technical" parameters interchangeably.
    We use "biological," "biophysical," and "transcriptional" parameters interchangeably.
    When used in the context of CMEModel, a sampling parameter "samp" is gene-specific,
    and typically corresponds to "regressor" in inference.py.

    The sampling parameters use the following conventions:
        Null model uses samp=None.
        Bernoulli model uses probabilities.
        Poisson model uses log-sampling rates.

    Attributes
    ----------
    bio_model: str
        type of transcriptional model describing biological variation.
    seq_model: str
        type of sequencing model describing technical variation.
    amb_model: str
        type of model describing ambiguity in RNA assignements.
    available_biomodels: tuple of str
        implemented transcriptional model types.
    available_seqmodels: tuple of str
        implemented technical model types.
    fixed_quad_T: float
        time horizon used for integration with Gaussian quadature.
    quad_order: int
        Gaussian quadrature order.
    quad_vec_T: float or np.inf
        time horizon used for adaptive integration.
    quad_method: str
        quadrature method to use.
        if 'fixed_quad', use Gaussian quadrature via scipy.integrate.fixed_quad.
        if 'quad_vec', use adaptive integration via scipy.integrate.quad_vec.

    """

    def __init__(
        self,
        bio_model,
        seq_model,
        amb_model="None",
        quad_method="fixed_quad",
        fixed_quad_T=10,
        quad_order=60,
        quad_vec_T=np.inf,
    ):
        """Initialize the CMEModel instance.

        A typical mammalian or bacterial cell will be effectively described by bio_model = "Bursty".
        A typical sequencing workflow will be effectively described by seq_model = "Poisson";
            this model is mandatory for incorporating a length bias.
        A high-fidelity fit uses quad_method = "quad_vec". A low-fidelity fit uses "fixed_quad".
            High-fidelity fits are several orders of magnitude slower.
        amb_model is not used in a typical workflow.

        Parameters
        ----------
        bio_model: str
            type of transcriptional model describing biological variation.
        seq_model: str
            type of sequencing model describing technical variation.
        amb_model: str, optional
            type of model describing ambiguity in RNA assignements.
        quad_method: str, optional
            quadrature method to use.
            if 'fixed_quad', use Gaussian quadrature via scipy.integrate.fixed_quad.
            if 'quad_vec', use adaptive integration via scipy.integrate.quad_vec.
        fixed_quad_T: float, optional
            time horizon used for integration with Gaussian quadature.
        quad_order: int, optional
            Gaussian quadrature order.
        quad_vec_T: float or np.inf, optional
            time horizon used for adaptive integration.
        """
        self.bio_model = bio_model
        self.available_biomodels = (
            "Delay",
            "Bursty",
            "Extrinsic",
            "Constitutive",
            "CIR",
            "DelayedSplicing",
        )
        self.available_seqmodels = ("None", "Bernoulli", "Poisson")
        self.available_ambmodels = ("None", "Equal", "Unequal")
        self.amb_model = amb_model
        if seq_model in ["None", None, "Null"]:
            self.seq_model = "None"
        else:
            self.seq_model = seq_model
        self.set_integration_parameters(
            fixed_quad_T, quad_order, quad_vec_T, quad_method
        )

    def set_integration_parameters(
        self, fixed_quad_T, quad_order, quad_vec_T, quad_method
    ):
        """Set quadrature parameters.

        Parameters
        ----------
        fixed_quad_T: float
            time horizon used for integration with Gaussian quadature.
        quad_order: int
            Gaussian quadrature order.
        quad_vec_T: float or np.inf
            time horizon used for adaptive integration.
        quad_method: str
            quadrature method to use.
            if 'fixed_quad', use Gaussian quadrature via scipy.integrate.fixed_quad.
            if 'quad_vec', use adaptive integration via scipy.integrate.quad_vec.
        """
        self.fixed_quad_T = fixed_quad_T
        self.quad_order = quad_order
        self.quad_vec_T = quad_vec_T
        self.quad_method = quad_method

    def get_log_name_str(self):
        """Return the names of log-parameters for the model instance.

        Returns
        -------
        name: tuple of str
            TeX-formatted log-parameter names for the current transcriptional model.
            Useful for plot labels.
        """
        if self.bio_model == "Constitutive":
            return (r"$\log_{10} \beta$", r"$\log_{10} \gamma$")
        elif self.bio_model == "Delay":
            return (r"$\log_{10} b$", r"$\log_{10} \beta$", r"$\log_{10} \tau^{-1}$")
        elif self.bio_model == "DelayedSplicing":
            return (r"$\log_{10} b$", r"$\log_{10} \tau^{-1}$", r"$\log_{10} \gamma$")
        elif self.bio_model == "Bursty":
            return (r"$\log_{10} b$", r"$\log_{10} \beta$", r"$\log_{10} \gamma$")
        elif self.bio_model == "Extrinsic":
            return (r"$\log_{10} \alpha$", r"$\log_{10} \beta$", r"$\log_{10} \gamma$")
        elif self.bio_model == "CIR":
            return (r"$\log_{10} b$", r"$\log_{10} \beta$", r"$\log_{10} \gamma$")
        else:
            raise ValueError(
                "Please select a biological noise model from {}.".format(
                    self.available_biomodels
                )
            )

    def get_num_params(self):
        """Return the number of parameters for the model instance.
        Note that these are gene-specific parameters, i.e., global technical noise
        parameters are not used here.

        Returns
        -------
        numpars: int
            The number of gene-specific parameters.
        """
        numpars = 0
        if self.bio_model == "Constitutive":
            numpars += 2
        else:
            numpars += 3
        if self.amb_model == "Equal":
            numpars += 1
        elif self.amb_model == "Unequal":
            numpars += 2
        return numpars

    def eval_model_logL(self, p, limits, samp, data, hist_type="unique", EPS=1e-15):
        """Compute the log-likelihood of data under a set of parameters.

        Parameters
        ----------
        p: np.ndarray
            log10 biological parameters.
        limits: list of int
            grid size for PMF evaluation, size n_species.
        samp: None or np.ndarray
            sampling parameters, if applicable.
        data: tuple or np.ndarray
            experimental data histogram.
        hist_type: str, optional
            if "grid": the search data histogram was generated using np.histogramdd.
            if "unique": the search data histogram was generated using np.unique.
            "unique" is the preferred method for histogram generation, as it requires less memory to store.
        EPS: float, optional
            minimum allowed proposal probability mass. anything below this value is rounded up to EPS.

        Returns
        -------
        logL: float
            log-likelihood.
        """
        if hist_type == "grid":
            raise ValueError("Not yet implemented!")
            # proposal = self.eval_model_pss(p, limits, samp)
            # proposal[proposal < EPS] = EPS
            # filt = data > 0
            # data = data[filt]
            # proposal = proposal[filt]
            # d = data * np.log(data / proposal)
            # return np.sum(d)
        elif hist_type == "unique":
            x, f = data
            proposal = self.eval_model_pss(p, limits, samp)
            proposal[proposal < EPS] = EPS
            proposal = proposal[tuple(x.T)]
            logL = np.log(proposal)
            return np.sum(logL)

    def eval_model_kld(self, p, limits, samp, data, hist_type="unique", EPS=1e-15):
        """Compute the Kullback-Leibler divergence between data and a fit.

        Parameters
        ----------
        p: np.ndarray
            log10 biological parameters.
        limits: list of int
            grid size for PMF evaluation, size n_species.
        samp: None or np.ndarray
            sampling parameters, if applicable.
        data: tuple or np.ndarray
            experimental data histogram.
        hist_type: str, optional
            if "grid": the search data histogram was generated using np.histogramdd.
            if "unique": the search data histogram was generated using np.unique.
            "unique" is the preferred method for histogram generation, as it requires less memory to store.
        EPS: float, optional
            minimum allowed proposal probability mass. anything below this value is rounded up to EPS.

        Returns
        -------
        kld: float
            Kullback-Leibler divergence.
        """
        if hist_type == "grid":
            proposal = self.eval_model_pss(p, limits, samp)
            proposal[proposal < EPS] = EPS
            filt = data > 0
            data = data[filt]
            proposal = proposal[filt]
            d = data * np.log(data / proposal)
            return np.sum(d)
        elif hist_type == "unique":
            x, f = data
            # This was formerly the interface with nn_toolbox neural likelihood approximation methods.
            # It will be implemented in a future version.
            # if (
            #     (self.quad_method == "nn")
            #     and (self.bio_model == "Bursty")
            #     and (self.seq_model == "None")
            # ):
            #     log_EPS = np.log(EPS)
            #     log_proposal = ml_microstate_logP(p, x)
            #     filt = (log_proposal < log_EPS) | (~np.isfinite(log_proposal))
            #     log_proposal[filt] = log_EPS
            #     d = f * (np.log(f) - log_proposal)
            #     # print(np.sum(d))
            #     # raise ValueError
            #     return np.sum(d)
            # else:
            proposal = self.eval_model_pss(p, limits, samp)
            proposal[proposal < EPS] = EPS
            proposal = proposal[tuple(x.T)]
            d = f * np.log(f / proposal)
            return np.sum(d)

    def eval_model_pss(self, p, limits, samp=None):
        """Evaluate the PMF of the model over a grid at a set of parameters.

        Parameters
        ----------
        p: np.ndarray
            log10 biological parameters.
        limits: list of int
            grid size for PMF evaluation, size n_species.
        samp: None or np.ndarray, optional
            sampling parameters, if applicable.

        Returns
        -------
        Pss: np.ndarray
            the steady-state model PMF over a grid.
        """
        # This was formerly the interface with nn_toolbox neural likelihood approximation methods.
        # It will be implemented in a future version.

        # if (
        #     (self.quad_method == "nn")
        #     and (self.bio_model == "Bursty")
        #     and (self.seq_model == "None")
        # ):
        #     return basic_ml_bivariate(p, limits)
        # else:

        if (self.amb_model != "None") and (len(limits) == 2):
            raise ValueError("Please specify a limit for the ambiguous species.")

        u = []
        mx = np.copy(limits)
        mx[-1] = mx[-1] // 2 + 1
        for i in range(len(mx)):
            l = np.arange(mx[i])
            u_ = np.exp(-2j * np.pi * l / limits[i]) - 1
            u.append(u_)
        g = np.meshgrid(*[u_ for u_ in u], indexing="ij")
        for i in range(len(mx)):
            g[i] = g[i].flatten()
        g = np.asarray(g)[:, :, None]

        if self.amb_model == "Unequal":
            g_ = np.zeros((2, g.shape[1], g.shape[2]), dtype=np.complex128)
            p_amb = np.power(10, p[-2:])
            g_[0] = p_amb[0] * g[2] + (1 - p_amb[0]) * g[0]
            g_[1] = p_amb[1] * g[2] + (1 - p_amb[1]) * g[1]
            g = g_
            p = np.copy(p[:-2])  # better safe
        elif self.amb_model == "Equal":
            g_ = np.zeros((2, g.shape[1], g.shape[2]), dtype=np.complex128)
            p_amb = np.power(10, p[-1])
            g_[0] = p_amb * g[2] + (1 - p_amb) * g[0]
            g_[1] = p_amb * g[2] + (1 - p_amb) * g[1]
            g = g_
            p = np.copy(p[:-1])

        if self.seq_model == "Poisson":
            g = np.exp((np.power(10, samp))[:, None, None] * g) - 1
        elif self.seq_model == "Bernoulli":
            g *= np.asarray(samp)[:, None, None]
        elif self.seq_model == "None":
            pass
        else:
            raise ValueError(
                "Please select a technical noise model from {}.".format(
                    self.available_seqmodels
                )
            )

        gf = self.eval_model_pgf(p, g)
        gf = np.exp(gf)
        gf = gf.reshape(tuple(mx))
        Pss = irfftn(gf, s=tuple(limits))
        Pss = np.abs(Pss) / np.sum(np.abs(Pss))
        Pss = Pss.squeeze()
        return Pss

    def eval_model_pgf(self, p_, g):
        """Evaluate the log-PGF of the model over the complex unit sphere at a set of parameters.

        Parameters
        ----------
        p_: np.ndarray
            log10 biological parameters.
        g: np.ndarray
            complex PGF arguments, adjusted for sampling.

        Returns
        -------
        gf: np.ndarray
            generating function values at each argument.
        """
        p = 10**p_
        if self.bio_model == "Constitutive":  # constitutive production
            beta, gamma = p
            gf = g[0] / beta + g[1] / gamma
        elif self.bio_model == "Bursty":  # bursty production
            b, beta, gamma = p
            fun = lambda x: self.burst_intfun(x, g, b, beta, gamma)
            if self.quad_method == "quad_vec":
                T = self.quad_vec_T * (1 / beta + 1 / gamma + 1)
                gf = scipy.integrate.quad_vec(fun, 0, T)[0]
            elif self.quad_method == "fixed_quad":
                T = self.fixed_quad_T * (1 / beta + 1 / gamma + 1)
                gf = scipy.integrate.fixed_quad(fun, 0, T, n=self.quad_order)[0]
            else:
                raise ValueError("Please use one of the specified quadrature methods.")
        elif (
            self.bio_model == "Extrinsic"
        ):  # constitutive production with extrinsic noise
            alpha, beta, gamma = p
            gf = -alpha * np.log(1 - g[0] / beta - g[1] / gamma)
        elif self.bio_model == "Delay":  # bursty production with delayed degradation
            b, beta, tauinv = p
            tau = 1 / tauinv
            U = g[1] + (g[0] - g[1]) * np.exp(-beta * tau)
            gf = (
                -1 / beta * np.log(1 - b * U)
                + 1 / beta / (1 - b * g[1]) * np.log((b * U - 1) / (b * g[0] - 1))
                + tau * b * g[1] / (1 - b * g[1])
            )
        elif self.bio_model == "DelayedSplicing":
            b, tauinv, gamma = p
            tau = 1 / tauinv
            gf = tau * b * g[0] / (1 - b * g[0]) - 1 / gamma * np.log(1 - b * g[1])
        elif self.bio_model == "CIR":  # CIR-like:
            b, beta, gamma = p
            fun = lambda x: self.cir_intfun(x, g, b, beta, gamma)
            if self.quad_method == "quad_vec":
                T = self.quad_vec_T * (1 / beta + 1 / gamma + 1)
                gf = scipy.integrate.quad_vec(fun, 0, T)[0]
            elif self.quad_method == "fixed_quad":
                T = self.fixed_quad_T * (1 / beta + 1 / gamma + 1)
                gf = scipy.integrate.fixed_quad(fun, 0, T, n=self.quad_order)[0]
            else:
                raise ValueError("Please use one of the specified quadrature methods.")
            gf /= 2
        else:
            raise ValueError(
                "Please select a biological noise model from {}.".format(
                    self.available_biomodels
                )
            )
        return gf  # this is the log-generating function

    def cir_intfun(self, x, g, b, beta, gamma):
        """Evaluates the inverse Gaussian-driven CME process integrand at time x.

        This solution was reported by Gorin*, Vastola*, Fang, and Pachter (2021).
        It is a helper function for the CIR-like biological model.

        Parameters
        ----------
        x: float np.ndarray or float
            time or array of times to evaluate the integrand at.
        g: np.ndarray
            complex PGF arguments, adjusted for sampling.
        b: float
            burst size-like parameter.
        beta: float
            splicing rate.
        gamma: float
            degradation rate.

        Returns
        -------
        _: np.ndarray
            integrand value.
        """
        if np.isclose(beta, gamma):  # compute prefactors for the ODE characteristics.
            c_1 = g[0]  # nascent
            c_2 = x * beta * g[1]
        else:
            f = beta / (beta - gamma)
            c_2 = g[1] * f
            c_1 = g[0] - c_2

        U = b * (np.exp(-beta * x) * c_1 + np.exp(-gamma * x) * c_2)
        return 1 - np.sqrt(1 - 4 * U)

    def burst_intfun(self, x, g, b, beta, gamma):
        """Evaluates the bursty CME process integrand at time x.

        This solution was reported by Singh and Bokes (2012).
        It is a helper function for the bursty biological model.

        Parameters
        ----------
        x: float np.ndarray or float
            time or array of times to evaluate the integrand at.
        g: np.ndarray
            complex PGF arguments, adjusted for sampling.
        b: float
            burst size.
        beta: float
            splicing rate.
        gamma: float
            degradation rate.

        Returns
        -------
        _: np.ndarray
            integrand value.
        """

        if np.isclose(beta, gamma):  # compute prefactors for the ODE characteristics.
            c_1 = g[0]  # nascent
            c_2 = x * beta * g[1]
        else:
            f = beta / (beta - gamma)
            c_2 = g[1] * f
            c_1 = g[0] - c_2

        U = b * (np.exp(-beta * x) * c_1 + np.exp(-gamma * x) * c_2)
        return U / (1 - U)

    def get_MoM(self, moments, lb_log, ub_log, samp=None):
        """Compute method of moments parameter estimates.

        This method evaluates the method of moments biological parameter estimates
        at a particular set of sampling parameters and under the instantiated model.


        Parameters
        ----------
        moments: dict
            moments for the gene, including 'S_mean', 'U_mean', 'S_var', 'U_var'.
        lb_log: float np.ndarray
            log10 lower bounds on biological parameters.
        ub_log: float np.ndarray
            log10 upper bounds on biological parameters.
        samp: None or np.ndarray, optional
            sampling parameters, if applicable.

        Returns
        -------
        x0: np.ndarray
            log10 biological parameter estimates.
        """

        lb = 10**lb_log
        ub = 10**ub_log
        if self.seq_model == "Poisson":
            samp = 10**samp

        if self.bio_model == "Bursty" or self.bio_model == "CIR":
            try:
                b = moments["U_var"] / moments["U_mean"] - 1
            except:
                b = 1  # safe for U_mean = U_var = 0
            if self.seq_model == "Bernoulli":
                b /= samp[0]
            elif self.seq_model == "Poisson":
                b = b / samp[0] - 1

            b = np.clip(b, lb[0], ub[0])
            beta = b / moments["U_mean"]
            gamma = b / moments["S_mean"]
            x0 = np.asarray([b, beta, gamma])

        elif self.bio_model == "Delay":
            b = moments["U_var"] / moments["U_mean"] - 1

            if self.seq_model == "Bernoulli":
                b /= samp[0]
            elif self.seq_model == "Poisson":
                b = b / samp[0] - 1

            b = np.clip(b, lb[0], ub[0])
            beta = b / moments["U_mean"]
            tauinv = b / moments["S_mean"]
            x0 = np.asarray([b, beta, tauinv])

        elif self.bio_model == "DelayedSplicing":
            # b = moments["S_var"] / moments["S_mean"] - 1
            b = (moments["U_var"] / moments["U_mean"] - 1) / 2
            b = np.clip(b, lb[0], ub[0])
            tauinv = b / moments["U_mean"]
            gamma = b / moments["S_mean"]
            x0 = np.asarray([b, tauinv, gamma])

            if self.seq_model != "None":
                raise ValueError("Not implemented yet!")

        elif self.bio_model == "Constitutive":
            beta = 1 / moments["U_mean"]
            gamma = 1 / moments["S_mean"]
            x0 = np.asarray([beta, gamma])

        elif self.bio_model == "Extrinsic":
            if self.seq_model == "Poisson":
                alpha = moments["U_mean"] ** 2 / (
                    moments["U_var"] - moments["U_mean"] * (1 + samp[0])
                )
            else:
                alpha = moments["U_mean"] ** 2 / (moments["U_var"] - moments["U_mean"])

            beta = alpha / moments["U_mean"]
            gamma = alpha / moments["S_mean"]
            x0 = np.asarray([alpha, beta, gamma])
        else:
            raise ValueError("Please select from implemented models.")

        if self.seq_model in ("Bernoulli", "Poisson"):
            if self.bio_model == "Constitutive":
                x0 *= samp
            else:
                x0[1:] = x0[1:] * samp
        if self.amb_model == "Equal":  # basic
            x0 = np.concatenate((x0, [0.1]))  # just make a guess
        elif self.amb_model == "Unequal":
            x0 = np.concatenate((x0, [0.1, 0.1]))  # just make a guess
        for j in range(self.get_num_params()):
            x0[j] = np.clip(x0[j], lb[j], ub[j])
        x0 = np.log10(x0)
        if (~np.isfinite(x0)).any():
            x0 = (
                np.random.rand(self.get_num_params()) * (ub_log - lb_log) + lb_log
            )  # last resort -- makes it nondeterministic though
        return x0

    def eval_model_noise(self, p_, samp=None):
        """Compute CV2 fractions due to intrinsic, extrinsic, and technical noise.

        This method reports the fractions of normalized variance attributable to intrinsic (single-molecule),
        extrinsic (e.g., burst), and technical noise under the instantiated model.

        Parameters
        ----------
        p_: np.ndarray
            log10 biological parameters.
        samp: None or np.ndarray, optional
            sampling parameters, if applicable.

        Returns
        -------
        f: tuple of np.ndarrays
            array of size 3x2 or 2x2 reporting noise fractions for each species attributable to each source.
            dimension 0: variance fraction (intrinsic, extrinsic, technical)
            dimension 1: species (unspliced, spliced)
            The null technical noise model has dim 0 of size 2, as it has no technical noise component.
            dimension 0 sums to 1.

        """
        p = 10**p_
        if self.bio_model == "Constitutive":  # constitutive production
            beta, gamma = p
            mu = [1 / beta, 1 / gamma]
        elif self.bio_model == "Bursty":  # bursty production
            b, beta, gamma = p
            mu = [b / beta, b / gamma]
        elif (
            self.bio_model == "Extrinsic"
        ):  # constitutive production with extrinsic noise
            alpha, beta, gamma = p
            mu = [alpha / beta, alpha / gamma]
        elif self.bio_model == "Delay":  # bursty production with delayed degradation
            raise ValueError("Not yet implemented!")
        elif self.bio_model == "CIR":  # CIR-like:
            b, beta, gamma = p
            mu = [b / beta, b / gamma]

        mu = np.asarray(mu)
        noise_int = 1 / mu

        if self.bio_model == "Constitutive":  # constitutive production
            noise_ext = [0, 0]
        elif self.bio_model == "Bursty":  # bursty production
            noise_ext = [beta, beta * gamma / (beta + gamma)]
        elif (
            self.bio_model == "Extrinsic"
        ):  # constitutive production with extrinsic noise
            noise_ext = [1 / alpha, 1 / alpha]
        elif self.bio_model == "Delay":  # bursty production with delayed degradation
            raise ValueError("Not yet implemented!")
        elif self.bio_model == "CIR":  # CIR-like:
            noise_ext = [beta, beta * gamma / (beta + gamma)]
        noise_ext = np.asarray(noise_ext)

        if self.seq_model == "None":
            noise = noise_int + noise_ext
            return (noise_int / noise, noise_ext / noise)
        elif self.seq_model == "Bernoulli":
            noise = noise_int / samp + noise_ext
            return (
                noise_int / noise,
                noise_ext / noise,
                1 - noise_int / noise - noise_ext / noise,
            )
        elif self.seq_model == "Poisson":
            samp_ = 10**samp
            noise_tech = 1 / (mu * samp_)
            noise = noise_int + noise_ext + noise_tech
            return (noise_int / noise, noise_ext / noise, noise_tech / noise)
