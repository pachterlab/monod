import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, stats
import mminference

from extract_data import make_dir, log, extract_data
from cme_toolbox import CMEModel  # may be unnecessary
import multiprocessing
import os

# lbfgsb has a deprecation warning for .tostring(), probably in FORTRAN interface
import warnings
from plot_aesthetics import aesthetics

from tqdm import tqdm
code_ver_global = "029"  # bumping up version April 2024

# from tqdm.contrib.concurrent import process_map  # or thread_map
# warnings.filterwarnings("ignore", category=DeprecationWarning) #let's do more gargeted stuff...

def perform_inference(h5ad_filepath,
    model,
    transcriptome_filepath=None,
    n_genes=100,
    seed=2813308004,
    viz=True,
    modality_name_dict=None,
    dataset_string=None,
    cf=None,
    code_ver=code_ver_global,
    filt_param=None,
    exp_filter_threshold=1,
    genes_to_fit=[],
    gradient_params={
        "max_iterations": 10,
        "init_pattern": "moments",
        "num_restarts": 1},
    use_lengths=True,
    run_meta="",
    phys_lb=None,
    phys_ub=None,
    samp_lb=None,
    samp_ub=None,
    gridsize=None,
    hist_type='unique',
    exclude_sigma=True,
    poisson_average_log_length=5,
    mek_means_params=None,
    num_cores=1, AIC_EPS=1e-20, AIC_offs=0):
    '''
    Load and filter data from h5ad file.
    Run inference procedure for the desired model, save parameters, uncertainty from Hessian and AIC values automatically.
    '''
    
    if mek_means_params:
        k, epochs = mek_means_params
    
    if not dataset_string:
        try:
            dataset_string = ''.join(h5ad_filepath.split('/')[-1].split('.')[:-1])
        except AttributeError:
            # For anndata, use the name of the anndata object.
            dataset_string = model.bio_model + '_' + model.seq_model
            log.info("No dataset name given (dataset_string=None). Saving as {}".format(dataset_string))
        
    make_dir(dataset_string)
    
    monod_adata = extract_data(h5ad_filepath,
    model,
    dataset_name=dataset_string,
    transcriptome_filepath=transcriptome_filepath,
    n_genes=n_genes,
    seed=seed,
    viz=viz,
    filt_param=filt_param,
    modality_name_dict=modality_name_dict,
    cf=cf,
    code_ver=code_ver,
    exp_filter_threshold=exp_filter_threshold,
    genes_to_fit=genes_to_fit, hist_type=hist_type, mek_means_params=mek_means_params)
    log.info('Data extracted')
    log.debug('hist type is %s',hist_type)
    
    search_data = searchdata_from_adata(monod_adata)
    log.info('Search data created.')

    if not transcriptome_filepath:
        use_lengths = False
        log.info('Lengths have not been given so are not being used')

    if not mek_means_params:
        inference_parameters = InferenceParameters(
            dataset_string,
            model,
            use_lengths=use_lengths,
            gradient_params=gradient_params,
            run_meta=run_meta,
            phys_lb=phys_lb,
            phys_ub=phys_ub,
            samp_lb=samp_lb,
            samp_ub=samp_ub,
            gridsize=gridsize,
        poisson_average_log_length=poisson_average_log_length)

    else:
        inference_parameters = mminference.InferenceParameters(
            dataset_string,
            model,
            use_lengths=use_lengths,
            gradient_params=gradient_params,
            run_meta=run_meta,
            phys_lb=phys_lb,
            phys_ub=phys_ub,
            samp_lb=samp_lb,
            samp_ub=samp_ub,
            gridsize=gridsize,
        poisson_average_log_length=poisson_average_log_length,
        k=k, epochs=epochs)
    
    log.info('Global inference parameters set.')

    if not mek_means_params:
        # Fit the model at all values of technical parameters, and save the location of the results.
        search_result = inference_parameters.fit_all_grid_points(search_data, num_cores=num_cores)
        log.info('Grid points fit.')
        search_result.find_sampling_optimum(discard_rejected=False)
        parameters_per_gene = search_result.phys_optimum
        log.info('Optimal parameters found.')
        num_params = search_result.model.get_num_params()

    else:
        # Fit the model at all values of technical parameters, and save the location of the results.
        search_result_list = inference_parameters.fit_all_grid_points(search_data, num_cores=num_cores)
        log.info('Grid points fit.')

        cluster_params = {}
        # Assume results are ordered according to desired cluster labels.
        for i, sr in enumerate(search_result_list):
            sr.find_sampling_optimum(discard_rejected=False)
            parameters_per_gene = sr.phys_optimum
            log.info('Optimal parameters found.')
            cluster_params[i] = parameters_per_gene
            
        num_params = search_result_list[0].model.get_num_params()

    # Load the AnnData object into memory if it is in backed mode
    if monod_adata.isbacked:
        monod_adata = monod_adata.to_memory()

    # Save parameter results to adata.
    for i in range(num_params):
        
        if not mek_means_params:
            param_name = search_result.model.param_str[i]
            without_log = param_name.replace(r"\log_{10} ", "")

            param_values = parameters_per_gene[:,i]
            monod_adata.var[without_log] = param_values
        else:
            param_name = search_result_list[0].model.param_str[i]
            without_log = param_name.replace(r"\log_{10} ", "")

            for j in cluster_params.keys():
                param_values = cluster_params[j][:,i]
                # cluster_param_column = 
                monod_adata.var["c{}_".format(j) + without_log] = param_values
    
    if mek_means_params:
        # log.info('Optimal parameters for each cluster saved to anndata. For cluster i, the .var attribute is of the form: ci_' + param_name)
        log.info('Optimal parameters for each cluster saved to anndata. For cluster i, the .var attribute is of the form: \"ci_'+without_log+ '\". Note that the parameters are given in log-base 10.')
    else:
        log.info('Optimal parameters saved to anndata, under .var attributes in the form: \"'+without_log + '\". Note that the parameters are given in log-base 10.')

    
    # Save AIC values to adata.
    if not mek_means_params:
        AIC_per_gene(search_result, monod_adata)
    else:
        # AIC values from meK-Means are the same for all SRs.
        AICs = get_AIC_mek_means(search_result_list, search_data, AIC_EPS=AIC_EPS, AIC_offs=AIC_offs)
        monod_adata.var['AIC'] = AICs
    
    log.info('AIC values calculated and saved under .var attribute: AIC.')

    if not exclude_sigma:
        if not mek_means_params:
            # Save uncertainties from Hessian.
            search_result.compute_sigma(search_data,num_cores=1)
            all_sigmas = search_result.sigma
            
            for i in range(num_params):
                param_name = search_result.model.param_str[i]
                sigmas = all_sigmas[:,i]
                without_log = param_name.replace(r"\log_{10} ", "")
                # monod_adata.var[r'$\sigma$_' + param_name] = sigmas#[:,i]
                monod_adata.var['error_' + without_log] = sigmas#[:,i]

        else:
            # Save uncertainties from Hessian.
            for j, sr in enumerate(search_result_list):
                
                sd =  sr._subset_search_data(search_data)
                sr.compute_sigma(search_data, num_cores=1)
                all_sigmas = sr.sigma
                
                for i in range(num_params):
                    param_name = sr.model.param_str[i]
                    without_log = param_name.replace(r"\log_{10} ", "")

                    sigmas = all_sigmas[:,i]
                    monod_adata.var[r"c{}_error_".format(j) + without_log] = sigmas#[:,i]

        if not mek_means_params:
            # NB log.
            log.info('Uncertainties per gene calculated, saved to anndata in .var attribute of the form: \"error_' + without_log + '\". Note that the errors are given for log-base 10 parameters.')
        else:
            # NB log.
            log.info('Uncertainties per gene calculated for each cluster. E.g. for cluser, i, uncertainty saved to anndata in .var attribute of the form: \"ci_error_'+ without_log + '\". Note that the errors are given in log-base 10.')

            
    # If meK-Means, save clusters.
    if mek_means_params:
        monod_adata.obs['cluster'] = None
        log.info('Saving clusters')
        for sr in search_result_list:
            cluster_label = sr.assigns
            monod_adata.obs.loc[sr.filt, 'cluster'] = cluster_label
        log.info('Cluster labels added to anndata .obs')

    # Also save entire objects to adata.

    if not mek_means_params:
        monod_adata.uns['search_result'] = search_result
        sr = search_result.store_on_disk()
        log.info('Search Result stored to %s', sr)
    else:
        monod_adata.uns['search_result_list'] = search_result_list

    monod_adata.uns['search_data'] = search_data
    inference_string = search_result.inference_string
    sd = search_data.store_on_disk(inference_string)

    adata_file_path = inference_string + '/monod_adata.pkl'
    with open(adata_file_path, 'wb') as adfs:
        pickle.dump(monod_adata, adfs)
    log.info('Anndata object stored to %s', adata_file_path)

    return monod_adata

def searchdata_from_adata(adata):

    n_genes = adata.n_vars

    # NB the order of the layers here will be enforced to be the same as the order of the model 
    # modalities defined in cme_toolbox.
    modality_name_dict = adata.uns['modality_name_dict']
    model = adata.uns['model']

    ordered_modalities = model.model_modalities
    ordered_layer_names = [modality_name_dict[modality] for modality in ordered_modalities]
    
    layers = np.array([adata.layers[layer_name] for layer_name in ordered_layer_names])

    M = adata.uns['M']

    hist = adata.uns['hist']

    moments = get_gene_moments(adata)

    gene_names = adata.var.index

    n_cells = adata.n_obs

    hist_type = get_hist_type_adata(adata)

    attr_names = [
        "M",
        "hist",
        "moments",
        "n_genes",
        "gene_names",
        "n_cells",
        "layers",
        "hist_type",
        "layer_names"
    ]

    attr_values = [M, hist, moments, n_genes, gene_names, n_cells, layers, hist_type, ordered_layer_names]

    try:
        gene_log_lengths = adata.var['log_lengths']
        attr_names += ['gene_log_lengths']
        attr_values += [gene_log_lengths]
        
    except KeyError:
        pass

    try:
        k, epochs = adata.uns['k'], adata.uns['epochs']
        attr_names += ['k', 'epochs']
        attr_values += [k, epochs]
    except KeyError:
        pass        
    
    search_data = SearchData(attr_names, *attr_values)
    
    return search_data


def get_gene_moments(adata):
    
    # Specify the columns you want to extract
    columns_of_interest = [i for i in adata.var.columns if 'MOM' in i]
    
    # Ensure the columns exist in adata.var
    for col in columns_of_interest:
        if col not in adata.var.columns:
            raise ValueError(f"Column '{col}' does not exist in adata.var")
    
    # Extract the selected columns
    selected_data = adata.var[columns_of_interest]
    
    # Convert the DataFrame to a list of dictionaries
    list_of_dicts = selected_data.to_dict(orient='records')
    
    return list_of_dicts


def get_gene_moments_single(adata, gene_index):

    # Create an empty dictionary to store the gene attributes
    gene_moment_dict = {}
    
    # Iterate through each column in adata.var and add it to the dictionary
    for column in adata.var.columns:
        if 'MOM_' in column:
            gene_moment_dict[column[4:]] = adata.var[column].tolist()[gene_index]

    return gene_moment_dict


def reject_genes(adata, viz=False,
        EPS=1e-15,
        threshold=0.05,
        bonferroni=True,
        reject_at_bounds=True,
        bound_thr=0.01,
        grouping_thr=5,
        use_hellinger=True,
        hellinger_thr=0.05,
        mek_means=False,
        save_csq=True,
                save_pval=False,
                save_hellinger=False):
    '''
    Perform chi-squared testing and add list of rejected genes to anndata object.

    Parameters
    ------------------
    save_csq: boolean
        whether to save chi-squared values per cluster if performing mek-means 
    save_pval: boolean
        whether to save p-values per cluster if performing mek-means analysis
    '''

    if not mek_means:
        search_data = adata.uns['search_data']
        try:
            search_result = adata.uns['search_result']
        except AttributeError:
            log.error('Did you mean to run with meK-Means? If so, make sure to set mek_means=True')
    
        csq, pval, hellinger = search_result.chisquare_testing(search_data,
            viz=viz,
            EPS=EPS,
            threshold=threshold,
            bonferroni=bonferroni,
            reject_at_bounds=reject_at_bounds,
            bound_thr=bound_thr,
            grouping_thr=grouping_thr,
            use_hellinger=use_hellinger,
            hellinger_thr=hellinger_thr)
    
        # Save chi-squared values, p-values, and rejected genes to adata.
        adata.var['csq'] = csq
        log.info('Chi-squared values for each gene have been added as \"csq\" in .var')

        adata.var['pval'] = pval
        log.info('P-values for each gene have been added as \"pval\" in .var')

        adata.var['hellinger'] = hellinger
        log.info('Hellinger distances for each gene have been added as \"hellinger\" in .var')

        adata.var['rejected_genes'] = search_result.rejected_genes
        log.info('Rejected genes are recorded in \"rejected_genes\" in .var')

        
        # Reset search_data and search_result.
        adata.uns['search_data'] = search_data
        adata.uns['search_result'] = search_result
        adata.uns['rejection_index'] = search_result.rejection_index

    else:
        search_data = adata.uns['search_data']
        search_result_list = adata.uns['search_result_list']

        new_sr_list = []
        
        for i in range(len(search_result_list)):
            
            sr = search_result_list[i]
            sd =  sr._subset_search_data(search_data)
            cluster_filter = sr.filt
            cluster = sr.assigns
            
            csq, pval, hellinger = sr.chisquare_testing(sd,
            viz=viz,
            EPS=EPS,
            threshold=threshold,
            bonferroni=bonferroni,
            reject_at_bounds=reject_at_bounds,
            bound_thr=bound_thr,
            grouping_thr=grouping_thr,
            use_hellinger=use_hellinger,
            hellinger_thr=hellinger_thr)

            # Save chi-squared values, p-values, and rejected genes to adata.
            if save_csq:
                adata.var['{}_csq'.format(cluster)] = csq

            if save_pval:
                adata.var['{}_pval'.format(cluster)] = pval

            if save_hellinger:
                adata.var['{}_hellinger'.format(cluster)] = hellinger

            adata.var['{}_rejected_genes'.format(cluster)] = sr.rejected_genes

            new_sr_list += [sr]

        log.info('Rejected genes in cluster i have been recorded under \"ci_rejected_genes\" in .var')
        if save_csq:
            log.info('Chi-squared values for each gene in cluster i have been added as \"ci_csq\" in .var')
        if save_pval:
            log.info('P-values for each gene in cluster i have been added as \"ci_pval\" in .var')
        if save_hellinger:
            log.info('Hellinger distances for each gene in cluster i have been added as \"ci_hellinger\" in .var')
            
        # Reset search_data and search_result.
        adata.uns['search_result_list'] = new_sr_list
        # This is the same for all clusters (all sr objects), so can be set once (correct?)
        adata.uns['rejection_index'] = sr.rejection_index

    # Return list of rejected genes.
    return adata


# Use class to make inference faster.
class SearchData:
    """Container for data for for inference, visualization, and testing.

    Attributes
    ----------
    attr_names: tuple of str

    layers: int np.ndarray
        raw data from the layers of interest, size n_species x n_genes x n_cells.
    M: int np.ndarray
        grid size for PMF evaluation, size n_species x n_genes.
    hist: tuple or np.ndarray
        histogram of raw data, used to evaluate divergences.
        if tuple, generated by np.unique.
        if np.ndarray, generated by np.histogramdd.
    moments: list of dict
        length-n_genes list containing moments for each gene.
        moments include 'mod2_mean', 'mod1_mean', 'mod2_var', 'mod1_var', and are used to define MoM estimates.
        Also covariances in form 'mod1_mod2_covar'
    gene_log_lengths: float np.ndarray
        log lengths of analyzed genes.
    n_genes: int
        number of genes to analyze.
    gene_names: str np.ndarray
        list of genes to analyze.
    n_cells: int
        number of cells in the dataset.
    hist_type: str
        metadata defining the type of histogram.
    """

    def __init__(self, attr_names, *input_data):
        """Creates a SearchData object from raw data.

        Parameters
        ----------
        attr_names: tuple
            list of attributes to store, provided in extract_data.
        *input_data
            attributes to store, as enumerated in the class definition.
        """
        for j in range(len(input_data)):
            setattr(self, attr_names[j], input_data[j])

    def store_on_disk(self, inference_string):
        """This helper method attempts to store the SearchData object to disk.

        Returns
        -------
        full_result_string: str
            file location.
        """
        try:
            full_result_string = inference_string + "/search_data.res"
            with open(full_result_string, "wb") as sdfs:
                pickle.dump(self, sdfs)
            log.info("Search data stored to {}.".format(full_result_string))
        except:
            log.error(
                "Search data could not be stored to {}.".format(
                    full_result_string
                )
            )
        self.full_result_string = full_result_string
        return full_result_string

            

def AIC_per_gene(search_result, monod_adata):
    """
    Computes the AIC value for each gene for a single model.

    Parameters
    ----------
    search_result : SearchResult
        The search result object containing model fitting information.
    adata : anndata.AnnData
        The AnnData object containing the gene expression data.

    Returns
    -------
    np.ndarray
        The AIC values for each gene.
    """
    search_data = searchdata_from_adata(monod_adata)
    n_cells = monod_adata.n_obs
    # Calculate the log-likelihood for each gene
    logL = search_result.get_logL(search_data, n_cells)
    
    # Calculate the number of parameters
    n_params = search_result.sp.n_phys_pars
    
    # Calculate AIC for each gene
    AIC = 2 * n_params - 2 * logL

    monod_adata.var['AIC'] = AIC
    
    return AIC

def get_hist_type(search_data):
    """A helper function for backwards compatibility.

    If the histogram type is not specified in the SearchData object, assume it is the legacy
    type "grid".

    Parameters
    ----------
    search_data: monod.extract_data.SearchData
        SearchData object with the data to fit.

    Returns
    -------
    hist_type: str
        flavor of histogram used to generate search_data, either "unique" or "grid" or "none".
    """

    if hasattr(search_data, "hist_type") and search_data.hist_type == "unique" or "none":
        hist_type = search_data.hist_type
    else:
        hist_type = "grid"
    return hist_type



class InferenceParameters:
    """Stores parameters and distributes the multi-grid point inference procedure.

    Attributes
    ----------
    gradient_params: dict
        settings for gradient descent.
        "max_iterations" defines the maximum number of gradient descent iterations.
        "init_pattern" defines whether the first try starts at the method of moments estimate.
        "num_restarts" defines how many attempts should be made.
    phys_lb: float np.ndarray
        log10 lower bounds on biological parameters.
    phys_ub: float np.ndarray
        log10 upper bounds on biological parameters.
    grad_bnd: scipy.optimize.Bounds
        log10 lower and upper bounds on biological parameters.
    use_lengths: bool
        if True, the nascent Poisson model technical variation parameter is a
        coefficient multiplied by gene length.
        if False, the parameter is the genome-wide nascent sampling rate.
    samp_lb: np.ndarray
        log10 lower bounds on technical variation parameters.
    samp_ub: np.ndarray
        log10 upper bounds on technical variation parameters.
    gridsize: list of ints or int np.ndarray
        grid size for evaluating the technical variation parameters.
    model: monod.cme_toolbox.CMEModel
        CME model used for inference.
    n_phys_pars: int
        number of biological model parameters.
    n_samp_pars: int
        number of technical variation model parameters. Set to 2 for consistency.
    dataset_string: str
        dataset-specific directory location.
    inference_string: str
        run-specific directory location within dataset_string.
    sampl_vals: list of lists of floats
        list of grid points.
    
    X: np.ndarray
        grid point values representing unspliced RNA sampling parameters.
    Y: np.ndarray
        grid point values representing spliced RNA sampling parameters.
    n_grid_pts: int
        total number of grid points to evaluate.

    """

    def __init__(
        self,
        dataset_string,
        model,
        use_lengths=True,
        gradient_params={
            "max_iterations": 10,
            "init_pattern": "moments",
            "num_restarts": 1,
        },
        run_meta="",
        phys_lb=None,
        phys_ub=None,
        samp_lb=None,
        samp_ub=None,
        gridsize=None,
        poisson_average_log_length=5
    ):
        """Initialize the InferenceParameters instance.

        Parameters
        ----------
        phys_lb: list of floats or float np.ndarray
            log10 lower bounds on biological parameters.
        phys_ub: list of floats or float np.ndarray
            log10 upper bounds on biological parameters.
        samp_lb: list of floats or float np.ndarray
            log10 lower bounds on technical variation parameters.
        samp_ub: list of floats or float np.ndarray
            log10 upper bounds on technical variation parameters.
        gridsize: list of ints or int np.ndarray
            grid size for evaluating the technical variation parameters.
        dataset_string: str
            dataset-specific directory location.
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        use_lengths: bool, optional
            if True, the nascent Poisson model technical variation parameter is a
            coefficient multiplied by gene length.
            if False, the parameter is the genome-wide nascent sampling rate.
        gradient_params: dict, optional
            settings for gradient descent.
            "max_iterations" defines the maximum number of gradient descent iterations.
            "init_pattern" defines whether the first try starts at the method of moments estimate.
            "num_restarts" defines how many attempts should be made.
        run_meta: str, optional
            any additional metadata to append to the run directory name.
        """
        # Set biophysical parameter values to defaults.
        if phys_lb is None:
            phys_lb = model.bio_bounds['phys_lb']
        if phys_ub is None:
            phys_ub = model.bio_bounds['phys_ub']

        # Set technical sequencing parameter values to defaults.
        if samp_lb is None:
            samp_lb = model.seq_bounds['samp_lb']
        if samp_ub is None:
            samp_ub = model.seq_bounds['samp_ub']
        if gridsize is None:
            gridsize = model.seq_bounds['gridsize']
        
        self.gradient_params = gradient_params
        self.phys_lb = np.array(phys_lb)
        self.phys_ub = np.array(phys_ub)
        self.grad_bnd = scipy.optimize.Bounds(phys_lb, phys_ub)

        self.use_lengths = use_lengths
        self.poisson_average_log_length = poisson_average_log_length

        if model.seq_model == "None":
            log.info(
                "Sequencing model set to None. All sampling parameters set to null."
            )
            samp_lb = [0, 0]
            samp_ub = [0, 0]
            gridsize = [1, 1]

        self.samp_lb = np.array(samp_lb)
        self.samp_ub = np.array(samp_ub)
        self.gridsize = gridsize

        self.construct_grid()
        self.model = model

        self.n_phys_pars = model.get_num_params()
        self.n_samp_pars = len(self.samp_ub)  # this will always be 2 for now

        if len(run_meta) > 0:
            run_meta = "_" + run_meta

        self.dataset_string = dataset_string
        
        inference_string = f"{dataset_string}/{model.bio_model}_{model.seq_model}_"
        for i in range(len(gridsize)):
            inference_string += f"{gridsize[i]:.0f}x"
        inference_string = inference_string[:-1]
        inference_string += f"{run_meta}"
        
        make_dir(inference_string)
        self.inference_string = inference_string
        inference_parameter_string = inference_string + "/parameters.pr"
        self.store_inference_parameters(inference_parameter_string)

    def construct_grid(self):
        """Creates a grid of points over the two-dimensional technical variation parameter domain.

        Sets
        ----
        sampl_vals: list of lists of floats
            list of grid points.
        grid_values_sampl: list of np.ndarrays
            grid point values representing sampling parameters for each modality.
        n_grid_pts: int
            total number of grid points to evaluate.
        """
        linspaces = [np.linspace(self.samp_lb[i], self.samp_ub[i], self.gridsize[i]) for i in range(len(self.gridsize))]
        grid_values_sampl = np.meshgrid(*linspaces, indexing="ij")
        
        grid_values_sampl = [i.flatten() for i in grid_values_sampl]
        self.grid_values_sampl = grid_values_sampl
        
        self.sampl_vals = list(zip(*grid_values_sampl))
        
        self.n_grid_points = len(grid_values_sampl[0])

    def store_inference_parameters(self, inference_parameter_string):
        """This helper method attempts to save the InferenceParameters object.

        Parameters
        ----------
        inference_parameter_string: str
            file location.
        """
        try:
            with open(inference_parameter_string, "wb") as ipfs:
                pickle.dump(self, ipfs)
            log.info(
                "Global inference parameters stored to {}.".format(
                    inference_parameter_string
                )
            )
        except:
            log.error(
                "Global inference parameters could not be stored to {}.".format(
                    inference_parameter_string
                )
            )

    def fit_all_grid_points(self, search_data, num_cores=1):
        """Fits the search data for all genes over all grid points.

        Parameters
        ----------
        num_cores: int
            number of cores to use for parallelization over grid points.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        -------
        results: SearchResults object
            Saves search_result to disk and returns it.

        """

        t1 = time.time()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if num_cores > 1:
            log.info("Starting parallelized grid scan.")
            parallelize(
                function=self.par_fun,
                iterable=zip(
                    range(self.n_grid_points),
                    [[search_data, self.model]] * self.n_grid_points,
                ),
                num_cores=num_cores,
                num_entries=self.n_grid_points,
                completion_message="Parallelized grid scan complete.",
                termination_message="The scan has been manually terminated.",
                error_message="The scan has been terminated due to computation issues. Please check MoM estimates.",
            )
        else:
            log.info("Starting non-parallelized grid scan.")
            [
                self.par_fun(x)
                for x in zip(
                    range(self.n_grid_points),
                    [[search_data, self.model]] * self.n_grid_points,
                )
            ]
            log.info("Non-parallelized grid scan complete.")

        # warnings.resetwarnings()
        results = SearchResults(self, search_data)
        results.aggregate_grid_points()

        full_result_string = results.store_on_disk()


        t2 = time.time()
        log.info("Runtime: {:.1f} seconds.".format(t2 - t1))
        
        return results

    def par_fun(self, inputs):
        """Helper method for the grid point parallelization procedure.

        Parameters
        ----------
        inputs: tuple
            entry 0: int
                point index within [0, n_grid_points) to evaluate at.
            entry 1: tuple
                entry 0: monod.extract_data.SearchData
                    SearchData object with the data to fit.
                entry 1: monod.cme_toolbox.CMEModel
                    CME model used for inference.
        """
        point_index, (search_data, model) = inputs
        grad_inference = GradientInference(self, model, search_data, point_index)
        grad_inference.fit_all_genes(model, search_data)


class GradientInference:
    """Runs the grid point-specific inference procedures.

    Attributes
    ----------
    grid_point: list of floats
        genome-wide technical variation parameter values at the current grid point.
    point_index: int
        the index of the current point, within [0, n_grid_points).
    regressor: np.ndarray
        gene-specific technical variation parameter values at the current grid point.
        these values will be different for each gene if use_lengths=True in the
        InferenceParameters constructor.
    grad_bnd: scipy.optimize.Bounds
        log10 lower and upper bounds on biological parameters.
    gradient_params: dict
        settings for gradient descent.
        "max_iterations" defines the maximum number of gradient descent iterations.
        "init_pattern" defines whether the first try starts at the method of moments estimate.
        "num_restarts" defines how many attempts should be made.
    phys_lb: float np.ndarray
        log10 lower bounds on biological parameters.
    phys_ub: float np.ndarray
        log10 upper bounds on biological parameters.
    n_phys_pars: int
        number of biological model parameters.
    n_samp_pars: int
        number of technical variation model parameters.
    inference_string: str
        run-specific directory location.
    param_MoM: np.ndarray
        method of moments estimates for all genes under the current technical variation parameters.
    """

    def __init__(self, global_parameters, model, search_data, point_index):
        """Initialize a GradientInference object.

        Parameters
        ----------
        global_parameters: InferenceParameters
            information about the global parameter inference procedure.
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.
        point_index: int
            the index of the current point, within [0, n_grid_points).

        Sets
        ----
        grid_point: list of floats
            genome-wide technical variation parameter values at the current grid point.
        point_index: int
            the index of the current point, within [0, n_grid_points).
        regressor: np.ndarray
            gene-specific technical variation parameter values at the current grid point.
            these values will be different for each gene if use_lengths=True in the
            InferenceParameters constructor.
        grad_bnd: scipy.optimize.Bounds
            log10 lower and upper bounds on biological parameters.
        gradient_params: dict
            settings for gradient descent.
            "max_iterations" defines the maximum number of gradient descent iterations.
            "init_pattern" defines whether the first try starts at the method of moments estimate.
            "num_restarts" defines how many attempts should be made.
        phys_lb: float np.ndarray
            log10 lower bounds on biological parameters.
        phys_ub: float np.ndarray
            log10 upper bounds on biological parameters.
        n_phys_pars: int
            number of biological model parameters.
        n_samp_pars: int
            number of technical variation model parameters.
        inference_string: str
            run-specific directory location.
        param_MoM: np.ndarray
            method of moments estimates for all genes under the current technical variation parameters.

        """
 
        regressor = np.array(
            [global_parameters.sampl_vals[point_index]] * search_data.n_genes
        )

        if global_parameters.use_lengths:
            if model.seq_model == "Bernoulli":
                raise ValueError(
                    "The Bernoulli model does not yet have a physical length-based model."
                )
            elif model.seq_model == "None":
                raise ValueError(
                    "The model without technical noise has no length effects."
                )
            elif model.seq_model == "Poisson":
                regressor[:, 0] += search_data.gene_log_lengths
            else:
                raise ValueError(
                    "Please select a technical noise model from {Poisson}, {Bernoulli}, {None}."
                )

        else:
            # If no specific lengths given, multiply the unspliced sampling rate by an average length value for all genes.
            if model.seq_model == "Poisson" and model.fit_unspliced:
                regressor[:, 0] += global_parameters.poisson_average_log_length
            
        self.grid_point = global_parameters.sampl_vals[point_index]
        self.point_index = point_index
        self.regressor = regressor
        self.grad_bnd = global_parameters.grad_bnd
        self.gradient_params = global_parameters.gradient_params
        self.phys_lb = global_parameters.phys_lb
        self.phys_ub = global_parameters.phys_ub
        self.n_phys_pars = global_parameters.n_phys_pars
        self.n_samp_pars = global_parameters.n_samp_pars

        self.inference_string = global_parameters.inference_string
        if self.gradient_params["init_pattern"] == "moments":
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.param_MoM = np.asarray(
                [
                    model.get_MoM(
                        search_data.moments[i],
                        global_parameters.phys_lb,
                        global_parameters.phys_ub,
                        regressor[i],
                    )
                    for i in range(search_data.n_genes)
                ]
            )
            warnings.resetwarnings()

    def optimize_gene(self, gene_index, model, search_data):
        """Fit the data for a single gene using KL divergence gradient descent.

        If init_pattern = moments, the first search's starting point is set to the
            method of moments parameter estimate for the current grid point.
        If num_restarts>1, the optimum is only updated if new KLD is lower
            than 0.99*previous lowest KLD.

        Parameters
        ----------
        gene_index: int
            index of the current gene, as defined by the gene_names attribute of search_data.
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        -------
        x: np.ndarray
            optimal biological parameter values.
        err: float
            Kullback-Leibler divergence of the model at x, relative to data.
        """
        # Further cap the random initializations within a smaller box.
        restricted_bounds_lb = np.array([0, -2,-2,-2,-2])
        restricted_bounds_ub = np.array([2,2,2,2,2])
        x0 = (
            np.random.rand(self.gradient_params["num_restarts"], self.n_phys_pars)
            * (restricted_bounds_ub - restricted_bounds_lb)
            + restricted_bounds_lb
        )
        # x0 = (
        #     np.random.rand(self.gradient_params["num_restarts"], self.n_phys_pars)
        #     * (self.phys_ub - self.phys_lb)
        #     + self.phys_lb
        # )
        if (
            self.gradient_params["init_pattern"] == "moments"
        ):  # this can be extended to other initialization patterns, like latin squares
            x0[0] = self.param_MoM[gene_index]
        #x = x0[0]
        err = np.inf
        ERR_THRESH = 0.99
        log.info('Optimizing gene %d with initial value %s', gene_index, np.array2string(10**x0))
        
        hist_type = get_hist_type(search_data)
        for restart in range(self.gradient_params["num_restarts"]):
            res_arr = scipy.optimize.minimize(
                lambda x: model.eval_model_kld(
                    p=x,  # limits=[search_data.M[gene_index],search_data.N[gene_index]],\
                    limits=search_data.M[:, gene_index],
                    samp=self.regressor[gene_index],
                    data=search_data.hist[gene_index],
                    hist_type=hist_type,
                ),
                x0=x0[restart],
                method='Nelder-Mead',
                bounds=self.grad_bnd,
                options={
                    "maxiter": self.gradient_params["max_iterations"],
                    "disp": False,
                },
            )
            if (
                res_arr.fun < err * ERR_THRESH
            ):  # do not replace old best estimate if there is little marginal benefit
                x = res_arr.x
                err = res_arr.fun
        if not (np.isfinite(x).all()):
            log.warning("Gene index: " + str(gene_index))
            raise ValueError("Search failed. Please check input data.")
        log.info('Optimized parameters for gene %d is %s', gene_index, np.array2string(10**x))
        return x, err

    def iterate_over_genes(self, model, search_data):
        """Run gradient descent for every gene at the current grid point.

        Parameters
        ----------
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        Returns
        -------
        param_estimates: np.ndarray
            optimal biological parameter values for each gene, an n_genes x n_phys_pars array.
        klds: np.ndarray
            Kullback-Leibler divergence of the model for each gene at param_estimates.
        obj_func: float
            sum of klds; total error at the current grid point.
        d_time: float
            runtime in seconds.
        """
        t1 = time.time()
        
        param_estimates, klds = zip(
            *[
                self.optimize_gene(gene_index=gene_index, model=model, search_data=search_data)
                for gene_index in range(search_data.n_genes)
            ]
        )

        klds = np.asarray(klds)
        param_estimates = np.asarray(param_estimates)
        obj_func = klds.sum()

        t2 = time.time()
        d_time = t2 - t1

        return param_estimates, klds, obj_func, d_time

    def fit_all_genes(self, model, search_data):
        """Wraps iterate_over_genes and stores the results on disk.

        Parameters
        ----------
        model: monod.cme_toolbox.CMEModel
            CME model used for inference.
        search_data: monod.extract_data.SearchData
            SearchData object with the data to fit.

        """
        
        search_out = self.iterate_over_genes(model=model, search_data=search_data)
        results = GridPointResults(
            *search_out,
            self.regressor,
            self.grid_point,
            self.point_index,
            self.inference_string,
        )
        results.store_grid_point_results()

########################
## Helper functions
########################


def get_hist_type_adata(monod_adata):
    """A helper function for backwards compatibility.

    If the histogram type is not specified in the anndata object, assume it is the legacy
    type "grid".

    Parameters
    ----------
    monod_adata: Anndata
        anndata object with the data to fit.

    Returns
    -------
    hist_type: str
        flavor of histogram used to generate monod_adata, either "unique" or "grid" or "none".
    """
    
    if "hist_type" in monod_adata.uns and monod_adata.uns["hist_type"] in {"unique", "none"}:
        hist_type = monod_adata.uns["hist_type"]
        
    else:
        hist_type = "grid"
        
    return hist_type


########################
## Helper classes
########################
class GridPointResults:
    """Temporarily stores the fit parameters for a single grid point.

    Attributes
    ----------
    param_estimates: np.ndarray
        optimal biological parameter values for each gene, an n_genes x n_phys_pars array.
    klds: np.ndarray
        Kullback-Leibler divergence of the model for each gene at param_estimates.
    obj_func: float
        sum of klds; total error at the current grid point.
    d_time: float
        runtime in seconds.
    regressor: np.ndarray
        gene-specific technical variation parameter values at the current grid point.
        these values will be different for each gene if use_lengths=True in the
        InferenceParameters constructor.
    grid_point: list of floats
        genome-wide technical variation parameter values at the current grid point.
    point_index: int
        the index of the current point, within [0, n_grid_points).
    inference_string: str
        run-specific directory location.

    """

    def __init__(
        self,
        param_estimates,
        klds,
        obj_func,
        d_time,
        regressor,
        grid_point,
        point_index,
        inference_string,
    ):
        """Creates a GridPointResults object and sets all of its attributes."""
        self.param_estimates = param_estimates
        self.klds = klds
        self.obj_func = obj_func
        self.d_time = d_time
        self.regressor = regressor
        self.grid_point = grid_point
        self.point_index = point_index
        self.inference_string = inference_string

    def store_grid_point_results(self):
        """This helper method attempts to store the grid point results to disk as a grid_point_X.gp object."""
        try:
            grid_point_result_string = (
                self.inference_string + "/grid_point_" + str(self.point_index) + ".gp"
            )
            with open(grid_point_result_string, "wb") as gpfs:
                pickle.dump(self, gpfs)
            log.debug(
                "Grid point {:.0f} results stored to {}.".format(
                    self.point_index, grid_point_result_string
                )
            )
        except:
            log.error(
                "Grid point {:.0f} results could not be stored to {}.".format(
                    self.point_index, grid_point_result_string
                )
            )


class SearchResults:
    """Stores and analyzes the results of a single inference run.

    The first thirteen attributes relate to data loaded from the search.
    The others relate to data processing after the search is completed.

    Attributes
    ----------
    sp: InferenceParameters
        search parameters used to generate the run.
    inference_string: str
        run-specific directory location.
    model: monod.cme_toolbox.CMEModel
        CME model used for inference.
    n_genes: int
        number of analyzed genes.
    n_cells: int
        number of cells in the dataset.
    gene_log_lengths: float np.ndarray
        log lengths of analyzed genes.
    gene_names: str np.ndarray
        list of analyzed genes.
    param_estimates: float np.ndarray
        optimal biological parameter values for each gene, an n_grid_pts x n_genes x n_phys_pars array.
    klds: float np.ndarray
        Kullback-Leibler divergence of the model for each gene at param_estimates, an n_grid_pts x n_genes array.
    obj_func: float np.ndarray
        sum of klds at each grid points; total error at the current grid point, a length-n_grid_pts array.
    d_time: float np.ndarray
        runtime in seconds for each grid point, a length-n_grid_pts array.
    regressor: float np.ndarray
        gene-specific technical variation parameter values at each grid point.
        an n_grid_pts x n_genes array.
        these values will be different for each gene if use_lengths=True in the
        InferenceParameters constructor.
    analysis_figure_string: str
        directory for analysis figures.

    samp_optimum: list of floats
        estimated value of the technical noise parameters.
    samp_optimum_ind: int
        index of the sampling parameter optimum grid point.
    phys_optimum: float np.ndarray
        gene-specific physical parameter values at the sampling parameter optimum.
    regressor_optimum: float np.ndarray
        gene-specific technical variation parameter values at the sampling parameter optimum.
    csq: np.ndarray
        chi-squared statistics for all genes, computed at the grid point indexed by rejection_index.
    pval: np.ndarray
        p-values calculated by the chi-squared test at the grid point indexed by rejection_index.
    rejected_genes: bool np.ndarray
        a boolean filter that reports the genes rejected by the goodness-of-fit procedure,
        whose parameters cannot be safely interpeted.
    rejection_index: int
        the grid point at which the goodness-of-fit procedure was performed to generate
        the rejected_genes attribute.
    sigma: float np.ndarray
        the standard error of the parameter maximum likelihood estimate at the at the grid
        point indexed by sigma_index.
        a n_genes x n_phys_pars array.
    sigma_index: int
        the grid point at which the Fisher information procedure was performed to generate
        the sigma attribute.
    batch_analysis_string: str
        location of the directory for batch-wide analyses.
    """

    ####################################
    #   Construction and I/O methods   #
    ####################################
    def __init__(self, inference_parameters, search_data):
        """Creates a SearchResults object.

        Parameters
        ----------
        inference_parameters: InferenceParameters
            search parameters used to generate the run.
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        """
        # pull in info from search parameters.
        self.sp = inference_parameters

        self.inference_string = inference_parameters.inference_string
        self.model = inference_parameters.model

        # pull in small amount of non-cell-specific info from search data
        self.n_genes = search_data.n_genes
        self.n_cells = search_data.n_cells
        try:
            self.gene_log_lengths = search_data.gene_log_lengths
        # Do not define an attribute for gene_lengths if non is given.
        except AttributeError:
            pass
            
        self.gene_names = search_data.gene_names

        self.param_estimates = []
        self.klds = []
        self.obj_func = []
        self.d_time = []
        self.regressor = []

    def aggregate_grid_points(self):
        """This helper method concatenates all of the grid point results.

        The method runs append_grid_point for all grid points, then removes the original grid point files.
        """
        for point_index in range(self.sp.n_grid_points):
            self.append_grid_point(point_index)
        self.clean_up()

    def append_grid_point(self, point_index):
        """This helper method updates the result attributes from a GridPointResult object stored on disk.

        Parameters
        ----------
        point_index: int
            index of the grid point results to load from disk.
        """
        grid_point_result_string = (
            self.inference_string + "/grid_point_" + str(point_index) + ".gp"
        )
        with open(grid_point_result_string, "rb") as ipfs:
            grid_point_results = pickle.load(ipfs)
            self.param_estimates += [grid_point_results.param_estimates]
            self.klds += [grid_point_results.klds]
            self.obj_func += [grid_point_results.obj_func]
            self.d_time += [grid_point_results.d_time]
            self.regressor += [grid_point_results.regressor]

    def clean_up(self):
        """This helper method removes temporary files and finalizes the SearchResults object.

        The GridPointResult objects are erased from disk, the attributes are converted to
        np.ndarrays, and a directory for analysis figures is created.
        """
        for point_index in range(self.sp.n_grid_points):
            os.remove(self.inference_string + "/grid_point_" + str(point_index) + ".gp")
        log.info("All grid point data cleaned from disk.")
        self.param_estimates = np.asarray(self.param_estimates)
        self.klds = np.asarray(self.klds)
        self.obj_func = np.asarray(self.obj_func)
        self.d_time = np.asarray(self.d_time)
        self.regressor = np.asarray(self.regressor)

        analysis_figure_string = self.inference_string + "/analysis_figures"
        self.analysis_figure_string = analysis_figure_string
        make_dir(analysis_figure_string)

    def store_on_disk(self):
        """This helper method attempts to store the SearchResults object to disk.

        Returns
        -------
        full_result_string: str
            file location.
        """
        try:
            full_result_string = self.inference_string + "/grid_scan_results.res"
            with open(full_result_string, "wb") as srfs:
                pickle.dump(self, srfs)
            log.info("Grid scan results stored to {}.".format(full_result_string))
        except:
            log.error(
                "Grid scan results could not be stored to {}.".format(
                    full_result_string
                )
            )
        self.full_result_string = full_result_string
        return full_result_string

    def update_on_disk(self):
        """This helper method attempts to store a modified SearchResults object to disk.

        This is separate from store_on_disk() to avoid overwriting data.

        Returns
        -------
        upd_result_string: str
            file location.
        """
        try:
            upd_result_string = self.inference_string + "/grid_scan_results_upd.res"
            with open(upd_result_string, "wb") as srfs:
                pickle.dump(self, srfs)
            log.debug("Updated results stored to {}.".format(upd_result_string))
        except:
            log.error(
                "Updated results could not be stored to {}.".format(upd_result_string)
            )
        self.upd_result_string = upd_result_string
        return upd_result_string

    ####################################
    #         Analysis methods         #
    ####################################

    def find_sampling_optimum(self, gene_filter=None, discard_rejected=False):
        """Identify and set the technical parameter optimum by minimizing the total KLD.

        Parameters
        ----------
        gene_filter: None or np.ndarray, optional
            If None, use all genes.
            If a boolean or integer filter, use the filtered gene subset.
        discard_rejected: bool, optional
            whether to omit genes in the rejected_genes attribute during the calculation.

        Returns
        -------
        samp_optimum: list of floats
            estimated value of the technical noise parameters.
        """
        if gene_filter is None:
            total_divergence = self.obj_func
        else:
            gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
            total_divergence = self.klds[:, gene_filter].sum(1)
        samp_optimum_ind = np.argmin(total_divergence)
        self.set_sampling_optimum(samp_optimum_ind)
        return self.samp_optimum

    def set_sampling_optimum(self, samp_optimum_ind):
        """Set the technical parameter optimum to a specific grid point index.

        Define the value, then update the samp_optimum_ind, samp_optimum,
        phys_optimum, and regressor_optimum attributes that depend on this value.

        Parameters
        ----------
        samp_optimum_ind: int
            index of the grid point.

        Returns
        -------
        samp_optimum: list of floats
            value of the technical noise parameters at the grid point.
        """
        self.samp_optimum_ind = samp_optimum_ind
        self.samp_optimum = self.sp.sampl_vals[samp_optimum_ind]
        self.phys_optimum = self.param_estimates[samp_optimum_ind]
        self.regressor_optimum = self.regressor[samp_optimum_ind]
        return self.samp_optimum

    def plot_landscape(
        self,
        ax,
        plot_optimum=True,
        gene_filter=None,
        discard_rejected=False,
        logscale=True,
        colorbar=False,
        hideticks=False,
        savefig=False,
    ):
        """Plot the 2D Kullback-Leibler divergence (KLD) landscape over the evaluated grid points.

        The landscape is computed by adding together gene-specific values at each grid points,
        potentially only over a subset of the genes.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axes to plot into.
        plot_optimum: bool, optional
            whether to plot a dot at the estimated optimum.
        gene_filter: bool, optional
            If None, plot landscape for all genes.
            If a boolean or integer filter, plot lanscape determined by the filtered gene subset.
        discard_rejected: bool, optional
             whether to omit genes in the rejected_genes attribute during the calculation.
        logscale: bool, optional
            whether to show log10 of the total KLD instead of the raw value.
        colorbar: bool, optional
            whether to display a colobrar.
        hideticks: bool, optional
            whether to hide the ticks and coordinates around the plot.
        savefig: bool, optional
            whether to save the figure to disk.
        """

        # if gene_filter is None:
        #     total_divergence = self.obj_func
        # else:
        #     gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
        gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
        total_divergence = self.klds[:, gene_filter].sum(1)

        if logscale:
            total_divergence = np.log10(total_divergence)

        dx = (np.asarray(self.sp.samp_ub) - np.asarray(self.sp.samp_lb)) / (
            np.asarray(self.sp.gridsize) - 1
        )
        dx[dx < 1e-10] = 0.1
        extent = [
            self.sp.samp_lb[0] - dx[0] / 2,
            self.sp.samp_ub[0] + dx[0] / 2,
            self.sp.samp_lb[1] - dx[1] / 2,
            self.sp.samp_ub[1] + dx[1] / 2,
        ]
        lnd = ax.imshow(
            np.flipud(np.reshape(total_divergence, self.sp.gridsize).T), extent=extent
        )

        if plot_optimum:
            ax.scatter(self.samp_optimum[0], self.samp_optimum[1], c="crimson", s=50)
        if colorbar:
            plt.colorbar(lnd, ax=ax)
        if hideticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if savefig:
            fig_string = self.analysis_figure_string + "/landscape.png"
            plt.savefig(fig_string, dpi=450)
            log.info("Figure stored to {}.".format(fig_string))

    def get_bool_filt(self, gene_filter, discard_rejected):
        """This helper method constructs gene filters.

        The method intersects an arbitrary (potentially None, boolean, or integer) gene selection with the
        rejected_genes attribute (a boolean filter) to produce an overall set of genes to be analyzed or visualized.

        Parameters
        ----------
        gene_filter: None, bool np.ndarray, or int np.ndarray
            if an array, select the genes in the array,
            if None, consider all genes.
        discard_rejected: bool
            if True, omit genes that are rejected by the goodness-of-fit procedure.

        Returns
        -------
        gene_filter: bool np.ndarray
            genes that are retained by the filter.
        """

        if gene_filter is None:
            gene_filter = np.ones(self.n_genes, dtype=bool)
        else:
            if gene_filter.dtype is not bool:
                gf_temp = np.zeros(self.n_genes, dtype=bool)
                gf_temp[gene_filter] = True
                gene_filter = gf_temp

        if discard_rejected:
            if hasattr(self, "rejection_index"):
                if self.rejection_index != self.samp_optimum_ind:
                    raise ValueError("Sampling parameter value is inconsistent.")
                gene_filter = np.logical_and(~self.rejected_genes, gene_filter)
            else:
                log.info("No rejection statistics have been computed.")
        return gene_filter

    def plot_param_marg(
        self,
        gene_filter=None,
        discard_rejected=True,
        nbin=15,
        fitlaw=scipy.stats.norminvgauss,
        axis_search_bounds=True,
        figsize=None,
    ):
        """Plot and fit the biological parameter distributions at the sampling parameter optimum.

        Parameters
        ----------
        gene_filter: None or np.ndarray, optional
            If None, plot all genes.
            If a boolean or integer filter, plot the filtered gene subset.
        discard_rejected: bool, optional
             whether to omit genes in the rejected_genes attribute.
        nbin: int, optional
            number of bins used to construct the histogram.
        fitlaw: scipy.stats.rv_continuous, optional
            statistical law used to fit the parameter distributions.
        axis_search_bounds: bool, optional
            whether to place the x-limits of the plots at the parameter search bounds.
        figsize: tuple or None, optional
            figure dimensions.
        """
        num_params = self.sp.n_phys_pars
        figsize = figsize or (4 * num_params, 4)
        fig1, ax1 = plt.subplots(nrows=1, ncols=num_params, figsize=figsize)

        # identify genes to plot, extract their data
        gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
        param_data = self.phys_optimum[gene_filter, :]

        for i in range(num_params):
            ax1[i].hist(
                param_data[:, i],
                nbin,
                density=True,
                color=aesthetics["hist_face_color"],
            )
            if fitlaw is not None:
                fitparams = fitlaw.fit(param_data[:, i])

                xmin, xmax = ax1[i].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = fitlaw.pdf(x, *fitparams)
                ax1[i].plot(
                    x,
                    p,
                    "--",
                    linewidth=aesthetics["hist_fit_lw"],
                    color=aesthetics["hist_fit_color"],
                )

            if axis_search_bounds:
                ax1[i].set_xlim([self.sp.phys_lb[i], self.sp.phys_ub[i]])
            ax1[i].set_title(self.model.get_log_name_str()[i])
            ax1[i].set_xlabel(r"$log_{10}$ value")
        fig1.tight_layout()
        fig_string = self.analysis_figure_string + "/parameter_marginals.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))

    def plot_KL(self, ax, gene_filter=None, discard_rejected=True, nbin=15):
        """Plot the distribution of KL divergences at the sampling parameter optimum.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axes to plot into.
        gene_filter: None or np.ndarray, optional
            If None, plot all genes.
            If a boolean or integer filter, plot the filtered gene subset.
        discard_rejected: bool, optional
             whether to omit genes in the rejected_genes attribute.
        nbin: int, optional
            number of bins used to construct the histogram.
        """
        gene_filter = self.get_bool_filt(gene_filter, discard_rejected)
        kld_data = self.klds[self.samp_optimum_ind][gene_filter]

        ax.hist(kld_data, nbin, color=aesthetics["hist_face_color"])
        ax.set_xlabel("KL divergence")
        ax.set_ylabel("# genes")
        fig_string = self.analysis_figure_string + "/kldiv.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))

    def chisquare_testing(
        self,
        search_data,
        viz=False,
        EPS=1e-15,
        threshold=0.05,
        bonferroni=True,
        reject_at_bounds=True,
        bound_thr=0.01,
        grouping_thr=5,
        use_hellinger=True,
        hellinger_thr=0.05,
    ):
        """Perform goodness-of-fit testing at the current sampling parameter optimum to identify poor fits.

        This method performs two rounds of goodness-of-fit testing.
        First, it applies a chi-squared test to the distributions induced by biological parameter values
        at the sampling parameter optimum.
        Optionally, it also rejects genes that are too close to the search parameter bounds,
        as they typically exhibit poor gradient descent performance or do not have enough counts to
        reliably estimate parameters.
        This is typically sufficient to reject genes with proposed distributions that are grossly
        dissimilar to the raw data, whether due to inference failure or model misspecification.

        We typically expect about 5-15% of the genes to be rejected.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        viz: bool, optional
            whether to visualize the histogram of the chi-square statistic.
        EPS: float, optional
            probability rounding parameter: anything below this value is rounded to EPS.
        threshold: float, optional
            chi-square rejection criterion; everything below this critical p-value
            is rejected as unlikely to have been generated by the model.
        bonferroni: float, optional
            whether to apply the Bonferroni correction to the p-value threshold.
        reject_at_bounds: bool, optional
            whether to also discard genes that are near the search bounds.
        bound_thr: float, optional
            how close the parameters must be close to the bounds (in units of the allowed
            parameter range) to trigger rejection.
        grouping_thr: float or int, optional
            minimum bin size for chi-squared test.
        use_hellinger: bool, optional
            whether to use the Hellinger distance as an additional "effect size" estimate for goodness-of-fit.
        hellinger_thr: float, optional
            which threshold to use for the Hellinger distance rejection.

        Returns
        -------
        csq: np.ndarray
            chi-squared statistics for all genes, computed at the grid point indexed by rejection_index.
        pval: np.ndarray
            p-values calculated by the chi-squared test at the grid point indexed by rejection_index.
        hellinger: np.ndarray
            hellinger distances for all genes, computed at the grid point indexed by rejection_index.

        Sets
        ----
        csq: np.ndarray
            chi-squared statistics for all genes, computed at the grid point indexed by rejection_index.
        pval: np.ndarray
            p-values calculated by the chi-squared test at the grid point indexed by rejection_index.
        hellinger: np.ndarray
            hellinger distances for all genes, computed at the grid point indexed by rejection_index.
        rejected_genes: bool np.ndarray
            a boolean filter that reports the genes rejected by the goodness-of-fit procedure,
            whose parameters cannot be safely interpeted.
        rejection_index: int
            the grid point at which the goodness-of-fit procedure was performed to generate
            the rejected_genes attribute.

        """
        t1 = time.time()
        hist_type = get_hist_type(search_data)

        csqarr = []
        hellinger = []
        for gene_index in range(self.n_genes):
            # lm = search_data.M[:, gene_index]
            lm = search_data.M[:, gene_index]
            expect_freq = (
                self.model.eval_model_pss(
                    self.phys_optimum[gene_index],
                    lm,
                    self.regressor_optimum[gene_index],
                )
                * search_data.n_cells
            )
            # expected_freq[expected_freq < EPS] = EPS
            # expected_freq /= expected_freq.sum()
            # PROPOSAL = search_data.n_cells * expected_freq

            if hist_type == "grid":
                raise ValueError("Not implemented in current version.")
            elif hist_type == "unique":
                counts = np.concatenate(
                    (search_data.n_cells * search_data.hist[gene_index][1], [0])
                )
                # print(search_data.hist)
                number_of_modalities = len(search_data.hist[gene_index][0][0])
                if number_of_modalities == 2:
                    expect_freq = expect_freq[
                        search_data.hist[gene_index][0][:, 0],
                        search_data.hist[gene_index][0][:, 1],
                    ]
                elif number_of_modalities == 3:
                    expect_freq = expect_freq[
                        search_data.hist[gene_index][0][:, 0],
                        search_data.hist[gene_index][0][:, 1],
                        search_data.hist[gene_index][0][:, 2]
                    ]

                # expect_freq = expect_freq[ [
                #     search_data.hist[gene_index][0][:, i] for i in range(len(search_data.hist[gene_index][0][0]))
                # ]]

                expect_freq = np.concatenate(
                    (expect_freq, [search_data.n_cells - expect_freq.sum()])
                )

            hellinger_ = (
                1
                / np.sqrt(2)
                * (
                    (
                        np.sqrt(expect_freq / search_data.n_cells)
                        - np.sqrt(counts / search_data.n_cells)
                    )
                    ** 2
                ).sum()
            )

            bins = []
            bin_ind = 0
            run_bin_obs = 0
            run_bin_exp = 0
            bin_obs = []
            bin_exp = []
            for i in range(len(counts)):
                bins.append(bin_ind)
                run_bin_obs += counts[i]
                run_bin_exp += expect_freq[i]
                if min(run_bin_obs, run_bin_exp) < 5:  # and i
                    pass
                else:
                    bin_ind += 1
                    bin_obs.append(run_bin_obs)
                    bin_exp.append(run_bin_exp)
                    run_bin_obs = 0
                    run_bin_exp = 0
            bins = np.asarray(bins)
            observed = np.asarray(bin_obs)
            proposed = np.asarray(bin_exp)
            observed[-1] += run_bin_obs
            proposed[-1] += run_bin_exp
            bins[bins == len(observed)] = len(observed) - 1

            for b_ in range(len(bin_obs)):
                assert np.isclose(observed[b_], counts[bins == b_].sum())
                assert np.isclose(proposed[b_], expect_freq[bins == b_].sum())
            assert np.isclose(observed.sum(), search_data.n_cells)
            assert np.isclose(proposed.sum(), search_data.n_cells)
            assert np.isclose(search_data.n_cells, counts.sum())
            assert np.isclose(search_data.n_cells, expect_freq.sum())


            # Ensure the sums of f_obs and f_exp match
            f_obs_sum = observed.sum()
            f_exp_sum = proposed.sum()
            
            # Check if normalization is needed
            if not np.isclose(f_obs_sum, f_exp_sum, rtol=1e-9):
                # Print a warning/info message
                warnings.warn(
                    f"The sums of observed and expected frequencies differ by {f_obs_sum - f_exp_sum:.2e}. "
                    "Normalizing expected frequencies to match observed frequencies.",
                    UserWarning,
                )
                # Normalize f_exp
                proposed = proposed * (f_obs_sum / f_exp_sum)


            # chi-squared takes only the number of biological parameters?
            csqarr += [
                scipy.stats.mstats.chisquare(
                    observed,
                    proposed,  # chisq_data, chisq_prop,
                    self.model.get_num_params(),
                )
            ]

            hellinger.append(hellinger_)

        csq, pval = zip(*csqarr)
        csq = np.asarray(csq)
        pval = np.asarray(pval)
        hellinger = np.asarray(hellinger)

        log.info('P-value threshold: ' + str(threshold) + ', Adjusted P-value threshold:' + str(threshold)+ ', Hellinger Threshold:' + str(hellinger_thr))
        
        if bonferroni:
            threshold /= self.n_genes
        self.rejected_genes = pval < threshold
        if use_hellinger:
            rej_hellinger = hellinger > hellinger_thr
            self.rejected_genes = (self.rejected_genes) & rej_hellinger

        if reject_at_bounds:
            bound_range = self.sp.phys_ub - self.sp.phys_lb
            lb = self.sp.phys_lb + bound_range * bound_thr
            ub = self.sp.phys_ub - bound_range * bound_thr

            rej_at_bounds = ((self.phys_optimum < lb) | (self.phys_optimum > ub)).any(1)
            self.rejected_genes = (self.rejected_genes) | rej_at_bounds

        self.pval = pval
        self.csq = csq
        self.hellinger = hellinger
        self.rejection_index = self.samp_optimum_ind  # mostly for debug.

        if viz:
            plt.hist(csq)
            plt.xlabel("Chi-square statistic")
            plt.ylabel("# genes")
            fig_string = self.analysis_figure_string + "/chisquare.png"
            plt.savefig(fig_string, dpi=450)
            log.info("Figure stored to {}.".format(fig_string))

        t2 = time.time()
        log.info(
            "Chi-square computation complete. Rejected {:.0f} genes out of {:.0f}. Runtime: {:.1f} seconds.".format(
                np.sum(self.rejected_genes), self.n_genes, t2 - t1
            )
        )
        return (csq, pval, hellinger)

    def par_fun_hess(self, inputs):
        """Helper method for the Hessian parallelization procedure.

        Parameters
        ----------
        inputs: tuple
            entry 0: int
                gene index within [0, n_genes) to evaluate at.
            entry 1: monod.extract_data.SearchData
                SearchData object with the fit data.

        Returns
        -------
        hess: float np.ndarray
            Hessian of the current gene's KLD at the sampling parameter optimum and the
            corresponding biological parameters, evaluated with respect to the
            biological parameters.
        """
        import numdifftools  # this will fail if numdifftools has not been evaluted.

        gene_index, search_data = inputs
        hist_type = get_hist_type(search_data)
        # if hasattr(search_data, "hist_type") and search_data.hist_type == "unique":
        #     hist_type = "unique"
        # else:
        #     hist_type = "grid"
        Hfun = numdifftools.Hessian(
            lambda x: self.model.eval_model_kld(
                p=x,
                limits=search_data.M[:, gene_index],
                samp=self.regressor_optimum[gene_index],
                data=search_data.hist[gene_index],
                hist_type=hist_type,
            )
        )
        hess = Hfun(self.phys_optimum[gene_index])
        
        return hess

    def compute_sigma(self, search_data, num_cores=1):
        """Estimate uncertainty in biological parameter values.

        This method iterates over genes, computes the Fisher information matrix by inverting
        the Hessian with respect to biological parameters, and reports the local estimate
        of standard errors of the parameters.

        This method is fairly computationally demanding, so parallelization is recommended.

        If inversion fails, the gene-specific standard errors are replaced with their mean over
        all genes that did not fail.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        num_cores: int, optional
            number of cores to use for parallelization over genes.

        Sets
        ----
        sigma: float np.ndarray
            the standard error of the parameter maximum likelihood estimate at the sampling
            parameter optimum.
            a n_genes x n_phys_pars array.
        sigma_index: int
            the grid point at which the Fisher information procedure was performed to generate
            the sigma attribute.

        """
        log.info("Computing local Hessian.")
        t1 = time.time()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if num_cores > 1:
            log.info("Starting parallelized Hessian computation.")
            hess = parallelize(
                function=self.par_fun_hess,
                iterable=zip(range(self.n_genes), [search_data] * self.n_genes),
                num_cores=num_cores,
                num_entries=self.n_genes,
                completion_message="Parallelized Hessian computation complete.",
                termination_message="The Hessian computation has been manually terminated.",
                error_message="The Hessian computation has been terminated due to numerical issues.",
                use_tqdm=False,
            )
        else:
            log.info("Starting non-parallelized Hessian computation.")
            hess = [
                self.par_fun_hess(x)
                for x in zip(range(self.n_genes), [search_data] * self.n_genes)
            ]
            log.info("Non-parallelized Hessian computation complete.")
        warnings.resetwarnings()
        hess = np.asarray(hess)

        fail = np.zeros(self.n_genes, dtype=bool)
        sigma = np.zeros((self.n_genes, self.sp.n_phys_pars))

        for gene_index in range(self.n_genes):
            try:

                hess_inv = np.linalg.inv(hess[gene_index, :, :])
                sigma[gene_index, :] = np.sqrt(np.diag(hess_inv)) / np.sqrt(
                    self.n_cells
                )
            except:
                fail[gene_index] = True
                log.info(
                    "Gene {:.0f} ran into singularity; replaced with mean. (Search converged to local minimum?) ".format(
                        gene_index
                    )
                )
                # errorbars[i,:] = np.mean(errorbars[:i,:])
            if np.any(~np.isfinite(sigma[gene_index, :])):
                fail[gene_index] = True
                log.info(
                    "Gene {:.0f} gives negative stdev; replaced with mean. (Search converged to local minimum?)".format(
                        gene_index
                    )
                )
                # errorbars[i,:] = np.mean(errorbars[:i,:])
        sigma[fail, :] = sigma[~fail, :].mean(0)
        self.sigma = sigma
        self.sigma_index = self.samp_optimum_ind  # mostly for debug

        t2 = time.time()
        log.info(
            "Standard error of the MLE computation complete. Runtime: {:.1f} seconds.".format(
                t2 - t1
            )
        )

    def resample_opt_viz(
        self, resamp_vec=(5, 10, 20, 40, 60), Ntries=4, figsize=(10, 10)
    ):
        """Test the sensitivity of the technical noise parameter landscape to the number of genes analyzed.

        Parameters
        ----------
        resamp_vec: list or tuple of ints, optional
            number of genes to select for downsampling (without replacement).
        Ntries: int, optional
            number of times to resample.
        figsize: tuple of floats, optional
            figure dimensions.
        """
        Nsamp = len(resamp_vec)

        fig1, ax1 = plt.subplots(nrows=Nsamp, ncols=Ntries, figsize=figsize)
        for samp_num in range(Nsamp):
            for i_ in range(Ntries):
                axloc = (samp_num, i_)
                if resamp_vec[samp_num] <= self.n_genes:
                    gene_filter = np.random.choice(
                        self.n_genes, resamp_vec[samp_num], replace=False
                    )
                else:
                    gene_filter = np.random.choice(
                        self.n_genes, self.n_genes, replace=False
                    )
                subsampled_samp_optimum = self.find_sampling_optimum(gene_filter)
                self.plot_landscape(ax1[axloc], gene_filter=gene_filter, hideticks=True)

                if i_ == 0:
                    ax1[axloc].set_ylabel("n_genes = " + str(resamp_vec[samp_num]))

        fig_string = self.analysis_figure_string + "/subsampling.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))
        self.find_sampling_optimum()  # reset sampling optimum here

    def resample_opt_mc_viz(
        self, resamp_vec=(5, 10, 20, 40, 60), Ntries=1000, figsize=(16, 4)
    ):
        """Test the sensitivity of technical noise parameter optima under gene downsampling.

        The optima for each downsampled set are visualized on the parameter landscape
        generated from the entire gene set.

        Parameters
        ----------
        resamp_vec: list or tuple of ints, optional
            number of genes to select for downsampling (without replacement).
        Ntries: int, optional
            number of times to resample.
        figsize: tuple of floats, optional
            figure dimensions.
        """
        Nsamp = len(resamp_vec)

        fig1, ax1 = plt.subplots(nrows=1, ncols=Nsamp, figsize=figsize)
        for samp_num in range(Nsamp):
            axloc = samp_num
            subsampled_samp_optimum_array = []
            for i__ in range(Ntries):
                if resamp_vec[samp_num] <= self.n_genes:
                    gene_filter = np.random.choice(
                        self.n_genes, resamp_vec[samp_num], replace=False
                    )
                else:
                    gene_filter = np.random.choice(
                        self.n_genes, self.n_genes, replace=False
                    )
                subsampled_samp_optimum = self.find_sampling_optimum(gene_filter)
                subsampled_samp_optimum_array.append(subsampled_samp_optimum)
            subsampled_samp_optimum_array = np.asarray(subsampled_samp_optimum_array)

            self.plot_landscape(ax1[axloc], hideticks=True)
            jit = np.random.normal(scale=0.1, size=subsampled_samp_optimum_array.shape)
            subsampled_samp_optimum_array = subsampled_samp_optimum_array + jit
            ax1[axloc].scatter(
                subsampled_samp_optimum_array[:, 0],
                subsampled_samp_optimum_array[:, 1],
                c="r",
                s=3,
                alpha=0.3,
            )
            ax1[axloc].set_title("n_genes = " + str(resamp_vec[samp_num]))

        fig_string = self.analysis_figure_string + "/subsampling_stability.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))

        self.find_sampling_optimum()  # reset sampling optimum.

    def chisq_best_param_correction(
        self,
        search_data,
        threshold=0.05,
        Ntries=10,
        viz=True,
        szfig=(2, 5),
        figsize=(10, 4),
        overwrite=True,
    ):
        """Test the sensitivity of technical noise parameter optima after gene rejection.

        This method demonstrates the sensitivity of the sampling parameter landscape
        and optimum to the specific genes retained after chi-squared testing.
        It performs fixed-point iteration to illustrate whether the optimum converges.

        This can be used with viz=True to inspect the qualitative behavior of the optimum,
        or with viz=False as a sanity check to make sure the optimum is not strongly skewed
        by a small number of very poorly fit genes.

        The procedure does not typically move the optimum.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        threshold: float, optional
            chi-square rejection criterion; everything below this critical p-value
            is rejected as unlikely to have been generated by the model.
        Ntries: int, optional
            number of steps of chi-squared testing to perform.
        viz: bool, optional
            whether to visualize the results.
        szfig: tuple of ints, optional
            dimensions of the figure subplot grid.
        figsize: tuple of floats, optional
            figure dimensions.
        overwrite: bool, optional
            whether to retain the optimum obtained at the end of the procedure.
        """
        if viz:
            fig1, ax1 = plt.subplots(nrows=szfig[0], ncols=szfig[1], figsize=figsize)
        log.info(
            "Original optimum: {:.2f}, {:.2f}.".format(
                self.samp_optimum[0], self.samp_optimum[1]
            )
        )
        for i_ in range(Ntries):
            self.chisquare_testing(search_data, threshold=threshold)
            # gene_filter = ~self.rejected_genes
            well_fit_samp_optimum = self.find_sampling_optimum(discard_rejected=True)
            log.info(
                "New optimum: {:.2f}, {:.2f}.".format(
                    self.samp_optimum[0], self.samp_optimum[1]
                )
            )

            if viz:
                axloc = (
                    np.unravel_index(i_, szfig)
                    if (szfig[0] > 1 and szfig[1] > 1)
                    else i_
                )
                self.plot_landscape(ax1[axloc], discard_rejected=True, hideticks=True)
        if viz:
            fig_string = self.analysis_figure_string + "/chisquare_stability.png"
            plt.savefig(fig_string, dpi=450)
            log.info("Figure stored to {}.".format(fig_string))

        if overwrite:
            self.chisquare_testing(search_data, threshold=threshold)
            log.info(
                "Optimum retained at {:.2f}, {:.2f}.".format(
                    self.samp_optimum[0], self.samp_optimum[1]
                )
            )
        else:
            self.find_sampling_optimum()
            self.chisquare_testing(search_data, threshold=threshold)
            log.info(
                "Optimum restored to {:.2f}, {:.2f}.".format(
                    self.samp_optimum[0], self.samp_optimum[1]
                )
            )

    def plot_param_L_dep(
        self,
        gene_filter_=None,
        plot_errorbars=False,
        figsize=None,
        c=2.576,
        axis_search_bounds=True,
        plot_fit=False,
        distinguish_rej=True,
    ):
        """Plot and fit the biological parameter dependence on length at the sampling parameter optimum.

        This dependence is expected to be weak; burst size may show a very slight negative trend.

        Parameters
        ----------
        gene_filter_: None or np.ndarray, optional
            If None, plot all genes.
            If a boolean or integer filter, plot the filtered gene subset.
        plot_errorbars: bool, optional
            whether to use inferred standard error of maximum likelihood estimates to plot error bars.
        figsize: None or tuple of floats, optional
            figure dimensions.
        c: float
            if plotting the errorbars, the number of standard deviations to display.
            this can be used for Gaussian approximations to the MLE confidence intervals.
        axis_search_bounds: bool, optional
            whether to place the x-limits of the plots at the parameter search bounds.
        plot_fit: bool, optional
            whether to plot a linear fit to the data points.
        distinguish_rej: bool, optional
             whether to distinguish the genes in the rejected_genes attribute by color.
        """
        try: 
            gene_log_lengths = self.gene_log_lengths
        except AttributeError:
            log.info('No gene lengths given, length-dependence cannot be plotted. Function exiting')
            return 
            
        num_params = self.sp.n_phys_pars
        figsize = figsize or (4 * num_params, 4)

        fig1, ax1 = plt.subplots(nrows=1, ncols=num_params, figsize=figsize)

        gene_filter = self.get_bool_filt(gene_filter_, discard_rejected=False)
        gene_filter_rej = np.zeros(self.n_genes, dtype=bool)

        if distinguish_rej:  # default
            filt_rej = self.get_bool_filt(gene_filter_, discard_rejected=True)
            gene_filter_rej = np.logical_and(
                gene_filter, np.logical_not(filt_rej)
            )  # subset for rejected genes
            gene_filter = np.logical_and(gene_filter, filt_rej)
            acc_point_aesth = (
                "accepted_gene_color",
                "accepted_gene_alpha",
                "accepted_gene_ms",
            )
            rej_point_aesth = (
                "rejected_gene_color",
                "rejected_gene_alpha",
                "rejected_gene_ms",
            )
        else:  # don't distinguish
            acc_point_aesth = (
                "generic_gene_color",
                "generic_gene_alpha",
                "generic_gene_ms",
            )
            log.info("Falling back on generic marker properties.")

        for i in range(num_params):
            if plot_errorbars:

                lfun = lambda x, a, b: a * x + b
                if plot_fit:
                    
                    popt, pcov = scipy.optimize.curve_fit(
                        lfun,
                        self.gene_log_lengths[gene_filter],
                        self.phys_optimum[gene_filter, i],
                        sigma=self.sigma[gene_filter, i],
                        absolute_sigma=True,
                    )
                    xl = np.array(
                        [min(self.gene_log_lengths), max(self.gene_log_lengths)]
                    )

                    min_param = (
                        popt[0] - np.sqrt(pcov[0, 0]) * c,
                        popt[1] - np.sqrt(pcov[1, 1]) * c,
                    )
                    max_param = (
                        popt[0] + np.sqrt(pcov[0, 0]) * c,
                        popt[1] + np.sqrt(pcov[1, 1]) * c,
                    )
                    ax1[i].fill_between(
                        xl,
                        lfun(xl, min_param[0], min_param[1]),
                        lfun(xl, max_param[0], max_param[1]),
                        facecolor=aesthetics["length_fit_face_color"],
                        alpha=aesthetics["length_fit_face_alpha"],
                    )
                    ax1[i].plot(
                        xl,
                        lfun(xl, popt[0], popt[1]),
                        c=aesthetics["length_fit_line_color"],
                        linewidth=aesthetics["length_fit_lw"],
                    )
                ax1[i].errorbar(
                    self.gene_log_lengths[gene_filter],
                    self.phys_optimum[gene_filter, i],
                    self.sigma[gene_filter, i] * c,
                    c=aesthetics["errorbar_gene_color"],
                    alpha=aesthetics["errorbar_gene_alpha"],
                    linestyle="None",
                    linewidth=aesthetics["errorbar_lw"],
                )

            ax1[i].scatter(
                self.gene_log_lengths[gene_filter],
                self.phys_optimum[gene_filter, i],
                c=aesthetics[acc_point_aesth[0]],
                alpha=aesthetics[acc_point_aesth[1]],
                s=aesthetics[acc_point_aesth[2]],
            )
            if np.any(gene_filter_rej):
                ax1[i].scatter(
                    self.gene_log_lengths[gene_filter_rej],
                    self.phys_optimum[gene_filter_rej, i],
                    c=aesthetics[rej_point_aesth[0]],
                    alpha=aesthetics[rej_point_aesth[1]],
                    s=aesthetics[rej_point_aesth[2]],
                )

            ax1[i].set_xlabel(r"$\log_{10}$ L")
            ax1[i].set_ylabel(self.model.get_log_name_str()[i])
            if axis_search_bounds:
                ax1[i].set_ylim([self.sp.phys_lb[i], self.sp.phys_ub[i]])
        
        fig1.tight_layout()
        fig_string = self.analysis_figure_string + "/length_dependence.png"
        plt.savefig(fig_string, dpi=450)
        log.info("Figure stored to {}.".format(fig_string))

    def plot_gene_distributions(
        self,
        search_data,
        sz=(5, 5),
        figsize=(10, 10),
        marg="joint",
        logscale=None,
        title=True,
        genes_to_plot=None,
        savefig=True,
    ):
        """Plot the gene count distributions and their fits at the sampling parameter optimum.

        Parameters
        ----------
        search_data: monod.extract_data.SearchData
            SearchData object with the fit data.
        szfig: tuple of ints, optional
            dimensions of the figure subplot grid.
        figsize: tuple of floats, optional
            figure dimensions.
        marg: str, optional
            if 'unspliced', 'spliced', 'protein' etc., plot marginal for this modality.
            if 'joint': plot the bivariate distribution for spliced or unspliced
            if tuple of modality strings, plot the bivariate distribution for these modalities.
        logscale: None or bool, optional
            whether to plot probabilities or log-probabilities.
            by default, True for 'joint', False for marginals.
        title: bool, optional
            whether to report the gene name in each subplot title.
        genes_to_plot: bool or int np.ndarray or None, optional
            if array, which genes to plot.
            if None, plot by internal order.
        savefig: bool, optional
            whether to save the figure to disk.
        """

        if logscale is None:
            if marg == "joint" or type(marg)==tuple:
                logscale = True
            else:
                logscale = False

        (nrows, ncols) = sz
        fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        nax = np.prod(sz)
        if genes_to_plot is None:
            genes_to_plot = np.arange(nax)

        genes_to_plot = np.asarray(genes_to_plot)
        if genes_to_plot.dtype == bool:
            gtp_temp = np.arange(self.n_genes)
            genes_to_plot = gtp_temp[genes_to_plot]

        number_of_genes_to_plot = len(genes_to_plot)
        if number_of_genes_to_plot > self.n_genes:
            number_of_genes_to_plot = self.n_genes
            genes_to_plot = genes_to_plot[: self.n_genes]
        if number_of_genes_to_plot > nax:  # This should no longer break...
            number_of_genes_to_plot = nax
            genes_to_plot = genes_to_plot[:nax]

        j_ = 0

        for i_ in genes_to_plot:
            lm = np.copy(search_data.M[:, i_])
            
            # TODO: generalize by adding attribute names as attribute of e.g. CMEModel
            # attributes = ['unspliced', 'spliced', 'protein']
            self.modalities = self.model.model_modalities
            num_modalities = len(self.modalities)
            for i in range(num_modalities):
                if marg == self.modalities[i]:
                    lm[:i] = 1
                    lm[i+1:]=1
                    
            axloc = np.unravel_index(j_, sz) if (sz[0] > 1 and sz[1] > 1) else j_

            samp = self.regressor_optimum[i_]
            Pa = np.squeeze(self.model.eval_model_pss(self.phys_optimum[i_], lm, samp))

            if marg == "joint":
                if logscale:
                    Pa[Pa < 1e-10] = 1e-10
                    Pa = np.log10(Pa)

                if num_modalities==2:
                    ax1[axloc].imshow(Pa.T, aspect="auto", cmap="summer")
                # Set so that, for protein model, the joint distribution of protein and spliced is shown.
                elif num_modalities==3:
                    ax1[axloc].imshow(Pa.sum(axis=0).T, aspect="auto", cmap="summer")
                else:
                    log.error('Joint distribution plot only implemented for 2 or 3 modalities')
                    
                ax1[axloc].invert_yaxis()

                jitter_magn = 0.1
                jitter = np.random.randn(2, self.n_cells) * jitter_magn
                if num_modalities==2:
                    ax1[axloc].scatter(
                        *search_data.layers[:2, :, i_] + jitter, c="k", s=1, alpha=0.1)
                # for protein model, scatter protein and spliced counts.
                elif num_modalities==3:
                    ax1[axloc].scatter(
                        *search_data.layers[1:, :, i_] + jitter, c="k", s=1, alpha=0.1
                )
                if num_modalities==2:
                    ax1[axloc].set_xlim([-0.5, search_data.M[0, i_] - 1.5])
                    ax1[axloc].set_ylim([-0.5, search_data.M[1, i_] - 1.5])
                elif num_modalities==3:
                    ax1[axloc].set_xlim([-0.5, search_data.M[1, i_] - 1.5])
                    ax1[axloc].set_ylim([-0.5, search_data.M[2, i_] - 1.5])
            else:
                plot_hist_and_fit(ax1[axloc], search_data, i_, Pa, marg)
                if logscale:
                    ax1[axloc].set_yscale("log")
            if title:
                titlestr = self.gene_names[i_]
                if hasattr(self, "rejected_genes") and self.rejected_genes[i_]:
                    titlestr += " (rej.)"
                ax1[axloc].set_title(titlestr, fontdict={"fontsize": 9})
            ax1[axloc].set_xticks([])
            ax1[axloc].set_yticks([])
            j_ += 1
        fig1.tight_layout(pad=0.02)

        if savefig:
            fig_string = (
                self.analysis_figure_string + "/gene_distributions_{}.png".format(marg)
            )
            plt.savefig(fig_string, dpi=450)
            log.info("Figure stored to {}.".format(fig_string))

    # the next two functions are useful for model selection, but are not currently in use.
    def get_logL(self, search_data, n_cells, EPS=1e-20, offs=0):
        """
        This method calculates the log-likelihood for all genes at the sampling parameter optimum.

        Parameters
        ----------
        search_data: a SearchData instance.
        EPS: probability rounding parameter -- anything below this is rounded to EPS.

        Output:
        logL: a vector of size n_genes containing model log-likelihoods.
        """
        hist_type = get_hist_type(search_data)
        logL = np.zeros(self.n_genes)

        for gene_index in range(self.n_genes):
            logL[gene_index] = self.model.eval_model_logL(
                p=self.phys_optimum[gene_index],
                limits=search_data.M[:, gene_index] + offs,
                samp=self.regressor_optimum[gene_index],
                data=search_data.hist[gene_index],
                n_cells=n_cells,
                hist_type=hist_type,
                EPS=EPS,
            )
            # Pss = self.model.eval_model_pss(self.phys_optimum[gene_index],lm,samp)
            # if np.any(Pss<EPS):
            #     Pss[Pss<EPS] = EPS
            # expected_log_lik = np.log(Pss)
            # eval_model_kld(self, p, limits, samp, data, hist_type="unique", EPS=EPS)
            # logL[gene_index] = expected_log_lik[search_data.U[gene_index].astype(int),search_data.S[gene_index].astype(int)].sum()
        return logL

    def get_noise_decomp(self):
        """
        #obsolete.
        This method reports the fractions of normalized variance attributable to intrinsic,
        extrinsic, and technical noise under the instantiated model, using the eval_model_noise
        method of the CMEModel class.

        Output:
        f: array with size n_genes x 3 x 2.
            dim 0: gene
            dim 1: variance fraction (intrinsic, extrinsic, technical)
            dim 2: species (unspliced, spliced)
        The null technical noise model has dim 0 of size 2, as it has no technical noise component.
        """
        f = []
        for gene_index in range(self.n_genes):
            f.append(
                [
                    self.model.eval_model_noise(
                        self.phys_optimum[gene_index],
                        samp=self.regressor_optimum[gene_index],
                    )
                ]
            )
        return np.asarray(f).squeeze()

def get_AIC_mek_means(search_result_list, search_data, AIC_EPS=1e-20, AIC_offs=0):

    LL = np.zeros(search_result_list[0].n_genes)
    
    for sr in search_result_list:

        n_cells = sr.n_cells
        LL += sr.get_logL(search_data, EPS=AIC_EPS, offs=AIC_offs)

    # Calculate the number of parameters
    n_params = sr.sp.n_phys_pars
    
    # Calculate AIC for each gene
    AIC = 2 * n_params - 2 * LL

    return AIC


def parallelize(
    function,
    iterable,
    num_cores,
    num_entries,
    completion_message="Parallelization complete.",
    termination_message="Parallelization manually terminated.",
    error_message="Parallelization terminated due to error.",
    use_tqdm=True,
):
    """Helper function to safely parallelize computations.

    Inputs a single-parameter function and an iterable, requests a number of cores, \
    and gracefully shuts down if needed.

    Parameters
    ----------
    function: function
        a one-parameter function that can be mapped using entries of the iterable.
    iterable: iterable
        an iterable to be passed into the function.        
    num_cores: int
        number of cores to use for parallelization.
    num_entries: int
        length of the iterable, used for tqdm.
    completion_message: str, optional
        string to print to log upon completion.
    termination_message: str, optional
        string to print to log if the process is manually terminated.
    error_message: str, optional
        string to print to log if the process fails due to a ValueError.
    use_tqdm: bool, optional
        whether to visualize progress using tqdm.

    Returns
    -------
    x: iterable
        result of applying function to iterable.

    """
    try:
        pool = multiprocessing.Pool(processes=num_cores)
        if use_tqdm:
            x = list(tqdm(pool.imap(function, iterable), total=num_entries))  # hacky
        else:
            x = pool.map(function, iterable)
        pool.close()
        pool.join()
        log.info(completion_message)
    except KeyboardInterrupt:
        log.warning(termination_message)
        pool.terminate()
        pool.join()
    except ValueError:
        log.warning(error_message)
        pool.terminate()
        pool.join()
    return x


def plot_hist_and_fit(
    ax1,
    monod_adata,
    i_,
    Pa,
    marg="nascent",
    facecolor=aesthetics["hist_face_color"],
    fitcolor=aesthetics["hist_fit_color"],
    facealpha=aesthetics["hist_face_alpha"],
    linestyle=aesthetics["linestyle"],
):
    """Plots marginal gene count distributions and their fits at the sampling parameter optimum.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axes to plot into.
    monod_adata: anndata
        anndata object with the fit data.
    i_: int
        gene index to plot.
    Pa: float np.ndarray
        univariate probability mass function, typically computed through CMEModel.
    marg: str, optional
        which marginal to plot, typically 'nascent' or 'mature'.
    facecolor: str or tuple, optional
        histogram face color in a matplotlib-compatible format.
    fitcolor: str or tuple, optional
        model fit line color in a matplotlib-compatible format.
    facealpha: float, optional
        histogram face alpha.
    linestyle: str, optional
        model fit line style in a matplotlib-compatible format.
    """

    # Identify layer index of the layer we want to gain the marginal of.
    layer_names = [i for i in monod_adata.layers.keys()]
    for i in range(len(layer_names)):
        if marg==layer_names[i]:
            lind = i
    # Otherwise use first modality.
    else:
        lind = 0
    marg_name = layer_names[lind]
        
    ax1.hist(
        monod_adata.layers[marg_name][:, i_],
        bins=np.arange(monod_adata.uns['M'][lind, i_]) - 0.5,
        density=True,
        color=facecolor,
        alpha=facealpha,
    )
    ax1.plot(np.arange(sd.M[lind, i_]), Pa, color=fitcolor, linestyle=linestyle)
    ax1.set_xlim([-0.5, sd.layers[lind, i_].max() + 2.5])


# Differential expression analysis between mek-means clusters.
def make_fcs(sr,sd,clus1=0,clus2=1,gf_rej=False,thrpars=2,thrmean=1,outlier_de=False,nuc=False,correct_off=False):
    '''
    Utilize different metrics to find fold-changes (FCs) between cluster parameters

    sr: list of SearchResults objects from meK-Means runs
    sd: SearchData object that corresponds to full, input data
    clus1: cluster 1 (to compare FCS of cluster 1/cluster 2 )
    clus2: cluster 2 (to compare FCS of cluster 1/cluster 2 )
    gf_rej: whether to use boolean list of rejected genes from both clusters
    thrpars: FC threshold value (to call DE-theta genes)
    thrmean: Mean S expression threshold value, for genes to consider
    outlier_de: Use iterative outlier calling procedure to assign DE-theta genes (see Monod https://github.com/pachterlab/monod_examples/blob/main/Monod_demo.ipynb)
    nuc: is this nuclear RNA data
    correct_off: correct offset between cluster parameters (through ODR)
    '''

    all_filt_fcs = pd.DataFrame()
    fcs,types,which_pair,highFC,spliceFC,g_names,out_de = ([] for i in range(7))

    ind1 = [i for i in range(len(sr)) if clus1 == sr[i].assigns][0]
    ind2 = [i for i in range(len(sr)) if clus2 == sr[i].assigns][0]

    sr1 = sr[ind1]
    sr2 = sr[ind2]
    if correct_off:
        param_names = sr1.model.get_log_name_str()
        offsets = []
        par_vals = np.copy(sr2.param_estimates)
        for k in range(3):
            m1 = sr1.param_estimates[0,:,k]
            m2 = sr2.param_estimates[0,:,k]
            offset = diffexp_fpi(m1,m2,param_names[k],viz=False)[1]
            par_vals[0,:,k] -= offset

        fc_par = (sr1.param_estimates-par_vals)/np.log10(2)
    else:
        fc_par = (sr1.param_estimates-sr2.param_estimates)/np.log10(2)  #Get FCs between cluster params

    # print('fc_par.shape: ',fc_par.shape)
    if nuc:
        fc_s_par = np.log2(sd.layers[0][:,sr1.filt].mean(1)/sd.layers[0][:,sr2.filt].mean(1))
    else:
        fc_s_par = np.log2(sd.layers[1][:,sr1.filt].mean(1)/sd.layers[1][:,sr2.filt].mean(1)) #Get spliced FCs

    # print('fc_s_par.shape: ',fc_s_par.shape)

    if outlier_de:
        dr_analysis = monod.analysis.diffexp_pars(sr1,sr2,viz=True,modeltype='id',use_sigma=True)
        par_bool_de = dr_analysis[1].T

    parnames = ('b','beta','gamma')


  #-----is parameter FC significant -----
    if gf_rej is False:
        gf_rej = [True]*sd.n_genes
    else:
        gf_rej = (~sr1.rejected_genes) & (~sr2.rejected_genes)

    for n in range(len(parnames)):
        #Boolean for if large param FC and not rejected gene (with minimum expression)
        if nuc:
            gf_highnoise = (np.abs(fc_par[0,:,n])>thrpars)  \
                & ((sd.layers[0][:,sr1.filt].mean(1)>thrmean) | (sd.layers[0][:,sr2.filt].mean(1)>thrmean)) \
                & gf_rej
        else:
            gf_highnoise = (np.abs(fc_par[0,:,n])>thrpars)  \
                & ((sd.layers[1][:,sr1.filt].mean(1)>thrmean) | (sd.layers[1][:,sr2.filt].mean(1)>thrmean)) \
                & gf_rej

        #Boolean for FC (above) but no FC detected at S-level
        gf_highnoise_meanS = gf_highnoise & (np.abs(fc_s_par)<1) & gf_rej

        #Boolean for FC (above)
        gf_onlyhigh = gf_highnoise & gf_rej

        #For dataframe
        fcs += list(fc_par[0,gf_rej,n])
        g_names += list(sr1.gene_names[gf_rej])
        which_pair += [[sr1.assigns,sr2.assigns]]*np.sum(gf_rej)
        highFC += list(gf_onlyhigh[gf_rej])
        spliceFC += list(gf_highnoise_meanS[gf_rej])
        types += [parnames[n]]*np.sum(gf_rej)
        if outlier_de:
            out_de += list(par_bool_de[gf_rej,n])

    if outlier_de:
        all_filt_fcs['deTheta_outlier'] = out_de

    all_filt_fcs['log2FC'] = fcs
    all_filt_fcs['gene'] = g_names
    all_filt_fcs['cluster_pair'] = which_pair
    all_filt_fcs['deTheta_FC'] = highFC
    all_filt_fcs['deTheta_noDeMuS'] = spliceFC
    all_filt_fcs['param'] = types

    return all_filt_fcs
