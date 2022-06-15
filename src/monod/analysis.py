# from preprocess import *
from .cme_toolbox import *
from .inference import *
from .extract_data import *
import pickle
from scipy import stats
from scipy import odr


########################
## Helper functions
########################

def load_search_results(full_result_string):
    """
    This function attempts to load in search results.

    Input: 
    full_result_string: location of the SearchResults object.
    
    Output: 
    search_results: a SearchResults object.
    """
    try:
        with open(full_result_string,'rb') as srfs:
            search_results = pickle.load(srfs)
        log.info('Grid scan results loaded from {}.'.format(full_result_string))
        return search_results
    except:
        log.error('Grid scan results could not be loaded from {}.'.format(full_result_string))

def load_search_data(search_data_string):
    """
    This function attempts to load in search data.

    Input: 
    full_result_string: location of the SearchData object.
    
    Output: 
    sd: a SearchData object.
    """
    try:
        with open(search_data_string,'rb') as sdfs:
            sd = pickle.load(sdfs)
        log.info('Search data loaded from {}.'.format(search_data_string))
        return sd
    except:
        log.error('Search data could not be loaded from {}.'.format(search_data_string))


def make_batch_analysis_dir(sr_arr,dir_string):
    """
    This function creates a directory for batch analysis.

    Input: 
    sr_arr: list of multiple SearchResults objects.
    dir_string: batch directory.
    
    Output: 
    sd: a SearchData object.
    """
    batch_analysis_string = dir_string + '/analysis_figures'
    make_dir(batch_analysis_string)
    for sr in sr_arr:
        sr.batch_analysis_string = batch_analysis_string

def plot_params_for_pair(sr1,sr2,gene_filter_ = None,\
                     plot_errorbars=False,\
                     figsize=None,c=2.576,\
                     axis_search_bounds = True,
                     distinguish_rej = True,
                     plot_identity = True,
                     meta = '12',
                     xlabel = 'dataset 1',
                     ylabel = 'dataset 2'):
    """
    This function plots the inferred physical parameters at the sampling parameter optimum for a matched pair of datasets.

    Input:
    sr1: SearchResult instance 1.
    sr2: SearchResult instance 2.
    gene_filter_: 
        If None, plot all genes. 
        If a boolean or integer filter, plot only a subset of genes indicated by the filter.
    plot_errorbars: whether to use inferred standard error of MLEs to plot error bars.
    figsize: figure dimensions.
    c: error bar scaling factor. c=2.576 corresponds to a rough 99% CI.
    axis_search_bounds: whether to place the x-limits of the plots at the parameter search bounds.
    distinguish_rej: whether to distinguish genes in the rejected_genes attribute.
    meta: figure name metadata.
    xlabel: name of dataset 1.
    ylabel: name of dataset 2.
    """
    num_params = sr1.sp.n_phys_pars
    if figsize is None:
        figsize = (4*num_params,4)
    fig1,ax1=plt.subplots(nrows=1,ncols=num_params,figsize=figsize)


    gene_filter = sr1.get_bool_filt(gene_filter_,discard_rejected=False)
    # gene_filter2 = sr2.get_bool_filt(gene_filter_,discard_rejected=False)
    gene_filter_rej = np.zeros(sr1.n_genes,dtype=bool)
    # gene_filter_rej2 = np.zeros(sr2.n_genes,dtype=bool)

    if distinguish_rej: #default
        filt_rej1 = sr1.get_bool_filt(gene_filter_,discard_rejected=True)
        filt_rej2 = sr2.get_bool_filt(gene_filter_,discard_rejected=True)
        filt_rej = np.logical_and(filt_rej1, filt_rej2)

        gene_filter_rej = np.logical_and(gene_filter,np.logical_not(filt_rej))
        gene_filter = np.logical_and(gene_filter,filt_rej)
        acc_point_aesth = ('accepted_gene_color','accepted_gene_alpha','accepted_gene_ms')
        rej_point_aesth = ('rejected_gene_color','rejected_gene_alpha','rejected_gene_ms')
    else: #don't distinguish
        acc_point_aesth = ('generic_gene_color','generic_gene_alpha','generic_gene_ms')
        log.info('Falling back on generic marker properties.') 

    for i in range(3):
        if plot_errorbars:
            ax1[i].errorbar(sr1.phys_optimum[gene_filter,i],\
                            sr2.phys_optimum[gene_filter,i],\
                            sr1.sigma[gene_filter,i]*c,\
                            sr2.sigma[gene_filter,i]*c,\
                            c=aesthetics['errorbar_gene_color'],
                            alpha=aesthetics['errorbar_gene_alpha'],\
                            linestyle='None',
                            linewidth = aesthetics['errorbar_lw'])
        ax1[i].scatter(sr1.phys_optimum[gene_filter,i],
                       sr2.phys_optimum[gene_filter,i],\
                       c=aesthetics[acc_point_aesth[0]],\
                       alpha=aesthetics[acc_point_aesth[1]],\
                       s=aesthetics[acc_point_aesth[2]])
        if np.any(gene_filter_rej):
            ax1[i].scatter(sr1.phys_optimum[gene_filter_rej,i],
                           sr2.phys_optimum[gene_filter_rej,i],\
               c=aesthetics[rej_point_aesth[0]],\
               alpha=aesthetics[rej_point_aesth[1]],\
               s=aesthetics[rej_point_aesth[2]])

        ax1[i].set_xlabel(xlabel)
        ax1[i].set_ylabel(ylabel)
        ax1[i].set_title(sr1.model.get_log_name_str()[i])
        if axis_search_bounds:
            ax1[i].set_xlim([sr1.sp.phys_lb[i],sr1.sp.phys_ub[i]])
            ax1[i].set_ylim([sr1.sp.phys_lb[i],sr1.sp.phys_ub[i]])
        if plot_identity:
            xl = ax1[i].get_xlim()
            ax1[i].plot(xl,xl,'r--',linewidth=2)
    fig1.tight_layout()

    
    fig_string = sr1.batch_analysis_string+'/pair_parameter_comparison_{}.png'.format(meta)
    plt.savefig(fig_string,dpi=450)
    log.info('Figure stored to {}.'.format(fig_string))

def find_most_concordant_samp(sr1,sr2):
    """
    This function attempts to find a search parameter optimum by comparing two matched (control) datasets
    and finding the point at which their parameter values are most concordant, according to l2 distance over
    all genes and parameters.
    This typically works poorly.

    Input:
    sr1: SearchResult instance 1.
    sr2: SearchResult instance 2.

    Output:
    sampling parameter optimum value.
    """
    discordance = ((sr1.param_estimates - sr2.param_estimates)**2).sum(2).sum(1)
    srt =  np.argsort(discordance)
    samp_concordant_ind = srt[0]
    sr1.set_sampling_optimum(samp_concordant_ind)
    sr2.set_sampling_optimum(samp_concordant_ind)

    log.info('Optimum set to at {:.2f}, {:.2f}.'.format(sr1.samp_optimum[0],sr1.samp_optimum[1]))
    return sr1.samp_optimum

def get_AIC_weights(sr_arr,sd):
    """
    This method computes the Akaike Information Criterion weights according to the optimal
    physical and sampling parameters obtained for a single dataset (sd) under several models (results
    stored in (sr_arr).

    Input:
    sr_arr: list of multiple SearchResults objects.
    sd: SearchData instance.

    Output:
    w: AIC weights corresponding to each model, a n_models x n_genes array.
    """

    n_models = len(sr_arr)
    AIC = []
    for j in range(n_models):
        AIC += [2*sr_arr[j].sp.n_phys_pars-2*sr_arr[j].get_logL(sd)] 
    AIC = np.asarray(AIC)
    min_AIC = AIC.min(0)
    normalization = np.exp(-(AIC - min_AIC)/2).sum(0)
    w = np.exp(-(AIC - min_AIC)/2) / normalization
    return w

def plot_AIC_weights(sr_arr,sd,models,ax1=None,meta=None,figsize=None,                      
                      facecolor=aesthetics['hist_face_color'],\
                      facealpha=aesthetics['hist_face_alpha'],nbin=20,savefig=True):
    """
    This function calls get_AIC_weights and plots the resulting Akaike Information Criterion weights.

    Input:
    sr_arr: list of multiple SearchResults objects.
    sd: SearchData instance.
    models: model names.
    ax1: matplotlib axes to plot into.
    meta: figure metadata.
    figsize: figure dimensions.
    facecolor: histogram face color.
    facealpha: histogram face alpha.
    nbin: number of histogram bins.

    Output:
    w: AIC weights corresponding to each model, a n_models x n_genes array.
    """
    w=get_AIC_weights(sr_arr,sd)

    if meta is None:
        meta = ''
    else:
        meta = '_'+meta    

    n_models = w.shape[0]
    if figsize is None:
        figsize = (4*n_models,4)

    if ax1 is None:
        fig1,ax1=plt.subplots(nrows=1,ncols=n_models,figsize=figsize)
    else:
        fig1 = plt.gcf()

    for i in range(n_models):
        ax1[i].hist(w[i],bins=nbin,\
                        density=False,\
                        color=facecolor,alpha=facealpha)
        ax1[i].set_xlabel('AIC weight at MLE')
        ax1[i].set_ylabel('# genes')
        ax1[i].set_title(models[i])

    fig1.tight_layout()
    if savefig:
        fig_string = (sr_arr[0].batch_analysis_string)+('/AIC_comparison{}.png'.format(meta))

        plt.savefig(fig_string,dpi=450)
        log.info('Figure stored to {}.'.format(fig_string))
    return w

def compare_AIC_weights(w,dataset_names,batch_analysis_string,model_ind=0,figsize=(12,12),kde_bw=0.05):
    """
    This function compares the consistency of AIC weights for a single model across several datasets.
    For a given gene and model j, the function takes the weight w_j and compares its absolute difference
    between two datasets. Then, it aggregates the information over all genes and plots the
    kernel density estimates for a pair of datasets.
    If the KDE is near zero, inference on the same dataset tends to choose the same model.

    Input:
    w: AIC weights corresponding to each model, a n_datasets x n_models x n_genes array.
    dataset_names: dataset name metadata.
    batch_analysis_string: figure directory location.
    model_ind: which model to plot weights for.
    figsize: figure dimensions.
    kde_bw: kernel density estimate bandwidth.
    """
    fs = 12
    n_datasets = len(dataset_names)
    fig1,ax1=plt.subplots(nrows=n_datasets,ncols=n_datasets,figsize=figsize)
    for i in range(n_datasets):
        for k in range(n_datasets):
            if i>k:
                xx = np.linspace(-0.2, 1.2, 2000)
                
                kde = stats.gaussian_kde(np.abs(w[i,model_ind,:]-w[k,model_ind,:]),bw_method=kde_bw)
                ax1[i,k].plot(xx, kde(xx),'k')
            if i==k:
                ax1[i,k].hist(w[i,model_ind,:],30,facecolor=aesthetics['hist_face_color'])
            if i<k:
                fig1.delaxes(ax1[i,k])
            ax1[i,k].set_yticks([])
            if k==0:
                ax1[i,k].set_ylabel(dataset_names[i],fontsize=fs)
            if i==(n_datasets-1):
                ax1[i,k].set_xlabel(dataset_names[k],fontsize=fs)
    fig1.tight_layout()
    fig_string = batch_analysis_string+'/AIC_comparison_grid.png'

    plt.savefig(fig_string,dpi=450)
    log.info('Figure stored to {}.'.format(fig_string))

def diffexp_pars(sr1,sr2,modeltype='id',gene_filter_ = None,
                     discard_rejected = True,
                     use_sigma=True,
                     figsize=None,
                     meta = '12',viz=True,pval_thr=0.001,nit=10):
    """
    This function uses the optimal physical and sampling parameters obtained for a pair of datasets
    to attempt to identify sources of differential expression under a model of transcription,
    i.e., calling patterns of DE-theta, where theta is a biophysical parameter identifiable at steady state.
    Specifically, it uses a fixed-point iteration procedure to distinguish a set of genes with
    Gaussian aleatory variation from a set with systematic, high-magnitude deviations between the datasets.

    Input:
    sr1: SearchResult instance 1.
    sr2: SearchResult instance 2.
    modeltype: which statistical model of variation to fit.
        If 'id', the relationship between datasets' parameters is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' parameters is assumed to be ax + b = y; n_model_pars = 2.
    gene_filter_: 
        If None, use all genes. 
        If a boolean or integer filter, use only a subset of genes indicated by the filter.
    discard_rejected: whether to discard genes in the rejected_genes attribute.
    use_sigma: whether to use inferred standard error of MLEs to fit the aleatory variation model.
    figsize: figure dimensions.
    meta: figure name metadata.
    viz: whether to plot the histograms of residuals.
    pval_thr: p-value threshold to use for the Z-test.
    nit: number of FPI iterations to run.

    Output:
    gn: list of size n_phys_pars; each entry contains names of genes identified as DR.
    gf: ndarray of size n_phys_pars x n_genes; if true, the gene has been identified as DR.
    offs: ndarray of size n_phys_pars x n_model_pars, contains (b) or (b,a) for each physical parameter.
    resid: ndarray of size n_phys_pars x n_genes; contains residuals for each parameter under the final fit statistical model.
    """
    num_params = sr1.sp.n_phys_pars
    if viz:
        if figsize is None:
            figsize = (4*num_params,4)
        fig1,ax1=plt.subplots(nrows=1,ncols=num_params,figsize=figsize)


    gene_filter = sr1.get_bool_filt(gene_filter_,discard_rejected=False)

    if discard_rejected: #default
        filt_rej1 = sr1.get_bool_filt(gene_filter_,discard_rejected=True)
        filt_rej2 = sr2.get_bool_filt(gene_filter_,discard_rejected=True)
        filt_rej = np.logical_and(filt_rej1, filt_rej2)
        gene_filter = np.logical_and(gene_filter,filt_rej)

    
    gn = []
    gf = []
    offs = []
    resid = []
    
    parnames = sr1.model.get_log_name_str()
    for i in range(num_params):
        
        if viz:
            ax = ax1[i]
        else:
            ax = None
        
        resid_arr = np.asarray([np.nan]*sr1.n_genes)
        
        # xl = [sr1.sp.phys_lb[i],sr1.sp.phys_ub[i]]
        if use_sigma:
            gf_,offs_,resid_,pval_ = diffexp_fpi(sr1.phys_optimum[gene_filter,i],sr2.phys_optimum[gene_filter,i],parnames[i],\
                             modeltype=modeltype,ax1=ax,s1=sr1.sigma[gene_filter,i],s2=sr2.sigma[gene_filter,i],nit=nit,viz=viz,pval_thr=pval_thr)
        else:
            gf_,offs_,resid_,pval_ = diffexp_fpi(sr1.phys_optimum[gene_filter,i],sr2.phys_optimum[gene_filter,i],parnames[i],\
                             modeltype=modeltype,ax1=ax,s1=None,s2=None,nit=nit,viz=viz,pval_thr=pvapval_thrl)
        resid_arr[gene_filter] = resid_

        filtind = np.arange(sr1.n_genes)
        filtind = filtind[gene_filter]
        filtind = filtind[gf_]
        gf__ = np.zeros(sr1.n_genes,dtype=bool)
        gf__[filtind] = True
        gn_ = sr1.gene_names[gf__]
        gn.append(gn_)
        gf.append(gf__)
        offs.append(offs_)
        resid.append(resid_arr)
        
        if i==0 and viz:
            ax.legend()
    if viz:
        fig1.tight_layout()    
        fig_string = sr1.batch_analysis_string+'/parameter_residuals_{}.png'.format(meta)

        plt.savefig(fig_string,dpi=450)
        log.info('Figure stored to {}.'.format(fig_string))

    return  gn, np.asarray(gf), np.asarray(offs), np.asarray(resid)


def linoffset(B, x, modeltype='id'):
    """
    This helper function defines the statistical model for identifying DR genes.

    Input:
    B: vector of statistical model parameters of size n_model_pars.
    x: vector of optimal physical parameters of size n_phys_pars.
    modeltype: which statistical model of variation to fit.
        If 'id', the relationship between datasets' parameters is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' parameters is assumed to be ax + b = y; n_model_pars = 2.
    """
    if modeltype=='id':
        return x + B[0]
    elif modeltype=='lin':
        return B[1]*x + B[0]

def diffexp_fpi(m1,m2,parname=None,modeltype='id',ax1=None,s1=None,s2=None,nit=10,pval_thr = 0.001,viz=True):
    """
    This function uses the optimal physical and sampling parameters obtained for a pair of datasets
    to attempt to identify differentially expressed genes under a model of transcription, for a single stest statistic.
    Specifically, it uses a fixed-point iteration (FPI) procedure to distinguish a set of genes with
    Gaussian aleatory variation from a set with systematic, high-magnitude deviations between the datasets.

    Input:
    m1: parameter estimates from SearchResult instance 1.
    m2: parameter estimates from SearchResult instance 2.
        at this point, the sizes of m1 and m2 might be smaller than n_genes, because some genes may have been filtered out.
    modeltype: which statistical model of variation to fit.
        If 'id', the relationship between datasets' parameters is assumed to be x + b = y; n_model_pars = 1.
        If 'lin', the relationship between datasets' parameters is assumed to be ax + b = y; n_model_pars = 2.
    ax1: matplotlib axes to plot into.
    s1: standard error corresponding to m1 estimates.
    s2: standard error corresponding to m2 estimates.
    nit: number of FPI iterations to run.
    pval_thr: p-value threshold to use for the Z-test.
    viz: whether to plot the histograms of residuals.

    Output:
    gf: ndarray of size n_genes_filtered; if true, the gene has been identified as DR.
    out.beta: ndarray of size n_model_pars, contains (b) or (b,a) for the current physical parameter.
    resid: ndarray of size n_genes_filtered; contains residuals for the current physical parameter under the final fit statistical model.
    pval: p-value under the z-test.
    """
    fs=12
    gf = np.ones(len(m1),dtype=bool)
    # x = np.linspace(xl[0], xl[1], 100)
    fitlaw = scipy.stats.norm
    
    if modeltype == 'id':
        beta0 = [0]
    elif modeltype == 'lin':
        beta0 = [0,1]
    
    for j in range(nit):
        if s1 is None and s2 is None:
            d = odr.Data(m1[gf],m2[gf])
        else:
            d = odr.Data(m1[gf],m2[gf],s1[gf],s2[gf])
            
        md = lambda B,x : linoffset(B,x,modeltype=modeltype)
        odrmodel = scipy.odr.Model(md)

        myodr = odr.ODR(d, odrmodel, beta0=beta0)
        out = myodr.run()
        
        if modeltype == 'id':
            resid = m2-m1-out.beta[0]
        elif modeltype == 'lin':
            resid = m2-m1*out.beta[1]-out.beta[0]
        

        
        fitparams = fitlaw.fit(resid[gf])
        x = np.linspace(resid.min(),resid.max(),100)
        p = fitlaw.pdf(x, *fitparams)
        if j==0 and viz:
            ax1.plot(x, p, linestyle=aesthetics['init_fit_linestyle'], \
                        linewidth=aesthetics['init_fit_linewidth'],\
                        color=aesthetics['hist_fit_color_2'],label='Initial fit')

        z = (resid-fitparams[0])/fitparams[1]
        pval = (scipy.stats.norm.sf(np.abs(z))*2)
        gf = np.logical_not(pval<pval_thr)
        if j==(nit-1) and viz:
            n,bins = np.histogram(resid,200,density=True)
            binc = np.diff(bins)/2 + bins[:-1]
            ax1.bar(binc,n,width=np.diff(bins),color = aesthetics['hist_face_color'],align='center',
                   label='Putative static genes')
            ax1.plot(x, p, linestyle=aesthetics['linestyle'], \
                        linewidth=2,\
                        color=aesthetics['hist_fit_color'],label='FPI fit')
            y_,_ = np.histogram(resid[~gf],bins=bins,density=True)
            y_ *=(~gf).mean()
            ax1.bar(binc,y_,width=np.diff(bins),color = aesthetics['diffreg_gene_color'],align='center',
                label='Putative DR genes')
    if viz:
        ax1.set_xlabel(parname+' residual',fontsize=fs)
        ax1.set_ylabel('Density',fontsize=fs)
    gf = np.logical_not(gf)
    return gf,out.beta,resid,pval

def compare_gene_distributions(sr_arr,sd_arr,sz = (5,5),figsize = (10,10),\
               marg='mature',logscale=None,title=True,\
               genes_to_plot = None):
    """
    This function is analogous to the SearchResults method plot_gene_distributions, but it
    plots the marginal histograms for multiple datasets at a time (2 supported at this time).
    The intended use case is to identify a set of DR genes of interest, pass them in as 
    genes_to_plot, and inspect the difference between the distributions.

    By default, dataset 1 is plotted in red and dataset 2 is plotted in blue.

    Input:
    sr_arr: list of multiple SearchResults objects.
    sd_arr: list of multiple SearchData objects.
    sz: subplot dimensions.
    figsize: figure dimensions.
    marg: whether to plot 'nascent' or 'mature' marginal.
    logscale: whether to apply a log-transformation to the PMF for ease of visualization.
    title: whether to name the gene in the title.
    genes_to_plot: which genes to plot, either a boolean or integer array.
    """
    if logscale is None:
        if marg=='joint':
            logscale = True
        else:
            logscale = False
    (nrows,ncols)=sz
    
    fig1,ax1=plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)

    nax = np.prod(sz)
    if genes_to_plot is None:
        genes_to_plot = np.arange(nax)

    genes_to_plot = np.asarray(genes_to_plot)
    if genes_to_plot.dtype == bool:
        gtp_temp = np.arange(sr_arr[0].n_genes)
        genes_to_plot = gtp_temp[genes_to_plot]

    number_of_genes_to_plot = len(genes_to_plot)
    if number_of_genes_to_plot > sr_arr[0].n_genes:
        number_of_genes_to_plot = sr_arr[0].n_genes
        genes_to_plot = genes_to_plot[:self.n_genes]
    if number_of_genes_to_plot>nax: 
        number_of_genes_to_plot = nax
        genes_to_plot = genes_to_plot[:nax]

    
    j_ = 0
    for i_ in genes_to_plot:
        axloc = np.unravel_index(j_,sz) if (sz[0]>1 and sz[1]>1) else j_
        xlim = 1
        for j in range(len(sr_arr)):
            lm = [sd_arr[j].M[i_],sd_arr[j].N[i_]]
            if marg == 'mature':
                lm[0]=1
            if marg == 'nascent':
                lm[1]=1
            xlim = np.max([max(lm),xlim])
            samp = None if (sr_arr[j].model.seq_model == 'None') else sr_arr[j].regressor_optimum[i_]
            Pa = np.squeeze(sr_arr[j].model.eval_model_pss(sr_arr[j].phys_optimum[i_],lm,samp))

            if marg=='joint':
                log.error('Potentially implement this later...')
                raise ValueError('Cannot compare two 2D histograms!')
            else:
                plot_hist_and_fit(ax1[axloc],sd_arr[j],i_,Pa,marg,
                                  facealpha=aesthetics['hist_face_alpha_'+str(j+1)],
                                  facecolor=aesthetics['hist_face_color_'+str(j+1)],
                                  fitcolor=aesthetics['hist_fit_color_'+str(j+1)])
        ax1[axloc].set_xlim([-0.5,xlim-10])
        if logscale:
            ax1[axloc].set_yscale('log')
        if title: 
            ax1[axloc].set_title(sr_arr[0].gene_names[i_],fontdict={'fontsize': 9})
        
        ax1[axloc].set_xticks([])
        ax1[axloc].set_yticks([])

        j_+=1
    fig1.tight_layout(pad=0.02)


def diffexp_mean(sd1,sd2,sizefactor = 'pf',lognormalize=True,pcount=0,
                    pval_thr=0.001,method='ttest',bonferroni=True,modeltype='lin',viz=True,knee_thr=None,
                    fc_thr = 1,ax1=None,viz_resid=False):
    """
    This function attempts to identify differentially expressed genes using a simple comparison of 
    the means of gene-specific count distributions, i.e., DE-mu using the normalized mean as a test statistic.

    Input:
    sd1: SearchData instance 1.
    sd2: SearchData instance 2.
    sizefactor: what size factor to use. 
        'pf': Proportional fitting; set the size of each cell to the mean size.
        a number: use this number (e.g., 1e4 for cp10k).
        None: do not do size/depth normalization.
    lognormalize: whether to use a log2-transformation for the t-test.
    pcount: pseudocount added to ensure division by zero does not occur.
    pval_thr: p-value threshold for proposing that a gene is DE.
    method: the DE identification method to use.
        If 'ttest', use scipy.stats.ttest_ind, Welch’s t-test.
        If 'logmeanfpi', use the fixed-point iteration procedure on the distribution of log-means.
        If 'meanlogfpi', use the FPI procedure on the distribution of means of log+1 counts.
    bonferroni: whether to use a multiple comparison correction for the ttest method.
    modeltype: statistical model for variation between means, as in diffexp_fpi.
    viz: whether to visualize the results for non-'ttest' methods.
    knee_thr: knee plot UMI threshold used to filter out low-expression cells.
    fc_thr: fold change threshold for t-test.
    """
    s1 = np.copy(sd1.S)
    s2 = np.copy(sd2.S)

    if (viz or viz_resid) and ax1 is None:
        fig1,ax1 = plt.subplots(1,1)
        
    if method=='ttest':

        if knee_thr is not None:
            cf1 = sd1.knee_plot(thr=knee_thr[0])
            cf2 = sd2.knee_plot(thr=knee_thr[1])
            s1 = s1[:,cf1]
            s2 = s2[:,cf2]

        gf = np.zeros(sd1.n_genes,dtype=bool)
        if bonferroni:
            pval_thr /= sd1.n_genes
        s1 = normalize_count_matrix(s1,sizefactor = sizefactor,lognormalize=lognormalize,pcount=0,logbase=2)
        s2 = normalize_count_matrix(s2,sizefactor = sizefactor,lognormalize=lognormalize,pcount=0,logbase=2)

        p = np.zeros(sd1.n_genes)
        for i in range(sd1.n_genes):
            _,p_ = scipy.stats.ttest_ind(s1[i],s2[i],equal_var=False)
            if p_<pval_thr:
                gf[i] = True
            p[i] = p_
        if lognormalize:
            fc = s2.mean(1) - s1.mean(1)
        else:
            fc = np.log2(s1+1).mean(1) - np.log2(s2+1).mean(1)
        gf = gf & (np.abs(fc)>fc_thr)

        if viz: 
            pv = -np.log10(p)
            ax1.scatter(fc[gf],pv[gf],5,'r')
            ax1.scatter(fc[~gf],pv[~gf],3,'darkgray')
            ax1.set_xlabel(r'Fold change ($\log_2$)')
            ax1.set_ylabel(r'$-\log_{10} p$')
    else:
        if method=='logmeanfpi':
            m1 = np.log2(s1.mean(1))
            m2 = np.log2(s2.mean(1))
        elif method == 'meanlogfpi':
            m1 = np.log2(s1+1).mean(1)
            m2 = np.log2(s2+1).mean(1)
        gf,offs_,resid_,p = diffexp_fpi(m1,m2,'Spliced mean',\
                         modeltype=modeltype,ax1=ax1,s1=None,s2=None,nit=30,viz=viz_resid,pval_thr=pval_thr)
        # fc = m2-m1
        if viz: 
            pv = -np.log10(p)
            ax1.scatter(fc[gf],pv[gf],5,'r')
            ax1.scatter(fc[~gf],pv[~gf],3,'darkgray')
            ax1.set_xlabel(r'Fold change ($\log_2$)')
            ax1.set_ylabel(r'$-\log_{10} p$')
    return gf,fc