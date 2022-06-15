import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle

from .preprocess import * #identify_annotated_genes, get_transcriptome, import_raw, filter_by_gene, make_dir




########################
## Main code
########################

def extract_data(dataset_filepath, transcriptome_filepath, dataset_name,
                    dataset_string, dir_string,\
                    viz=True,\
                    dataset_attr_names=['spliced','unspliced','gene_name','barcode'],\
                    padding = [10,10],
                    filter_cells_S = 0, filter_cells_U = 0):
    """
    This function takes in a single dataset and outputs a single SearchData file with data corresponding to selected genes.

    Input: 
    dataset_filepath: string giving location of file with count data. Loom, adata, mtx supported.
    transcriptome_filepath: string giving location of file with gene length annotations.
    dataset_name: dataset metadata; name assigned to dataset-specific folder.
    dataset_string: dataset-specific directory location.
    dir_string: batch directory location.
    viz: whether to visualize and save an expression vs. gene length figure.
    dataset_attr_names: spliced layer, unspliced layer, gene name, and cell barcode attribute names in adata or loom file.
    padding: the PMF computation procedure uses a grid on (max U + padding[0], max S + padding[1]) for each gene.
        this list determines the amount of padding to be applied to the grid.
    filter_cells_S and filter_cells_U: the code provides an option to remove the highest-expression cells as outliers.
        Such outliers are identified by total number of spliced or unspliced molecules. 
        These arguments determine how many cells to remove.

    Output:
    search_data: SearchData object.

    Creates:
    Copy of search_data on disk in the dataset subdirectory.
    """
    log.info('Beginning data extraction.')
    log.info('Dataset: '+dataset_name)

    transcriptome_dict = get_transcriptome(transcriptome_filepath)
    S,U,gene_names,n_cells = import_raw(dataset_filepath,*dataset_attr_names)

    #identify genes that are in the length annotations. discard all the rest.
    #for models without length-based sequencing, this may not be necessary 
    #though I do not see the purpose of analyzing genes that not in the
    #genome.
    #if this is ever necessary, just make a different reference list.
    annotation_filter = identify_annotated_genes(gene_names,transcriptome_dict)
    S,U,gene_names = filter_by_gene(annotation_filter,S,U,gene_names)

    #initialize the gene length array.
    len_arr = np.array([transcriptome_dict[k] for k in gene_names])

    gene_result_list_file = dir_string+'/genes.csv'
    try:
        with open(gene_result_list_file, newline='') as f:
            reader = csv.reader(f)
            analysis_gene_list = list(reader)[0]
        log.info('Gene list extracted from {}.'.format(gene_result_list_file))
    except:
        log.error('Gene list could not be extracted from {}.'.format(gene_result_list_file))
        #raise an error here in the next version.

    if viz:
        var_name = ('S','U')
        var_arr = (S.mean(1),U.mean(1))
        fig1, ax1 = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
        for i in range(2):
            ax1[i].scatter(np.log10(len_arr), np.log10(var_arr[i] + 0.001),s=3,
                        c='silver',alpha=0.15)
            ax1[i].set_xlabel('log10 gene length')
            ax1[i].set_ylabel('log10 (mean '+var_name[i]+' + 0.001)')

    gene_names = list(gene_names)
    gene_filter = [gene_names.index(gene) for gene in analysis_gene_list]
    gene_names = np.asarray(gene_names)
    S,U,gene_names,len_arr = filter_by_gene(gene_filter,S,U,gene_names,len_arr)
    if filter_cells_S>0:
        log.info('Throwing out the {:.0f} highest spliced expression cells.'.format(filter_cells_S))
        n_cells -= filter_cells_S
        filt = np.argsort(-S.sum(0))[filter_cells_S:]
        S = S[:,filt]
        U = U[:,filt]
    if filter_cells_U>0:
        log.info('Throwing out the {:.0f} highest unspliced expression cells.'.format(filter_cells_U))
        n_cells -= filter_cells_U
        filt = np.argsort(-U.sum(0))[filter_cells_U:]
        S = S[:,filt]
        U = U[:,filt]

    if viz:
        for i in range(2):
            var_arr = (S.mean(1),U.mean(1))
            ax1[i].scatter(np.log10(len_arr), np.log10(var_arr[i] + 0.001),s=5,
                        c='firebrick',alpha=0.9)
        dataset_diagnostics_dir_string = dataset_string + '/diagnostic_figures'
        make_dir(dataset_diagnostics_dir_string)
        plt.savefig(dataset_diagnostics_dir_string+'/{}.png'.format(dataset_name),dpi=450)

    n_genes = len(gene_names)
    M = np.asarray([np.amax(U[gene_index]) for gene_index in range(n_genes)],dtype=int)+padding[0]
    N = np.asarray([np.amax(S[gene_index]) for gene_index in range(n_genes)],dtype=int)+padding[1]

    gene_log_lengths = np.log10(len_arr)

    hist = []
    moments = [] 
    raw_U = []
    raw_S = []
    for gene_index in range(n_genes):
        H, xedges, yedges = np.histogram2d(U[gene_index],S[gene_index], 
                                          bins=[np.arange(M[gene_index]+1)-0.5,
                                          np.arange(N[gene_index]+1)-0.5],
                                          density=True)
        hist.append(H)

        moments.append({'S_mean':S[gene_index].mean(), \
                        'U_mean':U[gene_index].mean(), \
                        'S_var':S[gene_index].var(), \
                        'U_var':U[gene_index].var()})
    
    attr_names = ('M','N','hist','moments','gene_log_lengths','n_genes','gene_names','n_cells','S','U')
    search_data = SearchData(attr_names,\
                             M,N,hist,moments,gene_log_lengths,n_genes,gene_names,n_cells,S,U)
    search_data_string = dataset_string+'/raw.sd'
    store_search_data(search_data,search_data_string)
    return search_data

########################
## Helper functions
########################
def store_search_data(search_data,search_data_string):
    """
    This function attempts to store a search data object to disk.

    Inputs:
    search_data: SearchData object.
    search_data_string: disk location.
    """
    try:
        with open(search_data_string,'wb') as sdfs:
            pickle.dump(search_data, sdfs)
        log.info('Search data stored to {}.'.format(search_data_string))
    except:
        log.error('Search data could not be stored to {}.'.format(search_data_string))

########################
## Helper classes
########################
class SearchData:
    """
    This class contains all of the dataset-specific information necessary for inference.

    Attributes:
    U: raw unspliced count data.
    S: raw spliced count data.
    n_cells: number of cells.
    gene_names: np string array of gene names.
    n_genes: number of genes.
    gene_log_lengths: log lengths of each gene.
    moments: dict with entries {S_mean, U_mean, S_var, U_var}, storing count data moments.
    hist: histogram produced from raw data on M x N grid.
    M: grid size for unspliced dimension.
    N: grid size for spliced dimension.
    """
    def __init__(self,attr_names,*input_data):
        for j in range(len(input_data)):
            setattr(self,attr_names[j],input_data[j])

    def knee_plot(self,ax1=None,thr=None,viz=False):
        """
        This method plots the knee plot for the spliced mRNA counts and filters for cells that have
        more UMIs than the threshold.

        Input:
        ax1: matplotlib axes to plot into.
        thr: UMI threshold.
        viz: whether to visualize.

        Output:
        cf: cell filter for cells to be retained.
        """
        umi_sum = self.S.sum(0)
        umi_rank = np.argsort(umi_sum)
        if ax1 is None and viz:
            fig1,ax1 = plt.subplots(1,1,figsize=(7,5))
        usf = np.flip(umi_sum[umi_rank])
        if viz:
            ax1.plot(np.arange(self.n_cells),usf,'k')
            ax1.set_xlabel('Cell rank')
            ax1.set_ylabel('UMI count+1')
            ax1.set_yscale('log')
        if thr is not None:
            cf = umi_sum>thr
            rank_ = np.argmin(np.abs(usf-thr))
            if viz:
                ax1.plot([0,self.n_cells+1],thr*np.ones(2),'r--')
                ax1.plot(rank_*np.ones(2),[umi_sum.min(),umi_sum.max()],'r--')
            return cf

    def get_noise_decomp(self,sizefactor = 'pf',lognormalize=True,pcount=0,knee_thr=None):
        """
        This method performs normalization and variance stabilization on the raw data, and
        reports the fractions of normalized variance retained and removed as a result of the process.

        Input:
        sizefactor: what size factor to use. 
            'pf': Proportional fitting; set the size of each cell to the mean size.
            a number: use this number (e.g., 1e4 for cp10k).
            None: do not do size/depth normalization.
        lognormalize: whether to do log(1+x).
        pcount: pseudocount added to ensure division by zero does not occur.
        knee_thr: knee plot UMI threshold used to filter out low-expression cells.

        Output: 
        f: array with size n_genes x 2 x 2. 
            dim 0: gene
            dim 1: variance fraction (retained, discarded)
            dim 2: species (unspliced, spliced)
        The unspliced and spliced species are analyzed independently.
        """
        f = np.zeros((self.n_genes,2,2)) #genes -- bio vs tech -- species

        S = np.copy(self.S)
        U = np.copy(self.U)

        if knee_thr is not None:
            cf = self.knee_plot(thr=knee_thr)
            S = S[:,cf]
            U = U[:,cf]
        
        CV2_1 = S.var(1)/S.mean(1)**2
        CV2_2 = U.var(1)/U.mean(1)**2

        S = normalize_count_matrix(S,sizefactor,lognormalize,pcount)
        U = normalize_count_matrix(U,sizefactor,lognormalize,pcount)

        CV2_1_ = S.var(1)/S.mean(1)**2
        CV2_2_ = U.var(1)/U.mean(1)**2    

        #compute fraction of CV2 eliminated for unspliced and spliced
        f[:,0,0] = CV2_2_/CV2_2
        f[:,1,0] = 1-f[:,0,0]
        
        f[:,0,1] = CV2_1_/CV2_1
        f[:,1,1] = 1-f[:,0,1]
        return f

def normalize_count_matrix(X,sizefactor = 'pf',lognormalize=True,pcount=0,logbase='e'):
    if sizefactor is not None:
        if sizefactor == 'pf':
            sizefactor = X.sum(0).mean()
        X = X/(X.sum(0)[None,:]+pcount)*sizefactor
    if lognormalize:
        if logbase=='e':
            X = np.log(X+1)
        elif logbase ==2:
            X = np.log2(X+1)
    return X