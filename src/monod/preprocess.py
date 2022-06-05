
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

import time
import loompy as lp
import anndata as ad
import os
from datetime import datetime
import pytz
import collections
import csv
import warnings



code_ver_global='022'

########################
## Debug and error logging
########################
import logging, sys

logging.basicConfig(stream=sys.stdout)
log = logging.getLogger()
log.setLevel(logging.INFO)

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

########################
## Main code
########################

def construct_batch(dataset_filepaths, transcriptome_filepath, dataset_names, batch_id=1,\
                    n_genes=100, seed=6, viz=True,\
                    filt_param={'min_U_mean':0.01,'min_S_mean':0.01,'max_U_max':400,\
                                'max_S_max':400,'min_U_max':3,'min_S_max':3},\
                    attribute_names=['spliced','unspliced','gene_name','barcode'],\
                    meta='batch',\
                    datestring=datetime.now(pytz.timezone('US/Pacific')).strftime("%y%m%d"),\
                    creator='gg', code_ver=code_ver_global,\
                    batch_location='.'):
    """
    This function runs basic pre-processing on the batch, creates directories, and writes a 
    list of genes to analyze.

    Input: 
    dataset_filepaths: string, or list of strings, giving locations of files with count data. Loom, adata, mtx supported.
    transcriptome_filepath: string giving location of file with gene length annotations.
    dataset_names: dataset metadata; names assigned to dataset-specific folders.
    batch_id: experiment number.
    n_genes: number of genes to select.
    seed: gene selection seed.
    viz: whether to visualize and save an expression vs. gene length figure.
    filt_param: bounds on mean and maximum counts.
    attribute_names: spliced layer, unspliced layer, gene name, and cell barcode attribute names in adata and loom files.
    meta: any string metadata, e.g. number of genes.
    datestring: current date, in ISO format. Califonia time by default.
    creator: creator initials.
    code_ver: version of the code used to perform the experiment.
    batch_location: directory where the analysis should be located.

    Output:
    dir_string: newly created batch directory location.
    dataset_strings: dataset-specific directories.

    Creates:
    batch directory and dataset subdirectories.
    genes.csv and gene_list.csv lists in the batch directory.
    """
    log.info('Beginning data preprocessing and filtering.')
    dir_string = batch_location.rstrip('/') + '/' + ('_'.join((creator, datestring, code_ver, meta, str(batch_id))))
    make_dir(dir_string)
    dataset_strings = []

    if type(dataset_filepaths) is str:
        dataset_filepaths = [dataset_filepaths] #if we get in a single string, we are processing one file /
    n_datasets = len(dataset_filepaths)

    if type(attribute_names[0]) is str:
        attribute_names = [attribute_names]*n_datasets

    transcriptome_dict = get_transcriptome(transcriptome_filepath)

    for dataset_index in range(n_datasets):
        dataset_filepath = dataset_filepaths[dataset_index]
        log.info('Dataset: '+dataset_names[dataset_index])
        dataset_attr_names = attribute_names[dataset_index] #pull out the correct attribute names
        S,U,gene_names,n_cells = import_raw(dataset_filepath,*dataset_attr_names)
        log.info(str(n_cells)+ ' cells detected.')

        #identify genes that are in the length annotations. discard all the rest.
        #for models without length-based sequencing, this may not be necessary 
        #though I do not see the purpose of analyzing genes that not in the
        #genome.
        #if this is ever necessary, just make a different reference list.
        annotation_filter = identify_annotated_genes(gene_names,transcriptome_dict)
        S,U,gene_names = filter_by_gene(annotation_filter,S,U,gene_names)

        #initialize the gene length array.
        len_arr = np.array([transcriptome_dict[k] for k in gene_names])

        #compute summary statistics
        gene_exp_filter = threshold_by_expression(S,U,filt_param)
        if viz:
            var_name = ('S','U')
            var_arr = (S.mean(1),U.mean(1))

            fig1, ax1 = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
            for i in range(2):
                ax1[i].scatter(np.log10(len_arr)[~gene_exp_filter], np.log10(var_arr[i][~gene_exp_filter] + 0.001),s=3,
                            c='silver',alpha=0.15)
                ax1[i].scatter(np.log10(len_arr[gene_exp_filter]), np.log10(var_arr[i][gene_exp_filter] + 0.001),s=3,
                            c='indigo',alpha=0.3)
                ax1[i].set_xlabel('log10 gene length')
                ax1[i].set_ylabel('log10 (mean '+var_name[i]+' + 0.001)')

        S,U,gene_names = filter_by_gene(gene_exp_filter,S,U,gene_names)

        if dataset_index == 0:
            set_intersection = set(gene_names)
        else:
            set_intersection = set_intersection.intersection(gene_names)
        
        dataset_dir_string = dir_string + '/' + dataset_names[dataset_index]
        make_dir(dataset_dir_string)
        dataset_strings.append(dataset_dir_string)

    log.info('Gene set size: '+str(len(set_intersection)))
    
    np.random.seed(seed)
    sampling_gene_set = np.sort(np.array(list(set_intersection)))
    if n_genes < len(sampling_gene_set):
        gene_select = np.random.choice(sampling_gene_set,n_genes,replace=False)
        log.info(str(n_genes)+' genes selected.')
    else:
        gene_select = sampling_gene_set
        log.warning(str(len(sampling_gene_set))+' genes selected: cannot satisfy query of '+str(n_genes)+' genes.')
    
    gene_select=list(gene_select)
    sampling_gene_set = list(sampling_gene_set)

    save_gene_list(dir_string,gene_select,'genes')
    save_gene_list(dir_string,sampling_gene_set,'gene_set')

    if viz:
        diagnostics_dir_string = dir_string + '/diagnostic_figures'
        make_dir(diagnostics_dir_string)
        for figure_ind in plt.get_fignums():
            plt.figure(figure_ind)
            plt.savefig(diagnostics_dir_string+'/{}.png'.format(dataset_names[figure_ind-1]),dpi=450)
    return dir_string,dataset_strings

########################
## Helper functions
########################

def filter_by_gene(filter,*args):
    """
    This function takes in a filter over genes,
    then selects the entries of inputs that match the filter.
    Usage: S_filt, U_filt = filter_by_gene(filter,S,U)

    Input: 
    filter: boolean or integer filter over the gene dimension.
    args: arrays with dimension 0 that matches the filter dimension.

    Output:
    tuple(out): tuple of filtered args.
    """
    out = []
    for arg in args:
        out += [arg[filter].squeeze()]
    return tuple(out)

def threshold_by_expression(S,U,
    filt_param={'min_U_mean':0.01,\
                'min_S_mean':0.01,\
                'max_U_max':350,\
                'max_S_max':350,\
                'min_U_max':4,\
                'min_S_max':4}):
    """
    This function takes in raw spliced and unspliced counts, as well as threshold parameters, and 
    outputs a boolean filter of genes that meet these thresholds.

    Input: 
    S: spliced matrix. Note this is always genes x cells.
    U: unspliced matrix. Note this is always genes x cells.

    Output:
    gene_exp_filter: genes that meet the expression thresholds.
    """
    S_max = S.max(1)
    U_max = U.max(1)
    S_mean = S.mean(1)
    U_mean = U.mean(1)  

    gene_exp_filter = \
        (U_mean > filt_param['min_U_mean']) \
        & (S_mean > filt_param['min_S_mean']) \
        & (U_max < filt_param['max_U_max']) \
        & (S_max < filt_param['max_S_max']) \
        & (U_max > filt_param['min_U_max']) \
        & (S_max > filt_param['min_S_max'])     
    log.info(str(np.sum(gene_exp_filter))+ ' genes retained after expression filter.')
    return gene_exp_filter


def save_gene_list(dir_string,gene_list,filename):
    """
    This function saves a list of genes to disk..

    Input: 
    dir_string: batch directory location.
    gene_list: list of genes.
    filename: file name string.
    """
    with open(dir_string+'/'+filename+'.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(gene_list)

def make_dir(dir_string):
    """
    This function attempts to create a directory.

    Input: 
    dir_string: directory string.
    """    
    try: 
        os.mkdir(dir_string) 
        log.info('Directory ' + dir_string+ ' created.')
    except OSError as error: 
        log.warning('Directory ' + dir_string + ' already exists.')

def import_raw(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr):
    """
    This function attempts to import raw data from a disk object (loom, adata, or mtx).

    Input: 
    filename: file location string is loom or adata, directory string if mtx.
    spliced_layer: name of spliced layer in disk object.
    unspliced_layer: name of unspliced layer in disk object.
    gene_attr: name of gene name attribute in disk object.
    cell_attr: name of cell barcode attribute in disk object.

    Output:
    Tuple of spliced matrix, unspliced matrix, gene names, number of cells.
    """

    fn_extension = filename.split('.')[-1]
    if fn_extension == 'loom': #loom file
        return import_vlm(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr)
    elif fn_extension == 'h5ad':
        return import_h5ad(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr)
    else:
        return import_mtx(filename)

def import_h5ad(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr):
    """
    Imports an anndata file with spliced and unspliced RNA counts.
    Note row/column convention is opposite loompy.
    Conventions as in import_raw.
    """
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    ds = ad.read_h5ad(filename)
    S = np.asarray(ds.layers[spliced_layer].todense()).T
    U = np.asarray(ds.layers[unspliced_layer].todense()).T
    gene_names = ds.var[gene_attr].to_numpy()
    nCells = len(ds.obs[cell_attr])
    warnings.resetwarnings()
    return S,U,gene_names,nCells

def import_mtx(dir_name):
    """
    Imports mtx files with spliced and unspliced RNA counts via anndata object.
    Note row/column convention is opposite loompy.
    Conventions as in import_raw.

    mtx files typically have *.genes.txt files with gene IDs rather than names. Beware incompatibilities.
    """
    dir_name = dir_name.rstrip('/') 
    ds = ad.read_mtx(dir_name+'/spliced.mtx')
    nCells = ds.shape[0]
    S = ds.X.todense().T
    U = ad.read_mtx(dir_name+'/unspliced.mtx').X.todense().T
    gene_names = np.loadtxt(dir_name+'/spliced.genes.txt',dtype=str)
    return S,U,gene_names,nCells

def import_vlm(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr):
    """
    Imports mtx files with spliced and unspliced RNA counts via anndata object.
    Note that there is a new deprecation warning in the h5py package 
    underlying loompy.
    Conventions as in import_raw.
    """
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    with lp.connect(filename) as ds:
        S = ds.layers[spliced_layer][:]
        U = ds.layers[unspliced_layer][:]
        gene_names = ds.ra[gene_attr]
        nCells = len(ds.ca[cell_attr])
    warnings.resetwarnings()
    return S,U,gene_names,nCells

def get_transcriptome(transcriptome_filepath,repeat_thr=15):
    """
    Imports transcriptome length/repeat from a previously generated file. 

    Input:
    transcriptome_filepath: path to the file. This is a simple space-separated file.
        The convention for each line is name - length - 5mers - 6mers -.... 50mers - more
    repeat_thr: threshold for minimum repeat length to consider. 
        By default, this is 15, and so will return number of polyA stretches of 
        length 15 or more in the gene.

    Output:
    len_dict: dictionary with structure {gene name : gene length}

    The repeat dictionary is not used in this version of the code.
    """
    repeat_dict = {}
    len_dict = {}
    thr_ind = repeat_thr-3
    with open(transcriptome_filepath,'r') as file:   
        for line in file.readlines():
            d = [i for i in line.split(' ') if i]
            repeat_dict[d[0]] =  int(d[thr_ind])
            len_dict[d[0]] =  int(d[1])
    return len_dict

def identify_annotated_genes(gene_names,feat_dict):
    """
    This function checks which gene names are unique and have annotations in a feature dictionary.

    Input:
    gene_names: string np array of gene names.
    feat_dict: annotation dictionary output by get_transcriptome.

    Output:
    ann_filt: boolean filter of genes that have annotations.
    """
    n_gen_tot = len(gene_names)
    sel_ind_annot = [k for k in range(len(gene_names)) if gene_names[k] in feat_dict]
    
    NAMES = [gene_names[k] for k in range(len(sel_ind_annot))]
    COUNTS = collections.Counter(NAMES)
    sel_ind = [x for x in sel_ind_annot if COUNTS[gene_names[x]]==1]

    log.info(str(len(gene_names))+' features observed, '+str(len(sel_ind_annot))+' match genome annotations. '
        +str(len(sel_ind))+' were unique.')

    ann_filt = np.zeros(n_gen_tot,dtype=bool)
    ann_filt[sel_ind] = True
    return ann_filt 
