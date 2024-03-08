import os
import importlib_resources

import numpy as np
import scipy
from scipy import stats


import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-18

def bursty_none_grid_nn_10(p,lm):
    """Evaluate the PMF of the model over a grid at a set of parameters using the 10 basis neural approximation.

    Parameters
    ----------
    p: np.ndarray
        log10 biological parameters.
    limits: list of int
        grid size for PMF evaluation, size n_species.


    Returns
    -------
    Pss: np.ndarray
        the steady-state model PMF over a grid (0,...,limits[0]-1) x (0,...,limits[1]-1). NOT log of PMF.
    """
    p = 10**p
    n,m = np.arange(lm[0]),np.arange(lm[1])
    n,m = np.meshgrid(n,m)
    Pss = np.exp(log_prob_nn_10(p,n.T,m.T))

    return Pss
    
def bursty_none_grid_microstate(p,lm):
    """Evaluate the PMF of the model over a grid at a set of parameters using the 10 basis neural approximation.

    Parameters
    ----------
    p: np.ndarray
        log10 biological parameters.
    limits: list of int
        grid size for PMF evaluation, size n_species.


    Returns
    -------
    Pss: np.ndarray
        the steady-state model PMF over a grid (0,...,limits[0]-1) x (0,...,limits[1]-1). NOT log of PMF.
    """
    n,m = np.arange(lm[0]),np.arange(lm[1])
    Pss = get_prob_joint(p,n,m)
    return Pss


def bursty_none_logL_microstate(p,data):
    """Evaluate the logL of data microstate.

    Parameters
    ----------
    p: np.ndarray
        log10 biological parameters.
    data: size (microstate,2) where the data[:,0] is n values and [:,1] is m values of microstates. 

    Returns
    -------
    np.log(P): np.ndarray
        the LOG probability of input (n,m) microstates.
    """
    n,m = data[:,0],data[:,1]
    P = get_prob_joint(p,n,m,grid=False)
    return np.log(P)


# for 10 basis functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):

    def __init__(self, input_dim, npdf, h1_dim, h2_dim):
        super().__init__()

        self.input = nn.Linear(input_dim, h1_dim)
        self.hidden = nn.Linear(h1_dim, h2_dim)
        self.output = nn.Linear(h2_dim, npdf)

        self.hyp = nn.Linear(h1_dim,1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = torch.sigmoid


    def forward(self, inputs):

        # pass inputs to first layer, apply sigmoid
        l_1 = self.sigmoid(self.input(inputs))

        # pass to second layer, apply sigmoid
        l_2 = self.sigmoid(self.hidden(l_1))

        # pass to output layer
        w_un = (self.output(l_2))

        # pass out hyperparameter, sigmoid so it is between 0 and 1, then scale between 1 and 6
        hyp = self.sigmoid(self.hyp(l_2))

        # apply softmax
        w_pred = self.softmax(w_un)

        return w_pred,hyp

try:
    package_resources = importlib_resources.files("monod")
    model_path = os.path.join(package_resources,'models/best_model_MODEL.zip')
    model_microstate_path = os.path.join(package_resources,"models/3npdf_2hl_256hu_100bs_0.001lr_update_256_MODEL")

except:
    import sys
    package_resources = importlib_resources.files("models")
    model_path = os.path.join(package_resources,'best_model_MODEL.zip')
    model_microstate_path = os.path.join(package_resources,"models/3npdf_2hl_256hu_100bs_0.001lr_update_256_MODEL")


# load in model for 10 basis functions
model = MLP(7,10,256,256)
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(torch.device(device))


def get_NORM(npdf,quantiles='cheb'):
    '''' Returns quantiles based on the number of kernel functions npdf.
    Chebyshev or linear, with chebyshev as default.
    '''
    if quantiles == 'lin':
        q = np.linspace(0,1,npdf+2)[1:-1]
        norm = stats.norm.ppf(q)
#         norm = torch.tensor(norm)
        return norm
    if quantiles == 'cheb':
        n = np.arange(npdf)
        q = np.flip((np.cos((2*(n+1)-1)/(2*npdf)*np.pi)+1)/2)

        norm = stats.norm.ppf(q)
#         norm = torch.tensor(norm)
        return norm

NORM_10 = torch.tensor(get_NORM(10)).to(torch.device(device))
NORM_5 = get_NORM(5)
NORM_3 = get_NORM(3)

def generate_grid(logmean_cond,logstd_cond,norm):
    ''' Generate grid of kernel means based on the log mean and log standard devation of a conditional distribution.
    Generates the grid of quantile values in NORM, scaled by conditional moments.
    '''

    logmean_cond = torch.reshape(logmean_cond,(-1,1))
    logstd_cond = torch.reshape(logstd_cond,(-1,1))
    translin = torch.exp(torch.add(logmean_cond,logstd_cond*norm))

    return translin

def get_ypred_at_RT(p,w,hyp,n,m,norm,eps=1e-8):
    '''Given a parameter vector (tensor) and weights (tensor), and hyperparameter,
    calculates ypred (Y), or approximate probability. Calculates over array of nascent (n) and mature (m) values.
    '''

    p_vec = 10**p[:,0:3]
    logmean_cond = p[:,3]
    logstd_cond = p[:,4]

    hyp = hyp*5+1

    grid = generate_grid(logmean_cond,logstd_cond,norm)
    s = torch.zeros((len(n),10)).to(torch.device(device))
    s[:,:-1] = torch.diff(grid,axis=1)
    s *= hyp
    s[:,-1] = torch.sqrt(grid[:,-1])


    v = s**2
    r = grid**2/((v-grid)+eps)
    p_nb = 1-grid/v

    Y = torch.zeros((len(n),1)).to(torch.device(device))


    y_ = m * torch.log(grid + eps) - grid - torch.lgamma(m+1)
    m_array = m.repeat(1,10)

    if (p_nb > 1e-10).any():
      index = [p_nb > 1e-10]
      y_[index] += torch.special.gammaln(m_array[index]+r[index]) - torch.special.gammaln(r[index]) \
                    - m_array[index]*torch.log(r[index] + grid[index]) + grid[index] + r[index]*torch.log(r[index]/(r[index]+grid[index]))

    y_ = torch.exp(y_)
    y_weighted = w*y_
    Y = y_weighted.sum(axis=1)

    EPS = 1e-40
    Y[Y<EPS]=EPS
    return Y

def log_prob_nn_10(p : np.array, n: np.array, m: np.array,  eps : float = 1e-15):
    ''' Calculates probability for bursty model given the most accurate trained model.
      -----------------------------------
      n,m
        nascent and mature values over which to calculate probability. Shape of n must equal shape of m.
      p
        parameters for bursty model: b,beta,gamma (NOT log10)
      eps
        numerical stability constant
    '''
    n,m = torch.tensor(n).to(torch.device(device)), torch.tensor(m).to(torch.device(device))
    b,beta,gamma = torch.tensor(p)
    b,beta,gamma = torch.ones_like(n)*b, torch.ones_like(n)*beta,torch.ones_like(n)*gamma
    b,beta,gamma= b.to(torch.device(device)),beta.to(torch.device(device)),gamma.to(torch.device(device))
    
    mu1 = b/beta
    mu2 = b/gamma
    
    r = 1/beta
    
    # negative binomial of nascent RNA n
    prob_nascent = torch.lgamma(n+r) - torch.lgamma(n+1) - torch.lgamma(r) + r * torch.log(r/(r+mu1)+eps) + n * torch.log(mu1/(r+mu1)+eps)

    
    # get moments
    var1 = mu1 * (1+b)
    var2 = mu2 * (1+b*beta/(beta+gamma))
    cov = b**2/(beta+gamma)

    # calculate conditional moments
    logvar1 = torch.log((var1/mu1**2)+1)
    logvar2 = torch.log((var2/mu2**2)+1)
    logstd1 = torch.sqrt(logvar1)
    logstd2 = torch.sqrt(logvar2)

    logmean1 = torch.log(mu1**2/torch.sqrt(var1+mu1**2))
    logmean2 = torch.log(mu2**2/torch.sqrt(var2+mu2**2))

    val = (logmean1 + logmean2 + (logvar1 + logvar2)/2)
    val[val<-88] = -88
    logcov = torch.log(cov * torch.exp(-(val)) +1 )
    logcorr = logcov/torch.sqrt(logvar1 * logvar2)

    logmean_cond = logmean2 + logcorr * logstd2/logstd1 * (torch.log(n+1) - logmean1)
    logvar_cond = logvar2 * (1-logcorr**2)
    logstd_cond = logstd2 * torch.sqrt(1-logcorr**2)

    xmax_m = torch.ceil(torch.ceil(mu2) + 4*torch.sqrt(var2))
    xmax_m = torch.clip(xmax_m,30,np.inf).int()
    
    # reshape and stack
    pv = torch.column_stack((torch.log10(b).reshape(-1),
                             torch.log10(beta).reshape(-1),
                             torch.log10(gamma).reshape(-1),
                             logmean_cond.reshape(-1),
                             logstd_cond.reshape(-1),
                             xmax_m.reshape(-1),
                             n.reshape(-1)
                             ))
    # run through model
    pv = pv.to(torch.float32)

    w_,hyp_= model(pv)


    n = n.reshape(-1,1)
    m = m.reshape(-1,1)
    
    # get conditional probabilites
    ypred_cond = get_ypred_at_RT(pv,w_,hyp_,n,m,NORM_10)

    # multiply conditionals P(m|n) by P(n)
    prob_nascent = torch.exp(prob_nascent)


    predicted = prob_nascent * ypred_cond.reshape((prob_nascent.shape))
    log_P = torch.log(predicted+eps).detach().cpu().numpy()


    return(log_P)


# now, for the microstate model 

# LOAD IN BEST MODEL

# this model learns weights AND returns scaled means and scaled standard deviations
class MLP_weights_scale(nn.Module):

    def __init__(self, input_size, npdf, num_layers, num_units, activate = 'relu',
                final_activation = 'sigmoid', max_mv = 'update', max_val = 2.0):
                   
        super().__init__()
        self.module_list = nn.ModuleList([])
        self.module_list.append(nn.Linear(input_size,num_units,bias=False))
        for k in range(num_layers-1):
            self.module_list.append(nn.Linear(num_units, num_units,bias=False))
        self.module_list.append(nn.Linear(num_units,npdf))
        
        self.scaling_mean = nn.Linear(num_units,npdf)
        self.scaling_std = nn.Linear(num_units,npdf)
        
        self.npdf = npdf
        self.activate = activate
        self.max_mv = max_mv                        
        self.final_activation = final_activation
        self.softmax = nn.Softmax(dim=1)
                              
        if self.max_mv == 'update':
            self.max_mean = nn.Parameter(torch.tensor(max_val,dtype=torch.float32),requires_grad=True)
            self.max_std = nn.Parameter(torch.tensor(max_val,dtype=torch.float32),requires_grad=True)
        else:
            self.max_mean = torch.tensor(max_val,dtype=torch.float32)
            self.max_std = torch.tensor(max_val,dtype=torch.float32)



    def forward(self, x):
        
        # store the unscaled means and standard deviations
        unscaled_means = x[:,4:4+self.npdf]
        unscaled_stds = x[:,4+self.npdf:]
        
        # pass through first layer 
        x = F.relu(self.module_list[0](x))

        # pass through next layers                      
        for f in self.module_list[1:-1]:
            x = f(x)
            if self.activate == 'relu':
                x = F.relu(x)
            elif self.activate == 'sigmoid':
                x = torch.sigmoid(x)
            elif self.activate == 'sin':
                x = torch.sin(x)
                
        scaling_factors_mean = torch.sigmoid(self.scaling_mean(x))
        scaling_factors_std = torch.sigmoid(self.scaling_std(x))  
        
        # apply softmax to weights
        w_pred = self.softmax(self.module_list[-1](x))   
        
        # calculate the scaled means and stds                       
        c_mean = scaling_factors_mean*(self.max_mean-(1/self.max_mean)) + (1/self.max_mean)
        c_std = scaling_factors_std*(self.max_std-(1/self.max_std)) + (1/self.max_std)
        scaled_means = c_mean*unscaled_means
        scaled_stds = c_std*unscaled_stds    

            
        return w_pred,scaled_means,scaled_stds

# load in microstate model
# model_microstate5 = MLP_weights_scale(input_size = 4+5*2,
#                             npdf = 5,
#                             num_layers = 2,
#                             num_units = 512,
#                             activate = "relu",
#                             final_activation = "sigmoid",
#                             max_mv = "update",
#                             max_val = 5.0,
#                             )
# model_microstate5.load_state_dict(torch.load(model_microstate_path))
# model_microstate5.eval()
# model_microstate5.to(torch.device(device))

model_microstate3 = MLP_weights_scale(input_size = 4+3*2,
                            npdf = 3,
                            num_layers = 2,
                            num_units = 256,
                            activate = "relu",
                            final_activation = "sigmoid",
                            max_mv = "update",
                            max_val = 5.0,
                            )


model_microstate3.load_state_dict(torch.load(model_microstate_path))
model_microstate3.eval()
model_microstate3.to(torch.device(device))
    


def get_moments(b,beta,gamma):
    ''' Returns mean, variance, standard deviation and max_n, max_m (mean_z + 4*std_z, where z is species) for
    nascent and mature molecules given parameters included in p.
    
    -------
    
    parameter p:
        b, beta, gamma
    '''

    
    mu_n = b/beta
    mu_m = b/gamma
    
    var_n = (mu_n)*(b + 1)
    var_m = (mu_m)*(b*beta/(beta+gamma) + 1)

    std_n,std_m = np.sqrt(var_n),np.sqrt(var_m)
    
    COV = b**2/(beta + gamma)
    
    return mu_n,mu_m,var_n,var_m,std_n,std_m,COV

def get_conditional_moments(MU, VAR, COV, n):
    ''' Get moments of conditional distributions (lognormal moment matching) given overall distribution
    mearn, variance, standard deviation, covariance over a range of nascent values.
    '''
    logvar = np.log((VAR/MU**2)+1)
    logstd = np.sqrt(logvar)
    logmean = np.log(MU**2/np.sqrt(VAR+MU**2))

    logcov = np.log(COV * np.exp(-(logmean.sum()+logvar.sum()/2)) +1 ) 
    logcorr = logcov/np.sqrt(logvar.prod())

    logmean_cond = logmean[1] + logcorr * logstd[1]/logstd[0] * (np.log(n+1) - logmean[0])
    logstd_cond = logstd[1] * np.sqrt(1-logcorr**2)   
   
    return(logmean_cond,logstd_cond)
    
def get_quantile_moments(logmean_cond, logstd_cond, NORM):
    ''' Returns the conditional quantile moments mu_1 to mu_npdf and std_1 to std_npdf for conditional m distributions.
    Moment matches to lognormal, returns quantiles.
    '''
    
    logmean_cond = np.reshape(logmean_cond,(-1,1))
    logstd_cond = np.reshape(logstd_cond,(-1,1))
    mus = np.exp(np.add(logmean_cond,logstd_cond*NORM))
    
    stds = np.zeros_like(mus) 
    stds[:,:-1] = np.diff(mus,axis=1)
    stds[:,-1] = np.sqrt(mus[:,-1])

    
    return mus, stds

def eval_cond_P_joint_(m,weights,cond_means,cond_stds,grid):   
        ''' Returns condP of the model trained on joint probabilities given weights, conditional means and standard deviations. Must specify whether this is a grid or individual microstates. 
        '''

        p_nb = 1 - cond_means/cond_stds**2
        poisson_index = p_nb < 1e-4
        nb_index = p_nb >= 1e-4
        cond_stds[poisson_index] = torch.sqrt(cond_means[poisson_index]*1.05)
        cond_rs = (cond_means**2/(cond_stds**2-cond_means))

        # reshape
        if grid == True:
            filt2 = nb_index[:,:,None].repeat(1,1,len(m))
            cond_means = cond_means[:,:,None]
            cond_rs = cond_rs[:,:,None]
            weights = weights[:,:,None]
        elif grid == False:
            filt2 = nb_index
            m = m[:,None]
        
        # poisson
        y_ = m * torch.log(cond_means + eps) - cond_means - torch.lgamma(m+1)
        
        # negative binomial
        step1 = torch.special.gammaln(m + cond_rs + eps)\
                - torch.special.gammaln(cond_rs + eps)\
                - m * torch.log(cond_rs + cond_means + eps)\
                + cond_means\
                + cond_rs * torch.log(cond_rs/(cond_rs + cond_means + eps) + eps)
        
        y_[filt2] += step1[filt2]         
        
        y_ = torch.mul(weights,torch.exp(y_))  
        Y = y_.sum(axis=1)
        
        return(Y)
    

def get_prob_joint(p,n_values,m_values,model=model_microstate3,npdf=3,grid=True):
    
    ''' Gets probability for grid of size (len(n_values),len(m_values)) if grid == True, or individual microstates if grid == False. 
    --------------
    p: log10 p
    n_values: values of n at which to calculate the probability
    m_values: corresponding m values, must be same size as n
    model: neural network model
    npdf: number of basis functions
    grid: calculate over grid or at individual microstates

    '''
    print('getting prob joint')
    b,beta,gamma = 10**p  
    num_n = len(n_values)
    

    mu_n,mu_m,var_n,var_m,std_n,std_m,COV = get_moments(b,beta,gamma)

    # get moment matched conditional means and stds
    logmean_cond, logstd_cond = get_conditional_moments(np.array([mu_n,mu_m]), np.array([var_n,var_m]), COV, n_values)

    # get the conditional grid parameters
    if npdf == 3:
        NORM = NORM_3
    elif npdf == 5:
        NORM = NORM_5
    elif npdf == 10:
        NORM = NORM_10
    mus, stds = get_quantile_moments(logmean_cond,logstd_cond,NORM)
    
    # create param to pass through NN
    param_ = torch.ones(num_n,4+2*npdf)
    
    param_[:,:3] = torch.tensor(p).repeat(num_n,1)
    param_[:,4:4+npdf] = torch.tensor(mus)
    param_[:,4+npdf:] = torch.tensor(stds)
    
    param_[:,3] = torch.tensor(n_values)
    param_ = param_.to(device)


    # pass through model
    weights, cond_means, cond_stds = model.forward(param_)
    m = torch.tensor(m_values).to(device)
    
    # get conditional probability
    p_cond = eval_cond_P_joint_(m,weights,cond_means,cond_stds,grid).detach().cpu().numpy()
    
    # negative binomial parameters
    n_binom = mu_n**2/(var_n - mu_n)
    p_binom = mu_n/var_n
    
    # nascent marginal probability
    comb =  scipy.special.loggamma(n_values + n_binom) - scipy.special.loggamma(n_binom) - scipy.special.loggamma(n_values + 1) 
    log_p_nascent = comb + n_binom*np.log(p_binom) + n_values*np.log(1-p_binom)
      

    if grid == True:
        prob = np.exp(log_p_nascent[:,None])*p_cond
    elif grid == False:
        prob = np.exp(log_p_nascent)*p_cond

    # clip tiny probabilities
    prob[prob < eps] += (eps - prob[prob<eps])
    
    return(prob)




