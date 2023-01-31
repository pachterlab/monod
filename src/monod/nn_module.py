import os
import importlib_resources

import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# YC added
try:
    package_resources = importlib_resources.files("monod")
    model_path = os.path.join(package_resources,'models/best_model_MODEL.zip')
    print(package_resources)
except:
    import sys
    package_resources = importlib_resources.files("models")
    model_path = os.path.join(package_resources,'best_model_MODEL.zip')

npdf = 10

# load in model
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
        norm = torch.tensor(norm)
        return norm
    if quantiles == 'cheb':
        n = np.arange(npdf)
        q = np.flip((np.cos((2*(n+1)-1)/(2*npdf)*np.pi)+1)/2)

        norm = stats.norm.ppf(q)
        norm = torch.tensor(norm)
        return norm

NORM = get_NORM(10).to(torch.device(device))
norm = NORM


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

    # grid_i = grid[:,i].reshape((-1,1))

    # r = r[:,i]
    # w = w[:,i].reshape((-1,1))
    # p_nb = p_nb[:,i]


    y_ = m * torch.log(grid + eps) - grid - torch.lgamma(m+1)

    if (p_nb > 1e-10).any():
      index = [p_nb > 1e-10]
      y_[index] += torch.special.gammaln(grid[index]+r[index]) - torch.special.gammaln(r[index])                 - grid[index]*torch.log(r[index] + grid[index]) + grid[index]                 + r[index]*torch.log(r[index]/(r[index]+grid[index]))

    y_ = torch.exp(y_)
    y_weighted = w*y_
    Y = y_weighted.sum(axis=1)

    EPS = 1e-40
    Y[Y<EPS]=EPS
    return Y


def log_prob_nnNB(p : np.array, n: np.array, m: np.array,  eps : float = 1e-15):
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
    print(pv.size())
    w_,hyp_= model(pv)


    n = n.reshape(-1,1)
    m = m.reshape(-1,1)
    # get conditional probabilites
    ypred_cond = get_ypred_at_RT(pv,w_,hyp_,n,m,norm)

    # multiply conditionals P(m|n) by P(n)
    prob_nascent = torch.exp(prob_nascent)

    predicted = prob_nascent * ypred_cond.reshape((prob_nascent.shape))
    log_P = torch.log(predicted+eps).detach().cpu().numpy()


    return(log_P)
