import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

def get_perplexity(Di, var):
  """ 
  Adapted from code by Laurens Van der Maaten T-SNE home page (https://lvdmaaten.github.io/tsne/).
  Computes the perplexity of a Gaussian distribution centered at a givne point i.
  
  Inputs:
  =======
  Di: Array with distances from a given point (i) to all other points in the dataset
  var: variance of the gaussian centered at point (i) 
        
  Returns:
  ========
  perplexity: perplexity value
  """

  beta = 1/(2 * var)
  # Compute perplexity
  P = np.exp(-(Di**2) * beta)
  sumP = sum(P)
  H = np.log(sumP) + beta * np.sum((Di**2) * P) / sumP
  return np.exp(H)[0]

def get_Pij_row(Di, var):
  """
  Computes a row of the similarity matrix for point i based on the Gaussian distribution applied to 
  the distances from point i to all other datapoints.
  
  Adapted from code by Laurens Van der Maaten T-SNE home page (https://lvdmaaten.github.io/tsne/)
        
  Inputs:
  =======
  Di: Array with distances from a given point (i) to all other points in the dataset
  var: variance of the gaussian centered at point (i) 
        
  Returns:
  ========
  P: conditional distribution for point i
  """
  beta = 1/(2 * var) # Half precision
  # Compute P-row
  P    = np.exp(-(Di**2) * beta)
  sumP = sum(P)
  P    = P / sumP
  return P
 
def gauss_var_search(D,perplexity,tol=1e-5,n_attempts=50, beta_init=1):
    """
    Estimates the variance for Gaussian centered at each point so that the perplexity falls 
    within a certain range from the desired one.
    
    Inputs:
    =======
    D: Dissimilarity matrix between all samples
    perplexity: desired perplexity for all points
    tol: tolerance in reaching the desired perplexity
    n_attempts: maximum number of iterations when estimating the variance that leads to the desired perplexity
    beta_init: initial value of precision / 2 --> beta_init =1 --> var_init = 0.5
    
    Returns:
    ========
    var_dict: dictionary with the final values of variance estimated for each sample
    perp_search_dict: dictionary with the perplexity search paths for all points
    """
    n = D.shape[0] # Number of points
    beta     = np.ones((n, 1)) * beta_init
    var_init = 1 / 2 * beta[0]
    perp_search_dict = {}
    var_dict = {}
    
    for i in tqdm(range(n)):
        # Print progress
        #if i % 100 == 0:
        #    print("Computing P-values for point %d of %d..." % (i, n))
        # Get distance from i to all other datapoints (remove i,i entry)
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        # Initial values for beta
        betamin = -np.inf
        betamax = np.inf

        # Compute initial perplexity for initial variance = 0.5
        perplexity_init     = get_perplexity(Di, var_init)
        perp_search_dict[i] = [perplexity_init]
        
        # Compute difference between perplexity_init and desired perplexity [we do this in terms of the natural logs]
        Hdiff = np.log(perplexity_init) - np.log(perplexity)
        tries = 0
        
        # Keep updating the variance, until we get to a perplexity that is within tolerance limits to the desired one or we hit the maximum number of attempts
        while np.abs(Hdiff) > tol and tries < n_attempts:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.
        
            # Recompute the values
            new_var    = 1 / (2*beta[i]) 
            this_perplexity = get_perplexity(Di,new_var)
            this_P          = get_Pij_row(Di, new_var)
            Hdiff           = np.log(this_perplexity) - np.log(perplexity)
            tries          += 1
            perp_search_dict[i].append(this_perplexity)
        var_dict[i] = new_var[0]
        
    return var_dict, perp_search_dict
   
def get_P(D,var_dict):
    n = D.shape[0]
    P = np.zeros((n,n))
    for i in tqdm(range(n)):
        D_i    = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        var_i  = var_dict[i]
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = get_Pij_row(D_i, var_i)
    return P