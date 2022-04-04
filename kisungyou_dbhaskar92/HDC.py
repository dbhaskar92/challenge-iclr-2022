#%% Routines for Hyperbolic Diffusion Condensation

import math
import numpy as np 
import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.hyperbolic  import Hyperbolic
from geomstats.geometry.poincare_ball import PoincareBall

'''
H2toDisk : This function transforms 3-d coordinate on H^2 (hyperboloid) onto the Poincare disk 

parameters

points  (n x 3) extrinsic coordinates of H^2 embedded in R^3.

output
 
trfpts  (n x 2) coordinates on Poincare disk for visualization.
'''
def H2toDisk(points):
    
    poincare_model = PoincareBall(dim=2)
    trfpts = poincare_model.from_coordinates(points, from_coords_type='extrinsic')
    return trfpts 

'''
main routine

the algorithm computes diffusion condensation on the hyperbolic space and records all configurations along the course of iterations. current implementation is constructed based on the while-loop structure so that the the algorithm returns all intermediate results for convenient post-hoc analysis.

parameters

points   : (n x p) extrinsic coordinates of H^(p-1) embedded in R^p.
epsilon  : initial bandwidth for Gaussian kernel (default: 0.001). 
tau      : criterion to determine whether there was enough contraction compared to the previous point configuration. must be a number in (0,1). (default : 0.9).
scaler   : multiplier for bandwidth parameter to expediate convergence. must be larger than 1 (default: 1.2). 
stopping : convergence threshold to halt itertions when the diameter of an updated point set becomes smaller than this (default: 1e-5).

output - assume the algorithm is halted after T-steps.

rec_configs   : length-T list of (n x p) point-set configurations. 
rec_bandwidth : length-T list of bandwidth values.
rec_diameter  : length-T list of diameter values.
'''
def runH2DC(points, epsilon=0.001, tau=0.9, scaler=1.2, stopping=1e-5):

    # checkers
    assert isinstance(points, np.ndarray)
    assert np.ndim(points) == 2
    assert epsilon > np.finfo(float).eps*10
    assert (tau > 0) and (tau < 1)
    assert scaler > 1

    ndim = points.shape[1]-1
    nsam = points.shape[0]
    hyper_model = Hyperbolic(dim=ndim)
    assert np.all(hyper_model.belongs(points))

    # initialize
    rec_configs   = []
    rec_bandwidth = []
    rec_diameter  = []

    par_eps  = epsilon
    old_X    = points
    old_dist = hyper_model.metric.dist_pairwise(old_X)
    old_diam = np.max(old_dist)

    # iterate
    m_eps = np.finfo(float).eps*100
    op_mean = FrechetMean(metric=hyper_model.metric)
    not_converged = True
    iteration = 0
    while not_converged:
        iteration += 1
        kernmat = np.exp(-(old_dist**2)/par_eps)

        new_X = np.zeros(points.shape)
        for i in np.arange(nsam):
            # adjust weights per row
            tgt = kernmat[i].flatten()
            wgt = tgt/np.sum(tgt)

            # compute Frechet mean with churning out the non-significant pts 
            if np.any(wgt <= m_eps):
                subid  = np.where(wgt > m_eps)[0]
                subpts = old_X[subid]
                subwgt = wgt[subid]/np.sum(wgt[subid])
                assert subpts.shape[0] == len(subwgt)
                op_mean.fit(subpts, weights=subwgt)
            else:
                op_mean.fit(points, weights=wgt)
            
            # assign
            new_X[i] = op_mean.estimate_.flatten()
        
        new_dist = hyper_model.metric.dist_pairwise(new_X)
        new_diam = np.max(new_dist)

        # stop if the updated configuration is small enough
        if new_diam < stopping:
            rec_configs.append(new_X)
            rec_bandwidth.append(par_eps)
            rec_diameter.append(new_diam)
            not_converged = False 
            break 
        else:
            if new_diam < (tau*old_diam):
                rec_configs.append(new_X)
                rec_bandwidth.append(par_eps)
                rec_diameter.append(new_diam)

                old_X = new_X 
                old_dist = new_dist 
                old_diam = new_diam 
            else:
                rec_configs.append(old_X)
                rec_bandwidth.append(par_eps)
                rec_diameter.append(old_diam)
                par_eps = par_eps*scaler

    # return the outputs
    return rec_configs, rec_bandwidth, rec_diameter
