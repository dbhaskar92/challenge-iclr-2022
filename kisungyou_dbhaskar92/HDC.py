#%% Routines for Hyperbolic Diffusion Condensation

import math
import numpy as np 
import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats import visualization
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.hyperbolic  import Hyperbolic
from geomstats.geometry.poincare_ball import PoincareBall

class HDC:

    def __init__(self, 
                data,                       # data coordinates
                labels=None,                # length n vector of labels for each node in the data,
                coord_type="extrinsic",     # 'extrinsic' or 'intrinsic'
                epsilon=0.001,              # initial bandwidth for Gaussian kernel
                tau=0.9,                    # threshold for contraction compared to previous point configuration (must be between 0 & 1)
                scalar=1.2,                 # multiplier for bandwidth parameter to expedite convergence (must be larger than 1)
                stopping=1e-5,              # convergence threshold to halt itertions based on diameter of the updated point set
                verbose=False
    ):
        # Sanity checks
        assert isinstance(data, np.ndarray)
        assert np.ndim(data) == 2
        assert epsilon > np.finfo(float).eps*10
        assert (tau > 0) and (tau < 1)
        assert scalar > 1
        # Initialize
        self.data = data.copy()
        self.labels = labels
        self.coord_type = coord_type
        self.epsilon = epsilon
        self.tau = tau
        self.scalar = scalar
        self.stopping = stopping
        self.verbose = verbose
        self.DC_configs = None
        self.DC_bandwidths = None
        self.DC_diameters = None
        self.ndim = None

    def H2toDisk(self, points):
        '''
        H2toDisk : This function transforms 3-d coordinate on H^2 (hyperboloid) onto the Poincare disk 
        parameters
        points  (n x 3) extrinsic coordinates of H^2 embedded in R^3.
        output
        trfpts  (n x 2) coordinates on Poincare disk for visualization.
        '''
        poincare_model = PoincareBall(dim=2)
        trfpts = poincare_model.from_coordinates(points, from_coords_type="extrinsic") # from_coords_type=self.coord_type)
        return trfpts 

    def diffusion_condensation(self):
        '''
        Computes diffusion condensation on the hyperbolic space and records state for each iteration. 
        
        Current implementation is based on the while-loop structure so that the the algorithm returns all intermediate results 
        for convenient post-hoc analysis.

        Returns
        -------
        rec_configs   : list of (n x p) point-set configurations over DC iterations
        rec_bandwidth : list of bandwidth values over DC iterations
        rec_diameter  : list of diameter values over DC iterations
        '''
        nsam = self.data.shape[0]

        if self.coord_type == "extrinsic":
            self.ndim = self.data.shape[1]-1
        elif self.coord_type == "intrinsic":
            self.ndim = self.data.shape[1]
        else:
            raise ValueError

        # model space
        hyper_model = Hyperbolic(dim=self.ndim, default_coords_type="extrinsic")
        self.data = hyper_model.from_coordinates(self.data, self.coord_type)
        if not np.all(hyper_model.belongs(self.data)):
            raise ValueError("* Some (or all) of the data points do not reside on Hyperboloid.")

        # initialize
        self.DC_configs = []
        self.DC_bandwidths = []
        self.DC_diameters = []

        par_eps  = self.epsilon
        old_X    = self.data
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

            new_X = np.zeros(self.data.shape)
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
                    op_mean.fit(self.data, weights=wgt)
                
                # assign
                new_X[i] = op_mean.estimate_.flatten()
            
            new_dist = hyper_model.metric.dist_pairwise(new_X)
            new_diam = np.max(new_dist)

            # stop if the updated configuration is small enough
            if new_diam < self.stopping:
                self.DC_configs.append(new_X)
                self.DC_bandwidths.append(par_eps)
                self.DC_diameters.append(new_diam)
                not_converged = False 
                break 
            else:
                if new_diam < (self.tau*old_diam):
                    self.DC_configs.append(new_X)
                    self.DC_bandwidths.append(par_eps)
                    self.DC_diameters.append(new_diam)

                    old_X = new_X 
                    old_dist = new_dist 
                    old_diam = new_diam 
                else:
                    self.DC_configs.append(old_X)
                    self.DC_bandwidths.append(par_eps)
                    self.DC_diameters.append(old_diam)
                    par_eps = par_eps*self.scalar

        
    def visualize_condensation_steps(self):

        if self.DC_configs == None:
            self.diffusion_condensation()

        fig, axes = plt.subplots(1,2, figsize=(20,6), dpi=200)

        axes[0].plot(np.arange(len(self.DC_configs)), self.DC_bandwidths)
        axes[0].set_title("Bandwidth", fontsize=20)
        axes[0].set_xlabel("iterations", fontsize=15)

        axes[1].plot(np.arange(len(self.DC_configs)), self.DC_diameters)
        axes[1].set_title("Diameter", fontsize=20)
        axes[1].set_xlabel("iterations", fontsize=15)

        plt.show()


    def visualize_DC_embeddings(self, **kwargs):

        # visualize 10 intermittent values
        show_nums = len(self.DC_configs)
        show_ids  = np.linspace(0, show_nums-1, num=10)

        circle = visualization.PoincareDisk(point_type="ball")
        colors = [{ni: indi for indi, ni in enumerate(set(self.labels))}[ni] for ni in self.labels]

        fig, axes = plt.subplots(2,5,figsize=(25,10))
        axes = axes.flatten()

        for i in np.arange(10):

            # get the 2-dimensional embedding
            now_id = round(show_ids[i])
            vis2d = self.H2toDisk(self.DC_configs[now_id])

            # draw
            axes[i].axes.xaxis.set_visible(False)
            axes[i].axes.yaxis.set_visible(False)
            circle.set_ax(axes[i])
            circle.draw(ax=axes[i])

            for i_embedding, embedding in enumerate(vis2d):
                x = embedding[0]
                y = embedding[1]
                pt_id = i_embedding
                axes[i].scatter(x, y, c=colors[i_embedding], s=25, cmap="plasma")
                #axes[i].annotate(self.labels[i_embedding], xy=(x, y), xytext=(x, y), textcoords='offset points', ha='right', va='bottom')
            
            axes[i].set_title("Iter {}| bandwidth={:.2f}| diameter={:.2f}".format(now_id+1, self.DC_bandwidths[now_id], self.DC_diameters[now_id]))

        plt.show()