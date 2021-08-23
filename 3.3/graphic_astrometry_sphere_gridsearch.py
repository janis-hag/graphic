#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: cheetham, matthews
"""
import numpy as np
import time
import os
import pickle
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from astropy.io import fits
from configparser import ConfigParser
import argparse
import graphic_companion_fitting_lib as graphic_cfl
import graphic_contrast_lib as graphic_cl
import sphereastrometry
import pca

from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import (gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma)


#########################################################################################################
#########################################################################################################


parser = argparse.ArgumentParser(description='Calculates astrometry of a companion using a grid search.')
parser.add_argument('--config', action="store", dest="config_file", type=str, default=None)

args = parser.parse_args()
if not os.path.exists(args.config_file):
    raise FileNotFoundError(f'No config file found at {args.config_file}')

config =ConfigParser()
config.read(args.config_file)

## extract values from config file
dataset = config.get('settings','dataset',fallback='')
prefix = config.get('settings','prefix',fallback='')
n_grid = config.getint('settings','n_grid',fallback=5)
pscale = config.get('settings','plate_scale',fallback='sph_h2')
stellar_error = config.getfloat('settings','stellar_error',fallback=3)
fwhm = config.getfloat('settings','fwhm',fallback=4)
image_uncert = config.getfloat('settings','image_uncert',fallback=10)
pix_rad = config.getint('settings','pix_rad',fallback=5)

sep_guess = config.getfloat('settings','sep_guess',fallback=44)
sep_range = config.getfloat('settings','sep_range',fallback=0.5)
pa_guess = config.getfloat('settings','pa_guess',fallback=0)
pa_range = config.getfloat('settings','pa_range',fallback=1.)
contrast_guess = config.getfloat('settings','contrast_guess',fallback=1e-5)
contrast_range = config.getfloat('settings','contrast_range',fallback=1e-6)

init_params = [sep_guess, pa_guess, contrast_guess]
range_params = [sep_range, pa_range, contrast_range]

calc_im_uncertainty = config.getboolean('functions','calc_im_uncertainty',fallback=False)
calc_psf_fwhm = config.getboolean('functions','calc_psf_fwhm',fallback=False)
test_likelihood = config.getboolean('functions','test_likelihood',fallback=False)
run_grid = config.getboolean('functions','run_grid',fallback=False)
plot_grid = config.getboolean('functions','plot_grid',fallback=False)
calc_values = config.getboolean('functions','calc_values',fallback=False)


#########################################################################################################
#########################################################################################################

fit_params = [init_params[0],init_params[1],pix_rad]

# platescale
if pscale == 'sph_h2':
    platescale = 12.255
else:
    raise ValueError('platescale {} not recognized'.format(pscale))

# PCA settings
pca_settings = {'n_modes':12,
                'n_annuli':1,
                'r_min':init_params[0]-2.0*fwhm,
                'r_max':init_params[0]+2.0*fwhm,
                'fwhm':fwhm,
                'n_fwhm':0.75,
                'arc_length':1e4,
                'min_reference_frames':15,
                'threads':6,
                'median_combine':True}

# Load the image
cube_fname = dataset+'master_cube_PCA.fits'
parang_fname = dataset+'parallactic_angle.txt'
psf_fname = dataset+'flux.fits'

cube,cube_hdr = fits.getdata(cube_fname,header=True)
parangs_deg = np.loadtxt(parang_fname)
psf,psf_hdr = fits.getdata(psf_fname,header=True)

parangs_rad = parangs_deg*np.pi/180.

# Make a folder for outputs of this process
if not os.path.isdir(dataset+'comp_values/'):
    os.mkdir(dataset+'comp_values/')
dataset = dataset+'comp_values/'

log_file = dataset+'astrometry.log'
if not prefix == '':
    prefix += '_'
grid_file = dataset+prefix+'gridsearch.pick'
values_file = dataset+prefix+'final_values.txt'

log_file = dataset+'astrometry.log'
f_log = open(log_file,'a')
f_log.write('command astrometry_hd4113.py run at '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+'\n')
f_log.write('config file: '+args.config_file+'\n\n')

f_log.write('calc_im_uncertainty = {}\n'.format(calc_im_uncertainty))
f_log.write('calc_psf_fwhm = {}\n'.format(calc_psf_fwhm))
f_log.write('test_likelihood = {}\n'.format(test_likelihood))
f_log.write('run_grid = {}\n'.format(run_grid))
f_log.write('plot_grid = {}\n'.format(plot_grid))
f_log.write('calc_values = {}\n'.format(calc_values))
f_log.write('\n')
f_log.close()
f_log = open(log_file,'a')

# Correct the flux frame for the exposure time differences
flux_factor = graphic_cl.correct_flux_frame_sphere(cube_hdr,psf_hdr)
psf *= flux_factor

#########################
if calc_im_uncertainty:
    # Run PCA using the same settings
    pca_cube = pca.smart_annular_pca(cube,pca_settings['n_modes'],None,
                   parangs_deg,n_fwhm=pca_settings['n_fwhm'],fwhm=pca_settings['fwhm'],
                   n_annuli=pca_settings['n_annuli'],r_min=pca_settings['r_min'],
                   r_max=pca_settings['r_max'],arc_length=pca_settings['arc_length'],
                   min_reference_frames=pca_settings['min_reference_frames'],
                   threads=pca_settings['threads'],silent=True)
    derot_cube = pca.derotate_and_combine_multi(pca_cube, parangs_deg, save_name=None,
                     median_combine=False, threads=pca_settings['threads'],silent=True)
    final_im = np.nanmean(derot_cube,axis = 0)

    # Mask out the companion
    comp_x = int(init_params[0]*np.sin(-init_params[1]*np.pi/180.)+final_im.shape[0]/2)
    comp_y = int(init_params[0]*np.cos(-init_params[1]*np.pi/180.)+final_im.shape[0]/2)

    final_im[comp_y-5:comp_y+5,comp_x-5:comp_x+5] = np.nan
    
    plt.figure(2)
    plt.clf()
    plt.imshow(final_im,origin='lower')
    plt.savefig(dataset+prefix+'masked_companion.pdf')

    # Finally calculate the standard deviation of the image (ignoring the part with the companion)
    f_log.write('Standard deviation of pixel values around comp: {}\n'.format(np.nanstd(final_im)))
    image_uncert = np.nanstd(final_im)


#########################
if calc_psf_fwhm:

    # some (hard-coded) parameters!
    crop = 10
    fwhmguess = fwhm

    # flux array is psf. crop to a small box for efficient fitting.
    cx = int(psf.shape[0]/2)
    cy = int(psf.shape[1]/2)

    psf_to_fit = psf[cx-crop:cx+crop,cy-crop:cy+crop]

    # calculate some starting values
    ampl = np.max(psf_to_fit)

    #model: 2D gaussian
    gauss = models.Gaussian2D(amplitude=ampl, x_mean=crop, y_mean=crop, 
                              x_stddev=fwhm*gaussian_fwhm_to_sigma,
                              y_stddev=fwhm*gaussian_fwhm_to_sigma)

    # fit with Levenberg-Marquardt
    fitter = LevMarLSQFitter()
    y, x = np.indices(psf_to_fit.shape)
    fit_psf = fitter(gauss, x, y, psf_to_fit)

    fwhm_y = fit_psf.y_stddev.value*gaussian_sigma_to_fwhm
    fwhm_x = fit_psf.x_stddev.value*gaussian_sigma_to_fwhm

    fwhm = (fwhm_y+fwhm_x)/2


#########################
if test_likelihood:
    t0 = time.time()
    loglike = graphic_cfl.neg_fake_planet_loglikelihood(init_params,
                  cube=cube, parangs_rad=parangs_rad, psf=psf, plot=False,
                  uncert=image_uncert, pca_settings=pca_settings, fit_params=fit_params)
    t1 = time.time()
    chi2 = -2.*loglike

    f_log.write('Time taken for 1 iteration: {:.5f} secs\n'.format(t1-t0))
    f_log.write('Est time for n_grid^3 iterations (current n_grid={}): {:.2f}hrs\n'.format(n_grid,n_grid**3*(t1-t0)/3600.))
    f_log.write('Est time for 1k(=10^3) iterations: {:.2f}hrs\n'.format(1e3*(t1-t0)/3600.))
    f_log.write('Test chi2:{:.2f}\n'.format(chi2))

f_log.close()
f_log = open(log_file,'a')

#########################
if run_grid:

    f_log.write('image_uncert: {}\n'.format(image_uncert))
    f_log.write('fwhm value: {}\n'.format(fwhm))

    # Run a grid over the params, calculating the chi2 at each point
    seps = np.linspace(init_params[0]-range_params[0],init_params[0]+range_params[0], num=n_grid)
    pas = np.linspace(init_params[1]-range_params[1], init_params[1]+range_params[1], num=n_grid)
    cons = np.linspace(init_params[2]-range_params[2], init_params[2]+range_params[2], num=n_grid)

    grid_likes = np.zeros((n_grid,n_grid,n_grid))
    
    t0 = time.time()

    for sep_ix, sep in enumerate(seps):
        for pa_ix, pa in enumerate(pas):
            for con_ix, con in enumerate(cons):

                params = [sep,pa,con]
                loglike = graphic_cfl.neg_fake_planet_loglikelihood(params,
                    cube=cube, parangs_rad=parangs_rad, psf=psf, plot=False,
                    uncert=image_uncert, pca_settings=pca_settings, fit_params=fit_params)

                grid_likes[sep_ix, pa_ix, con_ix] = np.exp(loglike)

    t1 = time.time()
    f_log.write('Time taken for grid: {0:.1f} mins for {1} gridpoints\n'.format((t1-t0)/60, n_grid*n_grid*n_grid))

    # Save the grid and points
    save_arr = {'seps':seps,'pas':pas,'cons':cons,'grid_likes':grid_likes}
    with open(grid_file,'wb') as myf:
        pickle.dump(save_arr, myf)

#########################
if plot_grid:

    # Load the pickle file
    with open(grid_file,'rb') as f:
        save_arr = pickle.load(f)
    grid_likes = save_arr['grid_likes']
    seps = save_arr['seps']
    pas = save_arr['pas']
    cons = save_arr['cons']

    # Normalize the likelihoods
    grid_likes /= np.sum(grid_likes)

    # Plot it
    plt.figure(1)
    plt.clf()

    n_params = 3
    param_names = ['Sep (mas)', 'PA (deg)', 'Contrast Ratio (*1e-4)']
    params = [seps*platescale,pas,cons*1e4]
    init_params[2] = init_params[2]*1e4
    range_params[2] = range_params[2]*1e4
    marg_ixs = [(1,2),(0,2),(0,1)]

    for i in range(n_params):
        # First plot on each row is the marginal likelihood
        plt.subplot(n_params, n_params, n_params*i + i + 1)
        marg_like = np.sum(grid_likes, axis=marg_ixs[i])
        plt.plot(params[i],marg_like)
        plt.xlabel(param_names[i])
        plt.ylabel('Likelihood')

        # Then iterate through the other parameters and plot the conditional likelihood
        for j in range(i):
            plt.subplot(n_params, n_params, n_params*j + i + 1)
            cond_ixs = [0,1,2]
            cond_ixs.remove(i)
            cond_ixs.remove(j)
            cond_like = np.sum(grid_likes,axis=tuple(cond_ixs))

            plot_lims = [params[i][0],params[i][-1],params[j][0],params[j][-1]]
            plt.imshow(cond_like,extent=plot_lims,aspect='auto',origin='lower')

            plt.ylabel(param_names[j])
            plt.xlabel(param_names[i])

    plt.tight_layout()
    plt.savefig(dataset+prefix+'parameter_plot.pdf')


if calc_values:

    # Only need to load pickle file if it's not already loaded
    if not plot_grid:
        # Load the pickle file
        with open(grid_file,'rb') as f:
            save_arr = pickle.load(f)
        grid_likes = save_arr['grid_likes']
        seps = save_arr['seps']
        pas = save_arr['pas']
        cons = save_arr['cons']

        # Normalize the likelihoods
        grid_likes /= np.sum(grid_likes)

        init_params[2] = init_params[2]*1e4
        range_params[2] = range_params[2]*1e4
        marg_ixs = [(1,2),(0,2),(0,1)]

    f = open(values_file,'w')

    init_params[0] = init_params[0]*platescale
    range_params[0] = range_params[0]*platescale

    if plot_grid:
        plt.figure(1)
        plt.clf()

    for i in range(3):

        marg_like = np.sum(grid_likes,axis=marg_ixs[i])

        # Gaussian model
        gauss_model = models.Gaussian1D(amplitude=0.8,mean=init_params[i],stddev=range_params[i]/2.)

        # LM fit
        fitter = LevMarLSQFitter()
        fit = fitter(gauss_model,np.linspace(init_params[i]-range_params[i],init_params[i]+range_params[i], num=n_grid),
                     marg_like,acc=1e-4,maxiter=1000)

        idecimal = np.floor(np.log10(float('%.1g'%(fit.stddev.value))))
        ir = int(-1*idecimal + 1)

        # Store raw parameter values in output file
        param_name = ['Sep','PA','CR'][i]
        f.write('{0}: {1:.{ir}f}+/-{2:.{ir}f}\n'.format(param_name, fit.mean.value, fit.stddev.value, ir=ir)) 

        # Also save output values, so as to calculate on-sky values later
        if i == 0:
            store_sep = fit.mean.value/platescale
            store_sep_err = fit.stddev.value/platescale
        elif i == 1:
            store_pa = fit.mean.value
            store_pa_err = fit.stddev.value
        elif i == 2:
            store_cr = fit.mean.value
            store_cr_err = fit.stddev.value

        # Make the second, model plot, if making plot
        # This plot is the same as that in plot_grid above, but with the model overlaid.
        if plot_grid:

            # First plot the Gaussian model
            plt.subplot(n_params, n_params, n_params*i + i + 1)
            x_model = np.linspace(init_params[i]-range_params[i],init_params[i]+range_params[i], num=100)
            y_model = fit(x_model)
            plt.plot(x_model, y_model, 'r')

            # Superimpose the measured marginal likelihoods
            marg_like = np.sum(grid_likes, axis=marg_ixs[i])
            plt.plot(params[i],marg_like,'ko',markersize=2)
            plt.xlabel(param_names[i])
            plt.ylabel('Likelihood')

            # Then iterate through the other parameters and plot the conditional likelihood
            for j in range(i):
                plt.subplot(n_params, n_params, n_params*j + i + 1)
                cond_ixs = [0,1,2]
                cond_ixs.remove(i)
                cond_ixs.remove(j)
                cond_like = np.sum(grid_likes,axis=tuple(cond_ixs))

                plot_lims = [params[i][0],params[i][-1],params[j][0],params[j][-1]]
                plt.imshow(cond_like,extent=plot_lims,aspect='auto',origin='lower')

                plt.ylabel(param_names[j])
                plt.xlabel(param_names[i])


    # Finally, extract on-sky position and true errors for companion

    stellar_error = stellar_error/platescale


    sep_err = np.sqrt(stellar_error**2 + store_sep_err**2)
    stellar_error_pa = (stellar_error/store_sep) * (180/np.pi)
    pa_err = np.sqrt(stellar_error_pa**2 + store_pa_err**2)

    PAs, sep_mas, _, _ = sphereastrometry.pa_IRDIS_seppa([store_pa,pa_err],[store_sep,sep_err],pscale)
    f.write('\n')
    f.write('On-sky PA [deg]: {:.2f}+/-{:.2f}\n'.format(PAs[0], PAs[1]))
    f.write('On-sky sep [mas]: {:.1f}+/-{:.1f}\n'.format(sep_mas[0], sep_mas[1]))

    # Also convert flux to contrast
    v = -2.5*np.log10(store_cr*1e-4)
    vm = -2.5*np.log10((store_cr+store_cr_err)*1e-4)
    vp = -2.5*np.log10((store_cr-store_cr_err)*1e-4)

    idecimal = np.floor(np.log10(float('%.1g'%(v-vm))))
    ir = int(-1*idecimal + 1)

    f.write('Contrast: {0:.{ir}f} +{1:.{ir}f} / -{2:.{ir}f}\n'.format(v, vp-v, v-vm, ir=ir))
    f.write('Contrast errors may be overly optimistic!\n')
    f.write('Does not include uncertainty in weather or host star flux\n')

    # Tidy and save plot
    if plot_grid:
        plt.tight_layout()
        plt.savefig(dataset+prefix+'parameter_plot_model.pdf')

    # Clean up
    f.write('\n\n')
    f.write('---------------------------------------------------\n')
    f.close()

    f_log.write('values of parameters written to {}'.format(values_file))

f_log.write('\n\n---------------------------------------------------\n')
f_log.close()
