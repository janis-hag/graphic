# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:59:01 2016

@author: cheetham
"""

# Companion fitting procedures

import numpy as np
import time, emcee
import matplotlib.pyplot as plt
import astropy.io.fits as pf
from scipy import optimize
import bottleneck

import graphic_nompi_lib_330
import graphic_contrast_lib
import pca
import pickle

########################


def neg_fake_planet_loglikelihood(params, cube=None, parangs_rad=None, psf=None,
                                  plot=False, uncert=1, pca_settings=None,
                                  fit_params=None):
    ''' Likelihood function for fitting to a companion by injection of a fake
    companion in the raw frames and minimizing the residuals.
    A lot of things are assumed here. It uses smart_annular_pca on a single annulus
    This should only be used for extremely small datasets, since otherwise a likelihood
    evaluation will take minutes and emcee would take days to run.

    pca_settings is a dictionary that stores n_modes, n_annuli, r_min,r_max, fwhm,
    n_fwhm (protection angle), arc_length, min_reference_frames, threads and median_combine

    fit_params is the [sep,pa,radius] of the area used to calculate the goodness-of-fit

    uncert is the uncertainty on the flux values in the image. Could come from
    the standard deviation of the flux values.
    '''

    # Unpack the parameters and turn them into arrays
    radii, pa, fluxes = params
    radii = [radii]
    fluxes = [-fluxes]

    # Enforce some parameter limits
    # Separation within a small region of initial guess
    #    if np.abs(radii[0]-companion_guess[0]) > fitting_sz:
    #        return -np.Inf
    # PA between 0 and 360 deg
    if (params[1] > 360) or (params[1] < 0):
        return -np.Inf
    # Flux > 0
#    if (params[2] < 0):
#        return -np.Inf

# Convert position angle to azimuth offset and get the parangs in degrees as well
    parangs_deg = parangs_rad * 180. / np.pi
    azimuth_offset = -pa * np.pi / 180.

    # Add in a negative companion at the right location
    cube = graphic_contrast_lib.inject_companions(cube, psf, parangs_rad, radii,
                                                  fluxes,
                                                  azimuth_offset=azimuth_offset,
                                                  psf_pad=10, silent=True,
                                                  save_name=None)

    # Unpack the PCA settings
    n_modes = pca_settings['n_modes']
    n_annuli = pca_settings['n_annuli']
    r_min = pca_settings['r_min']
    r_max = pca_settings['r_max']
    fwhm = pca_settings['fwhm']
    n_fwhm = pca_settings['n_fwhm']
    arc_length = pca_settings['arc_length']
    min_reference_frames = pca_settings['min_reference_frames']
    threads = pca_settings['threads']
    median_combine = pca_settings['median_combine']
    if 'pca_type' in pca_settings.keys():
        pca_type = pca_settings['pca_type']
    else:
        pca_type = 'smart_annular_pca'

    # Now run PCA with the right settings
    if pca_type == 'smart_annular_pca':
        pca_cube = pca.smart_annular_pca(
                cube, n_modes, None, parangs_deg, n_fwhm=n_fwhm, fwhm=fwhm,
                n_annuli=n_annuli, r_min=r_min, r_max=r_max,
                arc_length=arc_length,
                min_reference_frames=min_reference_frames, threads=threads,
                silent=True)
    elif pca_type == 'noadi':
        pca_cube = cube

    # Derotate
    if threads > 1:
        derot_cube = pca.derotate_and_combine_multi(pca_cube, parangs_deg,
                                                    save_name=None,
                                                    median_combine=False,
                                                    threads=threads,
                                                    silent=True)
    else:
        derot_cube = pca.derotate_and_combine(pca_cube, parangs_deg,
                                              save_name=None,
                                              median_combine=False, silent=True)

    # Combine
    if median_combine:
        final_im = bottleneck.nanmedian(derot_cube, axis=0)
    else:
        final_im = bottleneck.nanmean(derot_cube, axis=0)

    # Background subtract it
    final_im -= np.nanmean(final_im)

    # Smooth the image
    #    nan_mask = np.isnan(final_im)
    #    final_im[nan_mask] = 0.
    #    final_im = signal.fftconvolve(final_im,ker,mode='same')
    #    final_im[nan_mask] = np.nan

    # And measure the goodness of fit from the area around the initial guess
    xpos = np.int(fit_params[0] * np.sin(-fit_params[1] * np.pi / 180.) +
                  final_im.shape[0] / 2)
    ypos = np.int(fit_params[0] * np.cos(-fit_params[1] * np.pi / 180.) +
                  final_im.shape[0] / 2)

    xmin = np.max([xpos - fit_params[2], 0])
    xmax = np.min([xpos + fit_params[2], final_im.shape[0]])
    ymin = np.max([ypos - fit_params[2], 0])
    ymax = np.min([ypos + fit_params[2], final_im.shape[0]])

    area = final_im[ymin:ymax, xmin:xmax]
    #    chi2 = np.nanstd(area) # Standard deviation
    chi2 = np.nansum((area / uncert)**2)
    #    chi2 = np.nansum(np.abs(area))

    # And check that it works
    #    print ypos,xpos
    #    final_im[ymin:ymax,xmin:xmax]+=1
    #    pf.writeto('/Users/cheetham/test.fits',final_im,clobber=True)

    if plot:
        plt.clf()
        plt.imshow(final_im, vmin=-1, vmax=1, origin='lowerleft')
        #        plt.imshow(area,vmin=-20,vmax=20,origin='lowerleft')
        plt.colorbar()

    return -chi2 / 2.


########################


def simple_companion_resids(params, image, psf_image):
    ''' Returns the residuals for a companion fit
    params = [x_centre, y_centre, flux] (where flux is the scaling factor between
    image and psf_image)
    Here x and y are the first and second dimensions of the array
    '''

    # First check if the centre position is inside the range of the image. If not, return inf
    #    if (params[0] < 0) or (params[0] > image.shape[0]):
    #        return np.inf
    #    elif (params[1] < 0) or (params[1] > image.shape[1]):
    #        return np.inf

    # Do a FFT to shift the psf image by a fraction of a pixel
    integer_cen = np.round(params[0:2]).astype(np.int)
    subinteger_cen = params[0:2] - integer_cen
    shifted_psf = params[2] * graphic_nompi_lib_330.fft_shift_pad(
            psf_image, subinteger_cen[0], subinteger_cen[1], pad=2)

    # and a roll to shift the rest
    shifted_psf = np.roll(shifted_psf, integer_cen[0], axis=0)
    shifted_psf = np.roll(shifted_psf, integer_cen[1], axis=1)

    # Now subtract the shifted psf frame from the image
    resids = image - shifted_psf

    #    plt.imshow(shifted_psf,cmap='ds9cool',origin='lowerleft')

    # And return the residuals
    return resids


########################

########################


def simple_companion_loglikelihood(params, image, psf_image, image_err=1.,
                                   plot=False):
    ''' A wrapper for the loglikelihood function for companion_resids
    params = [x_centre, y_centre, flux, nuisance parameter]
    The nuisance parameter is optional.
    '''

    # Enforce some parameter limits
    # Nuisance parameter can't be less than 0
    if len(params) > 3:
        if params[3] < 0:
            return -np.inf

    resids = simple_companion_resids(params, image, psf_image)

    if plot:
        plt.imshow(resids)

    # Turn into chi2 and loglikelihood
    if len(params) > 3:
        # Use the nuisance parameter
        #        print 'Using nuisance param',params[3]
        #        weight = 1./(params[3]**2 + image_err**2)
        weight = 1. / (params[3] * image_err**2)
        loglike = -0.5 * (np.nansum((resids**2 * weight) - np.log(weight)))
    else:
        chi2 = np.nansum(resids**2 / image_err**2)
        loglike = -chi2 / 2.

    return loglike


########################

########################


def simple_companion_stdev(params, image, psf_image, image_err=1.):
    ''' A wrapper for companion_resids that returns the standard deviation
    of the residuals
    params = [x_centre, y_centre, flux]
    '''

    resids = simple_companion_resids(params, image, psf_image)

    return np.std(resids)


########################

########################


def companion_mcmc(image, psf_image, initial_guess, image_err=1., n_walkers=50.,
                   n_iterations=1e3, threads=3, plot=False, burn_in=200,
                   pca_settings=None, parangs_rad=None,
                   likelihood_method='neg_fake_planet', fit_params=None,
                   save_name=None):
    ''' Run emcee, the MCMC Hammer on an image, subtracting a shifted and scaled
    version of the psf image from the derotated image.
    initial_guess = [x_centre, y_centre, flux] to start the MCMC chains around
    n_walkers is the number of chains to run simultaneously
    n_iterations
    threads = number of cores to use for multithreading
    image_err = the "uncertainty" of the image, used to turn the residuals into a chi2

    '''

    # Choose some initial parameters for the walkers
    n_params = len(initial_guess)  # number of parameters
    scatter = np.array([
            0.5, 0.5, 0.5 * initial_guess[2], 0.1 * initial_guess[-1]
    ])  # 0.5 pix scatter for the positions, 50% for the flux
    p0 = [
            initial_guess +
            scatter[0:n_params] * np.random.normal(size=n_params)
            for i in range(n_walkers)
    ]

    if likelihood_method == 'neg_fake_planet':
        like_func = neg_fake_planet_loglikelihood
        paramnames = ['Sep', 'PA', 'Flux']
        paramdims = ['(pix)', '(deg)', 'Ratio', '']
        args = [
                image, parangs_rad, psf_image, False, image_err, pca_settings,
                fit_params
        ]

    elif likelihood_method == 'simple_companion_fit':
        like_func = simple_companion_loglikelihood
        paramnames = ['Xpos', 'Ypos', 'Flux', 'Nuisance']
        paramdims = ['(pix)', '(pix)', 'Ratio', '']
        args = [image, psf_image, image_err]

    # Set up the sampler
    t0 = time.time()
    sampler = emcee.EnsembleSampler(n_walkers, n_params, like_func, args=args,
                                    threads=threads)

    # Run it
    sampler.run_mcmc(p0, n_iterations)
    tf = time.time()

    print('Time elapsed =',
          '{0:.2f}'.format(np.round((tf - t0) / 60., decimals=2)), 'mins')

    # Get the results
    chain = sampler.chain
    flatchain = sampler.flatchain

    # and clean up
    if threads > 1:
        sampler.pool.terminate()

    # Save the output if needed (before burn in)
    if save_name:
        mcmc_data = {
                'chain': chain,
                'param_names': paramnames,
                'param_dims': paramdims,
                'likelihood_method': likelihood_method
        }

        with open(save_name, 'w') as myf:
            pickle.dump(mcmc_data, myf)

    # Remove the burn in
    burn_in = np.int(burn_in)
    chain = chain[:, burn_in:, :]
    flatchain = flatchain[burn_in:]

    if plot == True:

        plt.figure(2)
        plt.clf()
        color = 'k'

        # Make a corner plot
        for i in range(n_params):

            # First plot on each row is the marginal likelihood
            plt.subplot(n_params, n_params, n_params * i + i + 1)
            plt.hist(flatchain[:, i], np.int((n_iterations)**(0.5)),
                     color=color, histtype="step")

            plt.ylabel('Counts')
            plt.xlabel(paramnames[i] + ' ' + paramdims[i])

            # Then iterate through the other parameters and plot the conditional likelihood
            for j in range(i):

                plt.subplot(n_params, n_params, n_params * j + i + 1)
                plt.hist2d(flatchain[:, i], flatchain[:, j],
                           np.int(0.5 * (n_iterations)**(0.5)))

                plt.ylabel(paramnames[j] + ' ' + paramdims[j])
                plt.xlabel(paramnames[i] + ' ' + paramdims[i])

    return chain


########################


def simple_companion_leastsq(image, psf_image, initial_guess, image_err=1.,
                             threads=3, plot=False, method='Powell',
                             fit_to_stdev=False):
    ''' Run least squares fitting on the companion position and flux
    initial_guess = array of [x_centre, y_centre, flux_multiplier] of companion
    '''

    # Define the function to use
    if fit_to_stdev:

        def min_funct(params, image, psf_image, image_err):
            stdev = simple_companion_stdev(params, image, psf_image,
                                           image_err=image_err)
            return stdev
    else:

        def min_funct(params, image, psf_image, image_err):
            loglike = simple_companion_loglikelihood(params, image, psf_image,
                                                     image_err=image_err)
            return (-loglike)

    result = optimize.minimize(min_funct, initial_guess, args=(image, psf_image,
                                                               image_err),
                               tol=1e-4, method=method)

    return result


########################


def simple_companion_flux_leastsq(image, psf_image, initial_guess, image_err=1.,
                                  threads=3, plot=False, method='Powell'):
    ''' Run least squares fitting on the companion flux only.
    initial_guess = array of [x_centre, y_centre, flux_multiplier] of companion
    This will keep x_centre and y_centre fixed.
    Note that flux is explicitly forced to be >=0
    '''

    # Define the function to use
    def flux_min_funct(x, xpos, ypos, image, psf_image, image_err):
        params = np.array([xpos, ypos, x[0]])
        loglike = simple_companion_loglikelihood(params, image, psf_image,
                                                 image_err=image_err)
        return (-loglike)

    guess = initial_guess[2]
    result = optimize.minimize(
            flux_min_funct, guess, tol=1e-5, method=method,
            args=(initial_guess[0], initial_guess[1], image, psf_image,
                  image_err))
    #    result = optimize.least_squares(flux_min_funct, guess, ftol=1e-04, xtol=1e-04,
    #                        gtol=1e-04,args=(initial_guess[0],initial_guess[1],image,psf_image,image_err))
    return result


########################


def neg_fake_planet_companion_leastsq(cube, parangs_rad, psf, initial_guess,
                                      image_err=1., threads=3, method='Powell',
                                      pca_settings=None, fit_params=None):
    ''' Run least squares fitting on the companion position and flux
    initial_guess = array of [x_centre, y_centre, flux_multiplier] of companion
    '''

    # Define the function to use
    def min_funct(params, cube, parangs_rad, psf, image_err, pca_settings,
                  fit_params):
        loglike = neg_fake_planet_loglikelihood(params, cube=cube,
                                                parangs_rad=parangs_rad,
                                                psf=psf, plot=False,
                                                uncert=image_err,
                                                pca_settings=pca_settings,
                                                fit_params=fit_params)
        return (-loglike)

    result = optimize.minimize(
            min_funct, initial_guess, args=(cube, parangs_rad, psf, image_err,
                                            pca_settings, fit_params), tol=1e-3,
            method=method)

    return result
