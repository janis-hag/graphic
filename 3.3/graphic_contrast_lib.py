# -*- coding: utf-8 -*-
"""

Some modules for calculating contrast curves

"""

import numpy as np
import astropy.io.fits as pyfits
import bottleneck
import scipy
import matplotlib.pyplot as plt
import os

from graphic_nompi_lib_330 import fft_shift
from scipy import signal, stats, interpolate

###############

###############

def measure_flux(image, radius=3, centre='Default'):
    ''' Measures the maximum flux of a point source in an image.
    Position is measured from the centre of the image, in pixels'''

    if centre =='Default':
        centre=[image.shape[0]//2,image.shape[1]//2]

    # Pixel distance map
    xarr=np.arange(0,image.shape[0])-centre[0]
    yarr=np.arange(0,image.shape[1])-centre[1]
    xx,yy=np.meshgrid(xarr,yarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)

#    plt.imshow(pix_dist_map)

    # Just take the max for now
    flux=np.nanmax(image[pix_dist_map<=radius])

    return flux

###############

###############

def aperture_photometry(image,radius=3,centre='Default',mean=False):
    ''' Performs aperture photometry on a point source in an image.
    It takes the total flux from a circle of a given radius and centre
    Position is measured from the centre of the image, in pixels'''

    if centre =='Default':
        centre=[image.shape[0]//2,image.shape[1]//2]

    # Pixel distance map
    xarr=np.arange(0,image.shape[0])-centre[0]
    yarr=np.arange(0,image.shape[1])-centre[1]
    xx,yy=np.meshgrid(xarr,yarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)

    # How much of each pixel is within the aperture?
    fraction = (radius+0.5) - pix_dist_map
    fraction[fraction<0] = 0
    fraction[fraction>1] = 1

    # Diagnostic plots
#    plt.clf()
#    plt.imshow(pix_dist_map)
#    plt.imshow(fraction)
#    plt.colorbar()

    # How close is the area enclosed by "fraction" to the area of the circle?
#    print np.sum(fraction),np.pi*radius**2

    # Take the total
    if mean:
        flux = np.nansum(image*fraction/np.nansum(fraction))
    else:
        flux = np.nansum(image*fraction)

    return flux

###############

###############

def median_flux(image,radius=3,centre='Default'):
    ''' Performs aperture photometry on a point source in an image.
    It takes the total flux from a circle of radius "radius" centred on "centre"
    Position is measured from the centre of the image, in pixels'''

    if centre =='Default':
        centre=[image.shape[0]//2,image.shape[1]//2]

    # Pixel distance map
    xarr=np.arange(0,image.shape[0])-centre[0]
    yarr=np.arange(0,image.shape[1])-centre[1]
    xx,yy=np.meshgrid(xarr,yarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)

#    plt.imshow(pix_dist_map)

    # Take the total
    flux=np.nanmedian(image[pix_dist_map<=radius])

    return flux

###############

###############

def noise_vs_radius(image,n_radii,r_min,fwhm,r_max='Default',mad=True,
                    robust_sigma=False):
    ''' Calculate the 1-sigma noise in an image as a function of radius.
    mad = median absolute deviation instead of standard deviation
    robust_sigma = use a robust estimator of the standard deviation
    '''


    # Pixel distance map that we will use later
    npix=np.min(image.shape)
    xarr=np.arange(0,npix)-npix/2
    xx,yy=np.meshgrid(xarr,xarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)

    if r_max == 'Default':
        r_max = int(npix/2.-fwhm/2.)
    r_min = int(r_min)

    # Get the arrays ready
    detec_r=np.linspace(r_min,r_max,n_radii) # the mean radii of each annulus
    noise=np.zeros(n_radii)

    # Loop through annuli
    for r in np.arange(n_radii):

        # The inner and outer edge of this annulus (in pix)
        r_in_pix=detec_r[r]-fwhm/2.
        r_out_pix=detec_r[r]+fwhm/2.

        # What pixels does that correspond to?
        pix=(pix_dist_map>r_in_pix) & (pix_dist_map < r_out_pix)
        vals=image[pix]

        # What is the scatter of these pixels?
        if mad:
            std = 1.4826 * np.nanmedian(np.abs(vals-np.nanmedian(vals)))
        elif robust_sigma:
            std = robust_std(vals)
        else:
            std = np.nanstd(vals)
        noise[r] = std

    return noise,detec_r

###############

###############

def robust_std(x):
    ''' '''
    y = x.flatten()
    n = len(y)
    y.sort()
    ind_qt1 = int(round((n+1)/4.))
    ind_qt3 = int(round((n+1)*3/4.))
    IQR = y[ind_qt3]- y[ind_qt1]
    lowFense = y[ind_qt1] - 1.5*IQR
    highFense = y[ind_qt3] + 1.5*IQR
    ok = (y>lowFense)*(y<highFense)
    yy=y[ok]
    return yy.std(dtype='double')

###############

###############

def noise_vs_radius_apertures(image,n_radii,r_min,fwhm,r_max='Default',
                              reps = 10,plot=False,mad=True,robust_sigma=False):
    ''' Calculate the 1-sigma noise in an image as a function of radius.
    This is done by calculating the average flux within a number of small FWHM wide
    circles at the same radius. The noise is estimated from the standard deviations
    of these mean fluxes.
    robust_sigma: Use a robust estimator of the standard deviation that ignores outliers.

    '''

    # Pixel distance map that we will use later
    npix=np.min(image.shape)
    xarr=np.arange(0,npix)-npix/2
    xx,yy=np.meshgrid(xarr,xarr)
    pix_dist_map = np.sqrt(xx**2 + yy**2)

    if r_max == 'Default':
        r_max = npix/2.-fwhm/2.

    # Get the arrays ready
    detec_r=np.linspace(r_min,r_max,n_radii) # the mean radii of each annulus
    noise=np.zeros(n_radii)

    circle_diam = fwhm # Might want to change this later

    plotim = 0*image

    # Let's fix the NaNs here to save time later
    nan_mask = np.isnan(image)
    image = np.nan_to_num(image)

    # Loop through annuli
    for r in np.arange(n_radii):

        # To speed up the next for-loop,
        # only consider the pixels within this annulus
        annulus_pix = (pix_dist_map < (detec_r[r]+circle_diam+1)) & \
            (pix_dist_map > (detec_r[r]-circle_diam-1))
        xx_annulus = xx[annulus_pix]
        yy_annulus = yy[annulus_pix]
        image_annulus = image[annulus_pix]

        # How many circles can we fit within this annulus?
        n_circles = np.int(np.floor(2*np.pi*detec_r[r] / (circle_diam)))

        # Repeat the calculation several times with different apertures
        # to smooth out the contrast curve
        noise_reps = np.zeros(reps)
        all_noise_reps = np.zeros((reps,n_circles))
        for rep in range(reps):

            # Loop over circles to calculate the stddev of each
            # offset the first one by a random angle
            azimuth_offset = np.random.uniform(low=0,high=2*np.pi)
            noise_circles = np.zeros((n_circles))
            for circle_ix in range(n_circles):

                circle_x = detec_r[r] * np.sin(circle_ix* 2*np.pi/n_circles + azimuth_offset)
                circle_y = detec_r[r] * np.cos(circle_ix* 2*np.pi/n_circles + azimuth_offset)

                # What is the distance from the circle centre?
                circ_dist_map = np.sqrt((xx_annulus-circle_x)**2 + (yy_annulus-circle_y)**2)

                ## What is the fraction of each pixel that is within the circle?
                ## this formula is wrong, but it is close to correct
                fraction = (circle_diam/2.+0.5) - circ_dist_map
                fraction[fraction<0] = 0
                fraction[fraction>1] = 1

                ## These lines neglect fractions of a pixel, which is safer but noisier
#                good_pix = circ_dist_map <= (circle_diam/2.)
#                fraction = good_pix # If we want to only use 1 or 0

                # Make the weights for the original NaN values 0
                fraction[nan_mask[annulus_pix]] = 0.

                noise_circles[circle_ix] = np.sum(image_annulus*fraction/np.nansum(fraction))

                if rep == 0 & circle_ix ==0:
                    # Plot the location of the circles
#                    plotim[annulus_pix] += circle_ix*good_pix
#                    plotim[annulus_pix] += circle_ix*fraction
#                    plotim[annulus_pix] = circ_dist_map
                    plotim[annulus_pix] += fraction

            # The noise is estimated by the standard deviation
            if mad:
                noise_reps[rep] = 1.4862*np.nanmedian(np.abs(noise_circles-np.nanmedian(noise_circles)))
            elif robust_sigma:
                noise_reps[rep] = robust_std(noise_circles)
            else:
                noise_reps[rep] = np.nanstd(noise_circles)



            # Or we can just combine all of the measurements and take the std deviation at the end
            # This gives almost exactly the same result
#            all_noise_reps[rep] = noise_circles
#
        # Take the mean over the repetitions
        noise[r] = np.nanmean(noise_reps)

        # Or take the standard deviation of all the circles from every repetition
        # This gives almost exactly the same result
#         noise[r] = np.nanstd(all_noise_reps)
#        noise[r] = 1.4826*np.nanmedian(np.abs(all_noise_reps - np.nanmedian(all_noise_reps))) # use MAD

    if plot:
        plt.clf()
        plt.imshow(plotim)

    return noise,detec_r

###############

###############

def mean_vs_radius(image,n_radii,r_min,fwhm,median = False):
    ''' Calculate the mean value of an image as a function of radius.
    Uses the mean over a 1fwhm annulus.
    Can also do the median.
    Useful for subtracting a radial profile'''

    if median:
        combine_algo = np.nanmedian
    else:
        combine_algo = np.nanmean

    # Pixel distance map that we will use later
    npix=np.min(image.shape)
    xarr=np.arange(0,npix)-npix/2
    xx,yy=np.meshgrid(xarr,xarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)

    # Get the arrays ready
    detec_r=np.linspace(r_min,npix/2-fwhm/2.,n_radii) # the mean radii of each annulus
    mean=np.zeros(n_radii)

    # Loop through annuli
    for r in np.arange(n_radii):

        # The inner and outer edge of this annulus (in pix)
        r_in_pix=detec_r[r]-fwhm/2.
        r_out_pix=detec_r[r]+fwhm/2.

        # What pixels does that correspond to?
        pix=(pix_dist_map>r_in_pix) & (pix_dist_map < r_out_pix)
        vals=image[pix]

        # What is the scatter of these pixels?
        std=combine_algo(vals)
        mean[r]=std

    return mean,detec_r

###############

###############

def prepare_detection_image(filename,save_name=None,smooth_image_length=None,
         median_filter_length=None,convolve_with_circular_aperture=None):
    '''
    Perform various cosmetic steps to an image, which make it easier to see
    faint structures. Default is to do nothing. For each option, set the argument
    to the requested size of the filter or convolution.

    This assumes filename contains a 2D image, but we could vectorize or add
    some for-loops if we need more dimensions.

    Options:
        save_name: Save the image (and header) with this name. Otherwise, it will
            be returned but not saved.
        smooth_image_length: Convolve with a Gaussian kernel with this FWHM. This
            is the Gaussian Cross Correlation / Gaussian Matched Filter approach
            as defined in Ruffio+ 2017.
        median_filter_length: For each pixel, subtract the median calculated in
            a box with this radius. Useful for removing large-scale structures
            that are irrelevant for planet detectability but visually affect
            the final image.
        convolve_with_circular_aperture: Convolve with a circle with this radius.
            ( Note, this was used by one of the Mawet/Absil papers, but does not
            really make sense to apply to real data. )
    '''
    # Load the image (if filename is a string)
    if isinstance(filename,str):
        image,hdr = pyfits.getdata(filename,header=True)
        image=np.array(image,dtype=np.float)# To avoid endian problems with bottleneck
    else:
        # Otherwise assume the input was the image itself
        image = filename
        hdr=pyfits.Header()

    # Pixel distance map that we will use later
    npix=image.shape[1]
    xarr=np.arange(0,npix)-npix/2
    xx,yy=np.meshgrid(xarr,xarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)

    # Gaussian smooth the image
    if smooth_image_length:
        # We need to temporarily remove the nans
        nans=np.isnan(image)
        image[nans]=0.

        ker_sz=np.int(1+np.round(smooth_image_length)*4)
        x,y=np.indices((ker_sz,ker_sz),dtype=np.float64)-ker_sz/2
        ker=np.exp(- (x**2+y**2)/(2*smooth_image_length**2))
        ker/=np.sum(ker)
        smoothed_image=signal.fftconvolve(image,ker,mode='same')

        image=smoothed_image
        image[nans]=np.NaN

        hdr['HIERARCH GC CONTRAST SMOOTH FWHM'] = smooth_image_length

    # Median filter to smooth the image
    if median_filter_length:
        # Old way: use median_filter (doesnt handle nans well)
#        smoothed_image=ndimage.filters.median_filter(image,size=median_filter_length)
        # New way: do the median filter manually
        smoothed_image=np.zeros(image.shape)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                xmn=np.max([0,x-median_filter_length//2])
                xmx=np.min([image.shape[0]-1,x+median_filter_length//2])
                ymn=np.max([0,y-median_filter_length//2])
                ymx=np.min([image.shape[1]-1,y+median_filter_length//2])
                smoothed_image[x,y]=bottleneck.nanmedian(image[xmn:xmx,ymn:ymx])

        final_image=image-smoothed_image
        image=final_image

        hdr['HIERARCH GC CONTRAST MEDIAN FILT'] = median_filter_length

    if convolve_with_circular_aperture:
        # Convolve the image with a circular aperture of rad=FWHM
        circ_ap=np.zeros((npix,npix))
        circ_ap[pix_dist_map<(convolve_with_circular_aperture//2)]=1
        convol_sz=np.int(np.ceil(convolve_with_circular_aperture)+3)

        circ_ap=circ_ap[npix//2-convol_sz//2:npix//2+convol_sz//2,npix//2-convol_sz//2:npix//2+convol_sz//2]
#        plt.imshow(circ_ap)

        wherenan=np.isnan(image)
        image[wherenan]=0.
        image=signal.fftconvolve(image,circ_ap,mode='same')
        image[wherenan]=np.nan

        hdr['HIERARCH GC CONTRAST CONVOL CIRC'] = convolve_with_circular_aperture

    if save_name:
        pyfits.writeto(save_name,image,header=hdr,overwrite=True)

    return image

###############

###############

def contrast_curve(filename,flux_filename,n_radii=200,r_min=5.,r_max=None,fwhm=4.5,
                   remove_planet=False,planet_position=None,planet_radius=10.,
                   plate_scale=0.027,n_sigma=5,offset=0,label='',plot=False,
                   return_noise=False,sss_correction=True, self_subtraction_file=None,
                   save_contrast=None,save_noise=None,use_apertures=False,
                   mad=True,robust_sigma=False):
    '''
    Calculates a contrast curve from a given image and flux frame.
    This uses noise_vs_radius_apertures to calculate the noise in the image.
    n_radii: the number of radial positions to calculate the contrast at.
    fwhm: the expected full-width-half-max of the psf
    offset: adds a constant to the contrast curve. Useful for turning into a
          sensitivity curve (by adding the star magnitude)
    plate_scale : in arcsec
    sss_correction : apply small sample statistics correction? (Default=True)
    self_subtraction_file : a text file containing the fraction of flux that
        was self-subtracted during the PCA/ADI step, as a function of radius

    use_apertures: Measure the noise using aperture photometry instead of peak
        counts
    mad: use the median absolute deviation instead of the std. dev. to measure
        the noise
    robust_sigma: Use a robust estimator of the standard deviation instead.
    '''

    # Load the image
    image, hdr = pyfits.getdata(filename, header=True)
    image = np.array(image, dtype=np.float)
    # To avoid endian problems with bottleneck
    npix = image.shape[0]

    # Load the flux image
    if type(flux_filename) == type('a string'):
        flux_image,flux_header=pyfits.getdata(flux_filename,header=True)
    else:
        flux_image=1*flux_filename

    npix=image.shape[1]
    if r_max == None or r_max == 'Default':
        r_max = npix/2.-fwhm/2.

    # remove the planet (if needed)
    if remove_planet:
        planet_position = np.array(planet_position,dtype=np.int)
        planet_radius = np.int(planet_radius)
        image[planet_position[0]-planet_radius:planet_position[0]+planet_radius,
              planet_position[1]-planet_radius:planet_position[1]+planet_radius]=np.nan

    print('  Calculating contrast')
    # Measure the noise as a function of radius
    if use_apertures:
        noise,detec_r = noise_vs_radius_apertures(image,n_radii,r_min,fwhm,r_max,reps=20,
                                                 mad=mad,robust_sigma=robust_sigma)
    else:
        noise,detec_r=noise_vs_radius(image,n_radii,r_min,fwhm,r_max=r_max,mad=mad,
                                      robust_sigma=robust_sigma)
    detec_r_arcsec=detec_r*plate_scale

    # Small sample statistics correction
    initial_n_sigma=n_sigma
    if sss_correction:
        n_res_elements=np.floor(2*np.pi*detec_r/fwhm) #number of resolution elements
        sss_corr_factor=np.sqrt(1+ 1./(n_res_elements-1))
        cl=stats.norm.cdf(n_sigma) # confidence limit for the sigma value that was given
        # Replace n_sigma by this distribution to maintain a given confidence limit
        n_sigma=stats.t.ppf(cl,n_res_elements)/sss_corr_factor

    # Measure the primary flux:
    if use_apertures:
        primary_flux = aperture_photometry(flux_image,radius=fwhm,mean=True)
    else:
        primary_flux=measure_flux(flux_image,radius=5.)

    # Correct for self-subtraction
    if self_subtraction_file:
        # Load the file
        correction = np.loadtxt(self_subtraction_file)
        # Check it wasnt saved as columns instead of rows
        if (correction.shape[0] != 2) and (correction.shape[1] == 2):
            correction = np.transpose(correction)
        correction_seps,correction_factors = correction

        # Make an interpolation function
        correction_factor_interp = interpolate.interp1d(correction_seps,correction_factors,
            kind = 'linear', bounds_error=False, fill_value = 'extrapolate')

        # Interpolate at the locations of the contrast measurements
        throughput = correction_factor_interp(detec_r)

        # Now correct the calculated contrast
        noise /= throughput


    # Turn into contrast limits
    detec_contrast=-2.5*np.log10(n_sigma*noise/primary_flux)
    detec_limits=detec_contrast


    # Now plot it
    if plot:
    #    plt.clf()
        plt.plot(detec_r_arcsec,detec_limits+offset,label=label)
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel(str(initial_n_sigma)+r'$\sigma$ contrast')
        plt.xlim(xmin=0.)

        plt.tight_layout()
        plt.grid()

    if save_contrast:
        con_head ='Separation (arcsec) Contrast (mag)  using plate scale of '+str(plate_scale)
        np.savetxt(save_contrast, [detec_r_arcsec, detec_limits],
                   header=con_head)
        df = pd.DataFrame.from_records(
                np.array([detec_r_arcsec, detec_limits]).T,
                columns=['Separation (arcsec)', 'Contrast (mag)'])
        df.to_csv('contrast.csv', sep='\t')

    if save_noise:
        np.savetxt(save_noise,[detec_r,noise],header='Separation (pix) NoiseSigma (counts)')


    if return_noise:
        return noise, detec_r
    else:
        return [detec_r_arcsec,detec_limits]

##################

##################

def snr_map_quick(image_file,fwhm_pix,n_radii,r_min=3.,plot=False,
                  remove_planet=False,planet_position=None,
                   planet_radius=10.,save_name='snr_map.fits',
                   robust_sigma=False,mad=False):
    ''' Makes an SNR map from a PCA reduced image, by dividing into annuli and
    comparing the total flux in a region to the standard deviation of the values
    in the same annulus.
    Does not smooth or do anything fancy.
    Set remove_planet  = True, planet position = [x,y] to NaN out a region of
      size planet_radius pixels when calculating the noise, so that detected
      objects don't contaminate the noise estimate.
    '''

   # Load the datacube
    print("Loading cube")
    if isinstance(image_file,str):
        image,header=pyfits.getdata(image_file,header=True)
    else:
        image=image_file
        header={}

    # Pixel distance map that we will use later
    npix=image.shape[1]
    xarr=(np.arange(0,npix)-npix/2).astype(np.float)
    xx,yy=np.meshgrid(xarr,xarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)

    noise_image = 1*image
    # remove the planet (if needed)
    if remove_planet:
        xmin = int(planet_position[0]-planet_radius)
        xmax = int(planet_position[0]+planet_radius)
        ymin = int(planet_position[1]-planet_radius)
        ymax = int(planet_position[1]+planet_radius)
        noise_image[xmin:xmax,ymin:ymax]=np.nan


    # Get the noise using the contrast function
    noise_vs_rad,detec_r=noise_vs_radius(noise_image,n_radii,r_min,fwhm_pix,mad=mad,
                                         robust_sigma=robust_sigma)

    # Make an interpolation function
    noise_interp=interpolate.interp1d(detec_r,noise_vs_rad,kind='cubic',
                              bounds_error=False,fill_value=np.nan)

    # Mask out the points outside the range, then interpolate
    mask=(pix_dist_map < np.min(detec_r)) + (pix_dist_map > np.max(detec_r))
    noise_map=1*pix_dist_map
    noise_map[mask]=np.nan
    noise_map=noise_interp(noise_map)

    # Divide the image by the noise map
    snr_map=image/noise_map
    snr_map[mask] = np.nan

    if plot:
        plt.figure(1)
        plt.clf()
        im1=plt.imshow(snr_map)
        plt.colorbar(im1)
        plt.figure(2)
        plt.clf()
        plt.imshow(noise_image)
    if save_name:
        pyfits.writeto(save_name,snr_map,header=header,overwrite=True,output_verify='silentfix')
    else:
        return snr_map

##################

##################

def snr_map(image_file,noise_file,plot=False,
                  remove_planet=False,planet_position=None,
                   planet_radius=10.,save_name='snr_map.fits'):
    ''' Makes an SNR map from a PCA reduced image, by dividing by the noise,
    which is a required input. If you haven't calculated the noise, use snr_map_quick.
    Does not smooth or do anything fancy.
    Set remove_planet  = True, planet position = [x,y] to NaN out a region of
      size planet_radius pixels when calculating the noise, so that detected
      objects don't contaminate the noise estimate.
      noise_r = radii (pix) for the noise calcul
    '''

   # Load the datacube
    print("Loading cube")
    image,header=pyfits.getdata(image_file,header=True)

    # Load the file containing the pre-calculated noise vs radius
    noise_r,noise_vs_rad = np.loadtxt(noise_file)

    # Pixel distance map that we will use later
    npix=image.shape[1]
    xarr=(np.arange(0,npix)-npix/2).astype(np.float)
    xx,yy=np.meshgrid(xarr,xarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)

    # Make an interpolation function
    noise_interp=interpolate.interp1d(noise_r,noise_vs_rad,kind='cubic',
                              bounds_error=False,fill_value=np.nan)

    # Interpolate, then mask out the points outside the range
    mask=(pix_dist_map < np.min(noise_r)) + (pix_dist_map > np.max(noise_r))
    noise_map=1*pix_dist_map

    noise_map=noise_interp(noise_map)

    # Divide the image by the noise map
    snr_map=image/noise_map
    snr_map[mask] = np.nan

    if plot:
        plt.clf()
        im1=plt.imshow(snr_map)
        plt.colorbar(im1)
    if save_name:
        pyfits.writeto(save_name,snr_map,header=header,overwrite=True,output_verify='silentfix')


##################

##################

def inject_companions(cube_in, psf,parangs_rad, radii, fluxes,
                      azimuth_offset=0., psf_pad=10,
                      save_name='injected_companions.fits',
                      hdr=None, silent=False):
    '''
    Injects fake companions into an image cube by adding in copies of the psf.
    cube    = data cube (3D) or filename of data cube
    psf     = psf image (2D)
    parangs = list of parallactic angles (radians)
    radii   = list of separations (in pix) to put the companions
    fluxes  = list of fluxes for the companions
    azimuth_offset = the angle to rotate the injected companions relative to
    North (radians)
    psf_pad = the amount of pixels to pad the psf frame (on each side) before
    shifting
    hdr = the header of the cube (if cube is an array)

    It will inject all of the companions in a line'''

    if isinstance(cube_in, str):
        # cube,hdr=pyfits.getdata(cube,header=True)
        hdul = pyfits.open(cube_in)
        hdr = hdul[0].header
        cube = hdul[1].data
    else:
        cube = cube_in
        if hdr is None:
            hdr = pyfits.Header()

    if len(parangs_rad) != cube.shape[0]:
        raise Exception("Parallactic angle list must be the same size as the \
                        image cube!")

    # At the moment this function won't work if the psf goes over edge of array
    while np.max(radii) > (np.min(cube.shape[1:2])/2):
        print('Cube Radius: ' + str(np.min(cube.shape[1:2])/2) +
              '. Max radius:' + str(max(radii)))
        print('Radii of injected psfs is too large. Removing the last one.')
        radii = radii[:-1]
        fluxes = fluxes[:-1]
        # raise Exception("Radii of injected psfs is too large, and would go
        # outside the field of view")

    # Check that the number of fluxes is 1 or equal to the number of radii
    if (np.size(fluxes) != 1) and (np.size(fluxes) != np.size(radii)):
        raise Exception("Fluxes must be a single float or have length(fluxes) \
                        == length(radii)")

    if np.size(fluxes) == 1:
        fluxes = np.repeat(fluxes, np.size(radii))

    # Make sure everything is an array
    radii = np.array(radii)
    fluxes = np.array(fluxes)

    # Set up the psf in a padded array for when it is shifted
    pad_psf=np.zeros((psf.shape[0] + 2*psf_pad, psf.shape[1] + 2*psf_pad))
    pad_psf[pad_psf.shape[0]//2-psf.shape[0]//2:pad_psf.shape[0]//2+psf.shape[0]//2,
            pad_psf.shape[1]//2-psf.shape[1]//2:pad_psf.shape[1]//2+psf.shape[1]//2]=1*psf

    # And make the cube a bit bigger so that if the psf is near the edge
    # we dont have to worry about any index problems
    cube_extra_pad = psf_pad+psf.shape[1]
    cube2 = np.zeros((cube.shape[0], cube.shape[1] + 2*cube_extra_pad,
                      cube.shape[2] + 2*cube_extra_pad))
    cube2[:,cube_extra_pad:-cube_extra_pad,cube_extra_pad:-cube_extra_pad] = cube

    if not silent:
        print('Injecting psfs into the cleaned frames')
    # Loop over frames
    for frame_ix, frame in enumerate(cube2):

        # Loop over separations
        for sep_ix, sep in enumerate(radii):

#             Loop over position angles / fluxes:
#            for flux_ix,flux in enumerate(fluxes):
#            theta=flux_ix*2*np.pi/(len(fluxes))
            theta = azimuth_offset
            flux = fluxes[sep_ix]

            # Work out the relative companion positions in pixels
            pos_x = sep*np.cos(theta + parangs_rad[frame_ix])
            pos_y = sep*np.sin(theta + parangs_rad[frame_ix])
            pos_x_int = np.int(np.floor(pos_x)) # this will round down
            pos_y_int = np.int(np.floor(pos_y)) # this will round down

            # We can shift the psf by an integer number of pixels by hand, so
            # shift it by (pos_x mod 1, pos_y mod 1) first
            shifted_psf = fft_shift(pad_psf, pos_x % 1, pos_y % 1)

            # Now add in the flux-scaled psf to the frame
            x_ix_min = pos_x_int-pad_psf.shape[0]//2+frame.shape[0]//2
            y_ix_min = pos_y_int-pad_psf.shape[1]//2+frame.shape[1]//2
            if (x_ix_min < 0) or (y_ix_min < 0):
                print('Injected psf frame too close to edge!')
            elif ((x_ix_min+pad_psf.shape[0]) > frame.shape[0]) or ((y_ix_min+pad_psf.shape[1]) > frame.shape[1]):
                print('Injected psf frame too close to edge!')

            frame[x_ix_min:x_ix_min+pad_psf.shape[0],
                  y_ix_min:y_ix_min+pad_psf.shape[1]] += flux*shifted_psf

    # Now remove the extra pixels we just added
    cube = cube2[:, cube_extra_pad:-cube_extra_pad,
                 cube_extra_pad:-cube_extra_pad]
    if save_name:
        hdr['HIERARCH GC INJECT AZOFFSET'] = azimuth_offset
        if isinstance(cube_in, str):
            hdul.writeto(save_name)
        else:
            pyfits.writeto(save_name, cube, overwrite=True, header=hdr,
                           output_verify='silentfix')

    # The companions should be in cube already due to the way python handles
    # objects. So no need to return anything, but just in case, here it is
    return cube

##################

##################

def fit_injected_companions(adi_image,psf,radii,fluxes,azimuth_offset=0.,psf_pad=10.,
                         cutout_radius = 7,save_name = 'throughput.txt'):
    ''' Fits the throughput for injected companions in an image, assuming their
    positions are exactly where they were added.
    cube    = data cube (3D)
    psf     = psf image (2D)
    radii   = list of separations (in pix) for the injected companions
    fluxes  = list of fluxes for the injected companions
    azimuth_offset = the angle to rotate the injected companions relative to North (radians)
    psf_pad = the amount of pixels to pad the psf frame (on each side) before shifting
    cutout_radius = pix of cutout to fit the injected psf's flux

    This assumes a lot about inject_companions so make sure both are updated at the same time.'''

    if isinstance(adi_image,str):
        adi_image=pyfits.getdata(adi_image)

    # Set up the psf in a padded array for when it is shifted
    pad_psf=np.zeros((psf.shape[0]+2*psf_pad,psf.shape[1]+2*psf_pad))
    pad_psf[pad_psf.shape[0]//2-psf.shape[0]//2:pad_psf.shape[0]//2+psf.shape[0]//2,
            pad_psf.shape[1]//2-psf.shape[1]//2:pad_psf.shape[1]//2+psf.shape[1]//2]=1*psf

    print('Fitting to the flux for each psf')
    measured_throughputs=np.zeros((len(radii)))
    # Loop through the separations
    for sep_ix,sep in enumerate(radii):
        # Loop through the fluxes
#        for flux_ix,flux in enumerate(fluxes):
#            theta=flux_ix*2*np.pi/(len(fluxes))
#        theta = azimuth_offset
        flux = fluxes[sep_ix]

            # This is where we injected the companion
        pos_x=sep*np.cos(azimuth_offset)+(adi_image.shape[0])/2
        pos_y=sep*np.sin(azimuth_offset)+(adi_image.shape[1])/2
        pos_x_int=np.int(np.floor(pos_x)) # this will round down
        pos_y_int=np.int(np.floor(pos_y)) # this will round down

        # Shift the psf to the right position so we can fit to only the flux
        # We can shift the psf by an integer number of pixels by hand, so
        # shift it by (pos_x mod 1, pos_y mod 1) first
        shifted_psf=fft_shift(pad_psf,pos_x % 1,pos_y % 1)
        psf_shape=shifted_psf.shape

        # Cut out the region around the injected companion
        # And produce the expected psf array (scaled by the flux)
        region=adi_image[pos_x_int-cutout_radius:pos_x_int+cutout_radius,
                         pos_y_int-cutout_radius:pos_y_int+cutout_radius]
        psf_region=flux*shifted_psf[psf_shape[0]//2-cutout_radius:psf_shape[0]//2+cutout_radius,
                               psf_shape[1]//2-cutout_radius:psf_shape[1]//2+cutout_radius]


        # Define a function to fit to. We are directly fitting the throughput here,
        # since the flux was taken care of above
        def flux_func(flux,region,psf_region):
            return np.sum(np.abs(region-flux[0]*psf_region))
        # Fit to it using optimize.fmin
        measured_throughputs[sep_ix]=scipy.optimize.fmin(flux_func,[1],
                        xtol=1e-3,ftol=1e-3,args=(region,psf_region),disp=False)


    if save_name:
        save_array = [radii,measured_throughputs]
        thput_head = '# Radii of measurement (pix), '
        thput_head += 'Input fluxes (Multiple of stellar flux), '
        thput_head += 'Measured throughput (fraction)'
        np.savetxt(save_name,save_array,header=thput_head)
    return measured_throughputs

##################

def sphere_nd(hdr,ND_path_filename=None):
    '''
    Finds the ND filter transmission factor for a given SPHERE image file, using
    the information in the header and an ND filter file
    It is assumed that the ND filter is located in a subdirectory from the GRAPHIC directory.

    Also, the format of the ND filter file is assumed to be
    column 0: wavelength in nm
    columns 1-5: no ND, ND1, ND2, ND3.5
    '''

    if ND_path_filename == None:
        graphic_directory = os.path.dirname(os.path.realpath(__file__))
        ND_path_filename = graphic_directory+os.sep+'SPHERE_characterization/photometry_SPHERE/SPHERE_ND_filter_table.dat'

    Neutral_density=hdr['HIERARCH ESO INS4 FILT2 NAME']

    # Get the wavelength in microns from the header
    #       (added in cut_centre_cube_sphere_science_waffle)
    wavelength = hdr['HIERARCH GC WAVELENGTH'] *1e3


    ND_file = np.loadtxt(ND_path_filename,unpack=True)
    wavelength_vec = ND_file[0]

    if Neutral_density == 'OPEN':
        ND_transmission_vec = ND_file[1]
    elif Neutral_density == 'ND_1.0':
        ND_transmission_vec = ND_file[2]
    elif Neutral_density == 'ND_2.0':
        ND_transmission_vec = ND_file[3]
    elif Neutral_density == 'ND_3.5':
        ND_transmission_vec = ND_file[4]

    wav_difference = np.abs(wavelength_vec - wavelength)
    wav_ix = np.where(wav_difference == np.min(wav_difference))
    ND_transmission=ND_transmission_vec[wav_ix][0]

    return ND_transmission

##################

def correct_flux_frame_sphere(image_header,flux_header):
    ''' Calculate the correction factor that needs to be applied to the SPHERE flux image
    to account for exposure time and ND filter differences.
    This is used directly in GRAPHIC_contrast_pca
    '''

    # The ND values here are the transmission fraction (energy)
    flux_nd = sphere_nd(flux_header)
    image_nd = sphere_nd(image_header)

    flux_dit = flux_header['HIERARCH ESO DET SEQ1 REALDIT']
    image_dit = image_header['HIERARCH ESO DET SEQ1 REALDIT']

    # Calculate the amount we need to multiply the values in the flux image
    # to match the ND and exposure time for the science image
    nd_fact = image_nd / flux_nd
    expt_fact = image_dit / flux_dit

    return nd_fact*expt_fact

##################

##################

def naco_nd(nd_filter):
    ''' Returns the ND factor for a specfic ND filter'''
    if nd_filter == 'ND_Long':
        # We should really calculate the proper value by averaging over the
        # ND curve across the filter of interest
        # 57 is 1/0.0175, an estimate of the value at 3.8um from the NACO manual
        print('  WARNING: Assuming ND_Long reduces flux by 57x in graphic_contrast_lib!')
        return 57.
    else:
        print('  WARNING: Unknown ND filter '+str(nd_filter)+' in graphic_contrast_lib!')
        return 1.

##################


##################

def correct_flux_frame_naco(cube_header,flux_header):
    ''' Calculates the correction factor that needs to be applied to a NACO flux image
    to account for exposure time and ND filter differences.
    This is used directly in GRAPHIC_contrast_pca
    '''
    # Check the ND filter status for each file
    nd_filt_cube = cube_header['HIERARCH ESO INS OPTI3 ID']
    nd_filt_flux = flux_header['HIERARCH ESO INS OPTI3 ID']

    # Calculate the fraction of starlight rejected by the ND
    if nd_filt_cube == nd_filt_flux:
        nd_fact = 1.
    else:
        nd_fact = naco_nd(nd_filt_flux)/naco_nd(nd_filt_cube)


    # Calculate the difference in exposure time
    expt_fact = cube_header['ESO DET DIT']/flux_header['ESO DET DIT']


    flux_factor = nd_fact * expt_fact

    return flux_factor
