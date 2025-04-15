# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:39:56 2016
@author: cheetham


This program is a companion to GRAPHIC_pca, and calculates the contrast
as a function of radius, taking into account self subtraction.

Options:


"""

__version__ = '3.3'
__subversion__ = '0'

import numpy as np
import astropy.io.fits as pf
import graphic_contrast_lib
from scipy import interpolate
import argparse
import os
import pickle
import time
import sys

parser = argparse.ArgumentParser(
        description='This program calculates the contrast of a PCA reduced \
        dataset.')

parser.add_argument(
        '--cube_file', action="store",  dest="cube_file", type=str,
        default='master_cube_PCA.fits',
        help='The filename of the cleaned cube that PCA was performed on.')

parser.add_argument(
        '--psf_file', action="store",  dest="psf_file", type=str,
        default='flux.fits',
        help='The filename of the cleaned psf file, NOT corrected for \
        exposure time etc.')

parser.add_argument(
        '--parang_file', action="store", dest="parang_file", type=str,
        default='parallactic_angle.txt',
        help='The filename of the parallactic angles (in degrees).')

parser.add_argument(
        '--image_file', action="store", dest="image_file", type=str,
        default='GRAPHIC_PCA/smart_annular_pca_derot.fits',
        help='The filename of the final PCA reduced image.')

parser.add_argument(
        '--output_dir', action="store", dest="output_dir", type=str,
        default='GRAPHIC_PCA/', help='The folder to store the results in.')

parser.add_argument(
        '--threads', action="store",  dest="threads", type=int, default=3,
        help='Number of multiprocessing threads to use (not MPI).')

parser.add_argument(
        '--cutout_radius', action="store",  dest="cutout_radius", type=int,
        default=7,
        help='Number of pixels radius around the psf to use when fitting to \
        the flux of injected psfs.')

parser.add_argument(
        '--psf_pad', action="store",  dest="psf_pad", type=int, default=10,
        help='Number of pixels radius around the psf to pad when injecting \
        fake companions.')

parser.add_argument(
        '--n_sigma_inject', action="store",  dest="n_sigma_inject", type=int,
        default=7,
        help='Number of sigma for the injected PSFs (compared to initial \
        noise estimate).')

parser.add_argument(
        '--smooth_image_length', action="store", dest="smooth_image_length",
        type=np.float, default=1.25,
        help='Width of Gaussian used to smooth the final image before \
        calculating the contrast.')

parser.add_argument(
        '--plate_scale', action="store",  dest="plate_scale", type=np.float,
        default=1.,
        help='Plate scale in arcsec/pix. Default is 1, so the separations \
        will be in pixels.')

parser.add_argument(
        '--median_filter_length', action="store", dest="median_filter_length",
        type=int, default=20,
        help='Width of median filtered image subtracted from data to remove \
        large scale structure.')

parser.add_argument(
        '--n_radii', action="store", dest="n_radii", type=int, default=200,
        help='Number of radial points to use to calculate the contrast and \
        signal to noise map.')
parser.add_argument(
        '--r_max', action="store", dest="r_max", default=-1, type=np.float,
        help='Maximum radius (pix) for the contrast and SNR map.')

parser.add_argument(
        '--n_throughput', action="store",  dest="n_throughput", default=1,
        type=np.int,
        help='Number of times to repeat the throughput calculation, to get \
        an average value. Each one will use a different position angle')


args = parser.parse_args()

cube_file = args.cube_file
psf_file = args.psf_file
parang_file = args.parang_file
image_file = args.image_file
output_dir = args.output_dir
threads = args.threads
cutout_radius = args.cutout_radius
psf_pad = args.psf_pad
n_sigma_inject = args.n_sigma_inject
smooth_image_length = args.smooth_image_length
plate_scale = args.plate_scale
median_filter_length = args.median_filter_length
n_radii = args.n_radii
r_max = args.r_max
n_throughput = args.n_throughput

if threads > 1:
    import pca
else:
    import graphic_pca_lib as pca

print(sys.argv[0]+' started on ' + time.strftime("%c"))

# Fix the default r_max value so the other programs know it was not set
if r_max == -1:
    r_max = None

########################

# Set up the names of the output files
fp_name = 'fake_planets.fits'
fp_pca_name = 'fake_planets_pca.fits'
fp_derot_name = 'fake_planets_derot.fits'
throughput_file = output_dir + 'throughput.txt'
all_throughput_file = output_dir + 'all_throughputs.txt'
contrast_im_file = output_dir + 'contrast_im.fits'
contrast_file = output_dir + 'contrast.txt'
snr_map_file = output_dir + 'snr_map.fits'
noise_file = output_dir + 'noise.txt'


# Load the psf image, pca subtracted image and data cubes
# (and their headers)
psf_frame, psf_header = pf.getdata(psf_file, header=True)
image, header = pf.getdata(image_file, header=True)
cube, cube_header = pf.getdata(cube_file, header=True)

# Remove empt naxis3 if it exists
if psf_frame.shape[0] == 1:
    psf_frame = np.squeeze(psf_frame)

# Load the parallactic angles
parangs_deg = np.loadtxt(parang_file)
parangs_rad = parangs_deg*np.pi/180.

# Get the PCA params from the header
pca_type = header['HIERARCH GC PCA TYPE']
n_modes = np.int(header['HIERARCH GC PCA NMODES'])
n_fwhm = header['HIERARCH GC PCA NFWHM']
fwhm = header['HIERARCH GC PCA FWHM']
n_annuli = header['HIERARCH GC PCA NANNULI']
arc_length = header['HIERARCH GC PCA ARCLENGTH']
pca_r_min = header['HIERARCH GC PCA RMIN']
pca_r_max = header['HIERARCH GC PCA RMAX']
pca_input_file = header['HIERARCH GC PCA INPUTFILE']
min_reference_frames = np.int(header['HIERARCH GC PCA MINREFFRAMES'])

if pca_r_max == 'Default':
    pca_r_max = np.sqrt(2)*cube.shape[-1]/2

# Radii of injected companions
# Use the centre of every second PCA annulus
radii_edges = np.linspace(pca_r_min, pca_r_max, num=(n_annuli//2+1))
inject_radii = np.array([np.mean(
        radii_edges[ix:ix + 2]) for ix in range(n_annuli//2)])

# Quickly estimate the contrast so we have a rough idea of the flux
#   to use for the fake companion injection
noise, noise_radii = graphic_contrast_lib.noise_vs_radius(image, 2*n_annuli,
                                                          pca_r_min, fwhm)

# And estimate the flux of the star so we know what factor to multiply it by
# At the moment we treat pixels individually for the noise, so use radius=1
# If we calculate the noise using aperture photometry, use the same radius
stellar_flux = graphic_contrast_lib.measure_flux(psf_frame, radius=1)

# Calculate the fluxes to use for the fake companions.
# This is actually the multiplication factor for the psf
noise_interp = interpolate.interp1d(noise_radii, noise, kind='linear')
noise_at_inject_radii = noise_interp(inject_radii)
input_fluxes = n_sigma_inject*(noise_at_inject_radii/stellar_flux)

# Now we will inject the companions and measure their throughput
# Loop over this process so we can average out changes caused by throughput
# vs position angle
print('Iterating over throughput measurement')
all_throughputs = []
for ix in range(n_throughput):
    print('  Iteration '+str(ix+1)+' of '+str(n_throughput))
    intermediate_fp_derot_name = str(ix+1)+'.'+str(n_throughput)+'_'+fp_derot_name
    intermediate_throughputs_name = str(ix+1)+'.'+str(n_throughput)+'_throughputs.pickle'

#    if os.access(output_dir + os.sep + intermediate_fp_derot_name, os.F_OK):
#        print('Reading previously processed file.')
#        interm_fp_hdr = pf.getheader(
#                output_dir + os.sep + intermediate_fp_derot_name)
#        azimuth_offset = interm_fp_hdr['HIERARCH GC INJECT AZOFFSET']
    if os.access(output_dir + os.sep + intermediate_throughputs_name, os.F_OK):
        print('Reading previously processed throughputs.')
        with open(output_dir + os.sep + intermediate_throughputs_name, 'rb') as f:
            all_throughputs = pickle.load(f)
    else:
        # Randomly orient the line of injected psfs
        azimuth_offset = np.random.uniform(low=0., high=2*np.pi)  # radians

        # Inject the fake companions
        graphic_contrast_lib.inject_companions(
                cube_file, psf_frame, parangs_rad, inject_radii, input_fluxes,
                azimuth_offset=azimuth_offset, psf_pad=psf_pad,
                save_name=output_dir+fp_name)

        # Run PCA again with the same settings as the original
        pca.smart_annular_pca(
                output_dir+fp_name, n_modes, fp_pca_name, parang_file,
                n_annuli=n_annuli, arc_length=arc_length, r_max=pca_r_max,
                r_min=pca_r_min, n_fwhm=n_fwhm, fwhm=fwhm, threads=threads,
                min_reference_frames=min_reference_frames,
                output_dir=output_dir)

        # Derotate the image
        pca.derotate_and_combine_multi(output_dir+fp_pca_name, parang_file,
                                       threads=threads,
                                       save_name=intermediate_fp_derot_name,
                                       median_combine=True,
                                       output_dir=output_dir)

        # Now fit to the fluxes of the injected companions
        fp_derot_image, fp_derot_hdr = pf.getdata(
                output_dir+intermediate_fp_derot_name, header=True)

        measured_throughputs = graphic_contrast_lib.fit_injected_companions(
                fp_derot_image, psf_frame, inject_radii, input_fluxes,
                azimuth_offset=azimuth_offset, psf_pad=psf_pad,
                cutout_radius=cutout_radius, save_name=None)

        all_throughputs.append(measured_throughputs)

        with open(output_dir + os.sep + intermediate_throughputs_name, 'wb') as f:
            pickle.dump(all_throughputs, f)

# Now average over the repetitions and save it out
all_throughputs = np.array(all_throughputs)
measured_throughputs = np.nanmean(all_throughputs, axis=0)
# File with all of the individual values
np.savetxt(all_throughput_file, all_throughputs)
np.savetxt(throughput_file, [inject_radii, measured_throughputs])

# Correct the PSF for ND filters etc
# This needs to be properly implemented for NICI, SCEXAO etc.
if 'INSTRUME' in psf_header.keys():
    instrument = psf_header['INSTRUME'].strip()
else:
    raise Exception('Unknown instrument in GRAPHIC_contrast_pca')

if instrument == 'NAOS+CONICA':
    flux_factor = graphic_contrast_lib.correct_flux_frame_naco(
            cube_header, psf_header)
elif instrument == 'SPHERE':
    flux_factor = graphic_contrast_lib.correct_flux_frame_sphere(
            cube_header, psf_header)
else:
    raise Exception('Unknown instrument in GRAPHIC_contrast_pca: '+instrument)

psf_frame *= flux_factor
print('Multiplying the flux frame by:'+str(flux_factor))

# Apply cosmetics to the image and flux frame before measuring the contrast
graphic_contrast_lib.prepare_detection_image(
        image_file, save_name=contrast_im_file,
        smooth_image_length=smooth_image_length,
        median_filter_length=median_filter_length)
# Apply the same smoothing to the flux frame (but keep in memory rather than
# saving it)
# However, DON'T apply the median filter, since that would reduce the flux of
# the PSF much more than the image, due to the high SNR
psf_frame = graphic_contrast_lib.prepare_detection_image(
        psf_frame, smooth_image_length=smooth_image_length)

# Calculate the contrast
graphic_contrast_lib.contrast_curve(
        contrast_im_file, psf_frame, r_min=pca_r_min, r_max=r_max, fwhm=fwhm,
        plate_scale=plate_scale, self_subtraction_file=throughput_file,
        save_contrast=contrast_file, n_radii=n_radii, save_noise=noise_file,
        mad=True, robust_sigma=False)

# Make a signal-to-noise map
graphic_contrast_lib.snr_map(
        contrast_im_file, noise_file, remove_planet=False,
        planet_position=None, planet_radius=10., save_name=snr_map_file)

print('Cleaning up temporary files.')
for ix in range(n_throughput):
    intermediate_fp_derot_name = str(ix+1)+'.'+str(n_throughput)+'_'+fp_derot_name
    if os.access(output_dir + os.sep + intermediate_fp_derot_name,
                 os.F_OK | os.R_OK):
        os.remove(output_dir + os.sep + intermediate_fp_derot_name)

    intermediate_throughputs_name = str(ix+1)+'.'+str(n_throughput)+'_throughputs.pickle'
    if os.access(output_dir + os.sep + intermediate_throughputs_name,
                 os.F_OK | os.R_OK):
        os.remove(output_dir + os.sep + intermediate_throughputs_name)