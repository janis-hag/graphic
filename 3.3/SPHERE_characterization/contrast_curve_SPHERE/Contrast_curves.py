# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/home/spectro/peretti/GRAPHIC/version_seb/SPHERE_characterization/')
sys.path.insert(0, '/home/spectro/peretti/GRAPHIC/version_seb/SPHERE_characterization/photometry_SPHERE')
from mpi4py import MPI
import argparse

import graphic_contrast_adi
reload(graphic_contrast_adi)
from graphic_contrast_adi import *
import contrast_curve_sdi
reload(contrast_curve_sdi)
from contrast_curve_sdi import *

rank   = MPI.COMM_WORLD.Get_rank()
wdir='./'
path_results=wdir+"Contrast_curve_resulting_files/"

parser = argparse.ArgumentParser(description='Detection and Characterization of point sources')
parser.add_argument('--pattern_image', action="store", dest="pattern_image",  default="nomask*sdi", help='pattern of the image where we look for companions')
parser.add_argument('--pattern_psf', action="store", dest="pattern_psf",  default="psf_left", help='pattern of the psf file')
parser.add_argument('--age', action="store", dest="age",  default="1.000", help='age for models in order to compute the sdi detection limit in mass (needs a string with 3 decimal points)')
parser.add_argument('--pixel_scale', action="store", dest="pixel_scale", type=float,  default=0.01227, help='pixel scale of the image')
parser.add_argument('--fwhm', action="store", dest="fwhm", type=float,  default=4, help='fwhm of the point sources in the image')
parser.add_argument('--n_r', action="store", dest="n_r", type=float,  default=200, help='number of radial points for the plot')
parser.add_argument('--rmin', action="store", dest="rmin", type=float,  default=4, help='minimum radius (pixels) for the ADI contrast curve')
parser.add_argument('--smoothing_length', action="store", dest="smoothing_length", type=float,  default=3, help='minimum radius (pixels) for the ADI contrast curve')
parser.add_argument('-no_contrast_in_mags', dest='contrast_in_mags', action='store_const',const=False, default=True,help='compute the contrast in mag')
parser.add_argument('-convol_with_circ_ap', dest='convol_with_circ_ap', action='store_const',const=True, default=False,help='convolve the image with a circular apperture')
parser.add_argument('-smooth_image', dest='smooth_image', action='store_const',const=True, default=False,help='smooth the image for the contrast curve computation')


args = parser.parse_args()
pattern_image=args.pattern_image
pattern_psf=args.pattern_psf
age=args.age
pixel_scale=args.pixel_scale
fwhm=args.fwhm
n_r=args.n_r
rmin=args.rmin
smoothing_length=args.smoothing_length
contrast_in_mags=args.contrast_in_mags
convol_with_circ_ap=args.convol_with_circ_ap
smooth_image=args.smooth_image


if rank==0:

	import os
	try:
		os.stat(path_results)
	except:
		os.mkdir(path_results)

	smoothing_length=smoothing_length*fwhm # pixels
	##########################
	#routines
	##########################
	print("\nStart of ADI contrast curve computation\n")
	graphic_contrast_adi(wdir,path_results,pattern_image,pattern_psf,fwhm,n_r,rmin,pixel_scale,smoothing_length,contrast_in_mags,convol_with_circ_ap,smooth_image)
	print("\nEnd of ADI contrast curve computation. Resulting files are in directory: Contrast_curve_resulting_files\n")
	if "sdi" in pattern_image:
		print("\nStart of SDI contrast curve computation\n")
		contrast_curve_sdi(wdir,pattern_image,age,pixel_scale)
		print("\nEnd of SDI contrast curve computation. Resulting files are in directory: Contrast_curve_resulting_files\n")
	os._exit(1)
else:
	sys.exit(1)
