# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/home/spectro/peretti/GRAPHIC/version_seb/SPHERE_characterization/')
sys.path.insert(0, '/home/spectro/peretti/GRAPHIC/version_seb/SPHERE_characterization/photometry_SPHERE')

from mpi4py import MPI
import argparse

import signal_to_noise
reload(signal_to_noise)
from signal_to_noise import *
import comp_detection
reload(comp_detection)
from comp_detection import *
import companion_extraction
reload(companion_extraction)
from companion_extraction import *
import adi_self_subtraction_computation 
reload(adi_self_subtraction_computation)
from adi_self_subtraction_computation import *

rank   = MPI.COMM_WORLD.Get_rank()
wdir='/dace/work/peretti/SPHERE/SPHERE_Long_RV_Planet/HD4113/test_pipeline/'
path_results=wdir+"Contrast_curve_resulting_files/"

parser = argparse.ArgumentParser(description='Detection and Characterization of point sources')
parser.add_argument('--pattern_image', action="store", dest="pattern_image",  default="nomask*sdi", help='pattern of the image where we look for companions')
parser.add_argument('--pattern_psf', action="store", dest="pattern_psf",  default="psf_left", help='pattern of the psf file')
parser.add_argument('--pattern_image_FP', action="store", dest="pattern_image_FP",  default="nomask*FP*left", help='pattern of fake companion file')
parser.add_argument('--pixel_scale', action="store", dest="pixel_scale", type=float,  default=0.01227, help='pixel scale of the image')
parser.add_argument('--sep_FP', dest='sep_FP', type=float, nargs='+',required=True,help='Fake companion separations')
parser.add_argument('--dmag_given', action="store", dest="dmag_given", type=float,  default=10, help='dmag given for fake companion injection in order to compute the adi self subtraction')
parser.add_argument('--fwhm', action="store", dest="fwhm", type=float,  default=4, help='fwhm of the companion for masking it in order to compute the S/N')
parser.add_argument('-no_comp_detection', dest='no_comp_detection', action='store_const',const=True, default=False,help='Don t try to detec companions (in case of manual detection with ds9)')

args = parser.parse_args()
pattern_image=args.pattern_image
pattern_psf=args.pattern_psf
pattern_image_FP=args.pattern_image_FP
pixel_scale=args.pixel_scale
sep_FP=args.sep_FP
dmag_given=args.dmag_given
fwhm=args.fwhm
no_comp_detection=args.no_comp_detection

if rank==0:

	sep_FP=np.array(sep_FP)

	import os
	try:
		os.stat(path_results)
	except:
		os.mkdir(path_results)

	##########################
	#routines
	##########################
	if not no_comp_detection:
		print "sdi"
		print "\nStart of companion detection\n"
		comp_detection(wdir,pattern_image)
	print "\nStart of ADI self subtraction\n"
	adi_self_subtraction_computation(wdir,pattern_image_FP,pattern_psf,sep_FP,pixel_scale,dmag_given)

	print "\nStart of astrometry and photometry companion extraction\n"
	companion_extraction(wdir,path_results,pattern_image,pattern_psf,pixel_scale)
	print "\nStart of Signal to noise computation\n"
	signal_to_noise(wdir,path_results,pattern_image,pixel_scale,fwhm)
	os._exit(1)
else:
	sys.exit(1)
