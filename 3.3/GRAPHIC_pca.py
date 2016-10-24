# -*- coding: utf-8 -*-
"""
This is an example script for how to use the pca module.
The output will go into the directory that the data is in (wdir)
e.g.
    - smart_annular_pca.fits for the PCA-subtracted frames
    - smart_annular_pca_derot.fits for the final derotated and combined image
"""
from mpi4py import MPI
import argparse
import graphic_nompi_lib_330 as graphic_nompi_lib
import sys, os
from pca import *

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Detection of the star center for corono images with the waffle pattern')
parser.add_argument('--cube_filename', action="store", dest="cube_filename",  default="master_cube_PCA.fits", help='cube to apply the pca algo to')
parser.add_argument('--paralactic_filename', action="store", dest="paralactic_filename",  default="paralactic_angle.txt", help='file with the paralactic angle of individual frames for the pca algo')
parser.add_argument('--pca_type', action="store", dest="pca_type",  default="smart_annular_pca", help='type of the pca algo to run (pca, smart_pca, annular_pca or smart_annular_pca)')
parser.add_argument('--n_modes', action="store", dest="n_modes", type=float,  default=10, help='Number of PCA modes to subtract')
parser.add_argument('--fwhm', action="store", dest="fwhm", type=float,  default=4.5, help='fwhm of the psf')
parser.add_argument('--n_fwhm', action="store", dest="n_fwhm", type=float,  default=0.75, help='Minimum rotation used to choose frames (in fwhm)')
parser.add_argument('--r_min', action="store", dest="r_min", type=float,  default=5, help='Minimum radius to apply PCA in pixel')
parser.add_argument('--arc_length', action="store", dest="arc_length", type=float,  default=50, help='Approximate arc length of the annuli (in pixels)')
parser.add_argument('--n_annuli', action="store", dest="n_annuli", type=float,  default=30, help='Minimum radius to apply PCA in pixel')

parser.add_argument('-median_combine', dest='median_combine', action='store_const', const=True, default=True, help='use a median for the combination of the frames instead of a mean')


args = parser.parse_args()
cube_filename=args.cube_filename
paralactic_filename=args.paralactic_filename
pca_type=args.pca_type
n_modes=args.n_modes
fwhm=args.fwhm
n_fwhm=args.n_fwhm
r_min=args.r_min
arc_length=args.arc_length
n_annuli=args.n_annuli

median_combine=args.median_combine

if rank==0:

	t0=MPI.Wtime()
	sys.stdout.write("beginning of pca")
	sys.stdout.write("\n")
	sys.stdout.flush()
	
	### The directory that the data cubes are in:
	wdir='./'

	# The name of the image file and the parallactic angle file
	"""image_file=wdir+'master_cube_PCA.fits'
	parang_file=wdir+'paralactic_angle.txt'"""

	image_file=wdir+cube_filename
	parang_file=wdir+paralactic_filename

	threads=3 # Number of cores to use

	# PCA options
	"""pca_type='smart_annular_pca' # pca, smart_pca, annular_pca or smart_annular_pca
	n_modes=8    # Number of PCA modes to subtract 
	fwhm= 4.5     # Approximate FWHM (in pixels)
	n_fwhm = 0.75  # Minimum rotation used to choose frames (in fwhm)
	r_min=5       # Minimum radius to apply PCA (in pixels)
	arc_length=50 # Approximate arc length of the annuli (in pixels)
	n_annuli=30   # Number of radial annuli to use. Works best if set to (n_pixels/2)/fwhm

	# Derotation options:
	median_combine=True # True to use median, False to use mean"""

	#########
	# Everything below here should be automatic
	#########

	if pca_type=='pca':
	    simple_pca(image_file,n_modes,wdir+pca_type+'.fits',threads=3)
	elif pca_type=='smart_pca':
	    smart_pca(image_file,n_modes,wdir+pca_type+'.fits',
		                         parang_file,protection_angle=15.)
	elif pca_type=='annular_pca':
	    annular_pca(image_file,n_modes,wdir+pca_type+'.fits',n_annuli=30,arc_length=50,r_min=5)
	elif pca_type=='smart_annular_pca':
	    smart_annular_pca(image_file,n_modes,wdir+pca_type+'.fits',parang_file,n_annuli=n_annuli,arc_length=arc_length,r_min=r_min,n_fwhm=n_fwhm,fwhm=fwhm,threads=threads)
	###
	if pca_type !='':
	    derotate_and_combine_multi(wdir+pca_type+'.fits',parang_file,
		           threads=threads,save_name=wdir+pca_type+'_derot.fits',
		           median_combine=median_combine)

	sys.stdout.write("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
	sys.stdout.write("\n")
	sys.stdout.write("pca finished")
	sys.stdout.flush()
	# os._exit(1)
	sys.exit(0)
else:
	# sys.exit(1)
	sys.exit(0)
