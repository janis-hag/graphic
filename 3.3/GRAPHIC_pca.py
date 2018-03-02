# -*- coding: utf-8 -*-
"""
This script runs PCA (or a simple cADI reduction) on a data cube.
The output will go into the directory that the data is in (wdir)
e.g.
	- smart_annular_pca.fits for the PCA-subtracted frames
	- smart_annular_pca_derot.fits for the final derotated and combined image

pca_type must be:
	'pca','smart_pca','annular_pca','smart_annular_pca', or 'cadi' 
"""
from mpi4py import MPI
import argparse
import graphic_nompi_lib_330 as graphic_nompi_lib
import sys, os
from pca import *
import astropy.io.fits as pf

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Detection of the star center for corono images with the waffle pattern')
parser.add_argument('--cube_filename', action="store", dest="cube_filename",  default="master_cube_PCA.fits", help='cube to apply the pca algo to')
parser.add_argument('--parallactic_filename', action="store", dest="parallactic_filename",  default="parallactic_angle.txt", help='file with the parallactic angle of individual frames for the pca algo')
parser.add_argument('--pca_type', action="store", dest="pca_type",  default="smart_annular_pca", help='type of the pca algo to run (pca, smart_pca, annular_pca or smart_annular_pca)')
parser.add_argument('--n_modes', action="store", dest="n_modes", type=float,  default=10, help='Number of PCA modes to subtract. If a decimal between 0 and 1, it will use this fraction of the total frames')
parser.add_argument('--fwhm', action="store", dest="fwhm", type=float,  default=4.5, help='fwhm of the psf')
parser.add_argument('--n_fwhm', action="store", dest="n_fwhm", type=float,  default=0.75, help='Minimum rotation used to choose frames (in fwhm)')
parser.add_argument('--r_min', action="store", dest="r_min", type=float,  default=5, help='Minimum radius to apply PCA in pixel')
parser.add_argument('--r_max', action="store", dest="r_max", type=float, default=0, help='Maximum radius to apply PCA in pixel (Default is whole field of view)')
parser.add_argument('--arc_length', action="store", dest="arc_length", type=float,  default=50, help='Approximate arc length of the annuli (in pixels)')
parser.add_argument('--n_annuli', action="store", dest="n_annuli", type=int,  default=30, help='Minimum radius to apply PCA in pixel')
parser.add_argument('--min_reference_frames', action="store", dest="min_reference_frames", type=float,  default=0, help='Minimum number of reference frames to use for PCA. This overrules the n_fwhm*fwhm criteria in smart_annular_pca. Values between 0 and 1 will be treated as a fraction of the total frames.')

parser.add_argument('-median_combine', dest='median_combine', action='store_const', const=True, default=False, help='use a median for the combination of the frames instead of a mean')
parser.add_argument('--output_dir', dest='output_dir',type=str, action='store', default='./', help='The output directory for the final and intermediate PCA products')
parser.add_argument('--threads', dest='threads', action='store', type=int, default=3, help='Number of cores to use for processing (multiprocessing rather than MPI)')
parser.add_argument('-save_derot_cube', dest='save_derot_cube', action='store_const', const=True, default=False, help='Save the derotated cube before summing to make the final image.')



args = parser.parse_args()
cube_filename=args.cube_filename
parallactic_filename=args.parallactic_filename
pca_type=args.pca_type
n_modes=args.n_modes
fwhm=args.fwhm
n_fwhm=args.n_fwhm
r_min=args.r_min
r_max=args.r_max
arc_length=args.arc_length
n_annuli=args.n_annuli
median_combine = args.median_combine
min_reference_frames = args.min_reference_frames
save_derot_cube = args.save_derot_cube

output_dir = args.output_dir
threads = args.threads

# Fix r_max
if r_max ==0:
	r_max='Default'

if rank==0:

	t0=MPI.Wtime()
	sys.stdout.write("beginning of pca")
	sys.stdout.write("\n")
	sys.stdout.flush()
	
	### The directory that the data cubes are in:
	wdir=''

	# The name of the input image cube file and the parallactic angle file
	image_file=wdir+cube_filename
	parang_file=wdir+parallactic_filename

	# The name of the output files
	pca_reduced_cube_file = output_dir+pca_type+'.fits'
	derotated_final_image = output_dir+pca_type+'_derot.fits'

	# Check that the output directory exists and create it if necessary
	dir_exists=os.access(output_dir, os.F_OK)
	if not dir_exists:
		os.makedirs(output_dir)

	# Now convert n_modes to an integer if it is a fraction of the available modes
	if (n_modes < 1) and (n_modes > 0):
		n_frames = pf.getheader(image_file)['NAXIS3']
		n_modes = np.int(n_modes*n_frames)
		print('  Taking n_modes as a fraction of the total frames')
		print('  i.e. '+str(n_modes))
	else:
		n_modes = np.int(n_modes)

	# Convert min_reference_frames to an integer if it is a fraction of the available modes
	if (min_reference_frames < 1) and (min_reference_frames > 0):
		n_frames = pf.getheader(image_file)['NAXIS3']
		min_reference_frames = np.int(min_reference_frames*n_frames)
		print('  Taking min_reference_frames as a fraction of the total frames')
		print('  i.e. '+str(min_reference_frames))
	else:
		min_reference_frames = np.int(min_reference_frames)
	
	#########
	# Everything below here should be automatic
	#########

	if pca_type=='pca':
		simple_pca(image_file,n_modes,pca_reduced_cube_file,threads=threads)
	elif pca_type=='smart_pca':
		smart_pca(image_file,n_modes,pca_reduced_cube_file,
								 parang_file,protection_angle=15.)
	elif pca_type=='annular_pca':
		annular_pca(image_file,n_modes,pca_reduced_cube_file,n_annuli=n_annuli,arc_length=arc_length,r_min=r_min,r_max=r_max)
	elif pca_type=='smart_annular_pca':
		print parang_file,os.getcwd()
		smart_annular_pca(image_file,n_modes,pca_reduced_cube_file,parang_file,n_annuli=n_annuli,
				   arc_length=arc_length,r_min=r_min,n_fwhm=n_fwhm,fwhm=fwhm,threads=threads,r_max=r_max,
				   min_reference_frames = min_reference_frames)
	elif pca_type.lower() == 'cadi':
		classical_adi(image_file,pca_reduced_cube_file,parang_file,median=False)
	elif pca_type.lower() == 'noadi':
		pca_reduced_cube_file = image_file

	###
	if pca_type !='':
		derot_cube=derotate_and_combine_multi(pca_reduced_cube_file,parang_file,
				   threads=threads,save_name=derotated_final_image,
				   median_combine=median_combine,return_cube=save_derot_cube)
		if save_derot_cube:
			pf.writeto(output_dir+pca_type+'_derot_cube.fits',derot_cube,clobber=True)

	sys.stdout.write("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
	sys.stdout.write("\n")
	sys.stdout.write("pca finished")
	sys.stdout.flush()
	
	sys.exit(0)
else:
	
	sys.exit(0)
