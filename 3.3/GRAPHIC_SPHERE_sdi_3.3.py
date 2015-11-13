#!/usr/bin/python
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

This step is to perform SDI subtraction on each single frame.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""
__version__='3.3'
__subversion__='0'

## import numpy, scipy, pyfits, glob, shutil, os, sys, time, fnmatch, tables, argparse, string
import numpy, scipy, glob, shutil, os, sys, time, fnmatch, argparse, string
import graphic_nompi_lib_330
import graphic_mpi_lib_330
from mpi4py import MPI
from scipy import ndimage
import astropy.io.fits as pyfits
import GRAPHIC_sphere_tables

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

target_dir = "."
backup_dir = "prev"

iterations = 1
coefficient = 0.95

parser = argparse.ArgumentParser(description='Performs SDI subtraction using two cubes.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern_right', action="store", dest="pattern_right",  default='right_*', help='Filename pattern for right cubes')
parser.add_argument('--pattern_left', action="store", dest="pattern_left",  default='left_*', help='Filename pattern for left cubes')
parser.add_argument('-noinfo', dest='no_info', action='store_const',
				   const=False, default=True,
				   help='Ignore info files')
parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
## parser.add_argument('--info_type', action="store", dest="info_type",  default='rdb', help='Info directory')
parser.add_argument('--info_pattern', action="store", dest="info_pattern", default='all_info', help='Info filename pattern')
parser.add_argument('-s', dest='stat', action='store_const',
				   const=True, default=False,
				   help='Print benchmarking statistics')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
## parser.add_argument('-hdf5', dest='hdf5', action='store_const',
				   ## const=True, default=False,
				   ## help='Switch to use HDF5 tables')
parser.add_argument('-nofit', dest='fit', action='store_const',
				   const=False, default=True,
				   help='Do not use PSF fitting values.')


args = parser.parse_args()
d=args.d
pattern_right=args.pattern_right
pattern_left=args.pattern_left
filter_size=args.size
info_dir=args.info_dir
## info_type=args.info_type
info_pattern=args.info_pattern
stat=args.stat
log_file=args.log_file
## hdf5=args.hdf5
fit=args.fit

skipped=0

t_init=MPI.Wtime()
target_pattern="mfs"+str(window_size)

header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
	'frame_num', 'frame_time', 'paralactic_angle']

if rank==0:
	graphic_nompi_lib_330.print_init()

	dirlist_right=graphic_nompi_lib_330.create_dirlist(pattern_right,target_dir=target_dir,target_pattern=target_pattern+"_")
	dirlist_left=graphic_nompi_lib_330.create_dirlist(pattern_left,target_dir=target_dir,target_pattern=target_pattern+"_")
	if dirlist_right==None or dirlist_left==None:
		print("No files found. Check --pattern option!")
		for n in range(nprocs-1):
			comm.send(None,dest =n+1)
		sys.exit(1)

	if args.no_info:
		infolist=glob.glob(info_dir+os.sep+info_pattern+'*.rdb')
		infolist.sort() # Sort the list alphabetically
		if len(infolist)<2:
			print("No info files found, check your --info_pattern and --info_dir options.")
			for n in range(nprocs-1):
				comm.send(None,dest =n+1)
		cube_list_right, dirlist_right=graphic_nompi_lib_330.create_megatable(dirlist_right,infolist,keys=header_keys,nici=nici,fit=fit)
		cube_list_left, dirlist_left=graphic_nompi_lib_330.create_megatable(dirlist_left,infolist,keys=header_keys,nici=nici,fit=fit)
		comm.bcast(cube_list_right, root=0)
		comm.bcast(cube_list_left, root=0)

	start_right,dirlist_right=graphic_mpi_lib_330.send_dirlist(dirlist_right)
	comm.bcast(dirlist_left, root=0)


	# Create directory to store reduced data
	if not os.path.isdir(target_dir):
		os.mkdir(target_dir)

if not rank==0:
	if args.no_info:
		cube_list_right=comm.bcast(None, root=0)
		cube_list_left=comm.bcast(None, root=0)

	dirlist_right=comm.recv(source = 0)
	if dirlist_right==None:
		sys.exit(1)
	start_right=int(comm.recv(source = 0))

	dirlist_left=comm.bcast(None, root=0)




t0=MPI.Wtime()


for i in range(len(dirlist_right)):
	targetfile_right=target_pattern+"_"+dirlist_right[i]
	targetfile_left=target_pattern+"_"+str.replace(dirlist_right[i],'right', 'left')

	# Check if left counterpart exists.
	if not str.replace(dirlist_right[i],'right', 'left') in dirlist_left:
		print(str.replace(dirlist_right[i],'right', 'left')+' not found. Skipping.')
		continue
	else left_file=str.replace(dirlist_right[i],'right', 'left')

	##################################################################
	#
	# Read cube header and data
	#
	##################################################################

	print(str(rank)+': ['+str(start+i)+'/'+str(len(dirlist_right)+start)+"] "+dirlist_right[i]+" Remaining time: "+graphic_lib_330.humanize_time((MPI.Wtime()-t0)*(len(dirlist)-i)/(i+1-skipped)))
	cube_right,header_right=pyfits.getdata(dirlist_right[i], header=True)
	cube_left,header_left=pyfits.getdata(left_file, header=True)


	if args.no_info:
		all_info_right=cube_list_right['info'][cube_list_right['cube_filename'].index(dirlist_right[i])]
		all_info_left=cube_list_left['info'][cube_list_left['cube_filename'].index(left_file)]

	## lambda1=1667.1
	## lambda2=1588.8

	lambdas=sphere_tables.sdi_wavelength(header_right)

	im_subtracted=graphic_nompi_lib_330.sdi(cube_left,cube_right,lambdas['left'],lambdas['right'],additional=False)

	## header["HIERARCH GC MEDIAN FILTER SIZE"]=(window_size, "")
	header["HIERARCH GC SPH SDI"]=( __version__+'.'+__subversion__, "")
	header["HIERARCH GC SPH SDI WAVELENGHT"]=(str(translate).translate(None,'{}\''),"")

	graphic_nompi_lib_330.save_fits(targetfile, cube, hdr=header,backend='pyfits')

print(str(rank)+": Total time: "+graphic_nompi_lib_330.humanize_time((MPI.Wtime()-t0)))

if rank==0:
	if 'ESO OBS TARG NAME' in header.keys():
		log_file=log_file+"_"+string.replace(header['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
	else:
		log_file=log_file+"_"+string.replace(header['OBJECT'],' ','')+"_"+str(__version__)+".log"

	graphic_nompi_lib_330.write_log((MPI.Wtime()-t_init),log_file, comments=None)
sys.exit(0)
