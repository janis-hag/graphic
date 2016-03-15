#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of the
Parallel Angular Differential Imaging Pipeline (PADIP).

Its purpose is to subtract the star PSF on each frame in order to find
any hidden companion. To achieve this a PSF is generate for each
single frame.

If you find any bugs or have any suggestions email:
janis.hagelberg@unige.ch
"""

__version__='3.3'
__subversion__='0'

import scipy, glob, shutil, os, sys, time, fnmatch, argparse, string, time
import numpy as np
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
from mpi4py import MPI
## from astropy.io import fits as pyfits
import astropy.io.fits as pyfits
import bottleneck

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

target_dir = "."
backup_dir = "prev"

iterations = 1
coefficient = 0.95


parser = argparse.ArgumentParser(description='Subtracts the sky on each frame of the cube.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern",  default='*', help='Filename pattern')
parser.add_argument('--sky_pattern', action="store", dest="sky_pattern", help='Sky file pattern')
parser.add_argument('--sky_dir', action="store", dest="sky_dir", default='sky-OB', help='Give alternative sky directory')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",
					default='all_info', help='Info filename pattern.')
parser.add_argument('--info_dir', action="store", dest="info_dir",
					default='cube-info', help='Info directory')
parser.add_argument('-noinfo', dest='noinfo', action='store_const',
				   const=True, default=False,
				   help='Do not use PSF fitting values.')
parser.add_argument('-s', dest='stat', action='store_const',
				   const=True, default=False,
				   help='Print benchmarking statistics')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('-nofit', dest='fit', action='store_const',
				   const=False, default=True,
				   help='Do not use PSF fitting values.')
parser.add_argument('-norm', dest='normalise', action='store_const',
				   const=True, default=False,
				   help='Normalise the sky before subtracting.')
parser.add_argument('-nici', dest='nici', action='store_const',
				   const=True, default=False,
				   help='Switch for GEMINI/NICI data')
## parser.add_argument('-hdf5', dest='hdf5', action='store_const',
				   ## const=True, default=False,
				   ## help='Switch to use HDF5 tables')
parser.add_argument('-interactive', dest='interactive', action='store_const',
				   const=True, default=False,
				   help='Switch to set execution to interactive mode')
parser.add_argument('--flat_filename', dest='flat_filename', action='store',
				   default=None, help='Name of flat field to be used. If this argument is not set, the data will not be flat fielded')

args = parser.parse_args()
d=args.d
pattern=args.pattern
sky_pattern=args.sky_pattern
sky_dir=args.sky_dir
info_pattern=args.info_pattern
info_dir=args.info_dir
stat=args.stat
log_file=args.log_file
fit=args.fit
nici=args.nici
flat_filename=args.flat_filename
## hdf5=args.hdf5

header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
	'frame_num', 'frame_time', 'paralactic_angle']

skipped=0
header=None

t_init=MPI.Wtime()


if args.noinfo:
	infolist=None
	cube_list=None

if rank==0:
	graphic_nompi_lib.print_init()

	dirlist=graphic_nompi_lib.create_dirlist(pattern)

	if dirlist==None or len(dirlist)<1:
		print("No files found")
		MPI.Finalize()
		sys.exit(1)

	## if hdf5:
		## infolist=glob.glob(info_dir+os.sep+info_pattern+'*.hdf5')
	## else:
	if not args.noinfo:
		if info_pattern=='all_info':
			print('Warning, using default value: info_pattern=\"all_info\" wrong info file may be used.')
		infolist=graphic_nompi_lib.create_dirlist(info_dir+os.sep+info_pattern,extension='.rdb')
		cube_list,dirlist=graphic_nompi_lib.create_megatable(dirlist,infolist,keys=header_keys,nici=nici,fit=fit)

	print('Distributing dirlist to slaves.')
	start,dirlist=graphic_mpi_lib.send_dirlist(dirlist)

	comm.bcast(cube_list, root=0)

	skyls=glob.glob(sky_dir+os.sep+sky_pattern+"*.fits")
	skyls.sort()

	sky_med_frame=None

	if len(skyls)<1:
		print("No sky file found")
		comm.bcast("over", root=0)
		sys.exit(1)

	elif len(skyls)==1:
		print("One single sky file found. Will be used for all files.")
		skylist={}
		skylist[skyls[0]] = []
		for sfile in dirlist:
			skylist[skyls[0]].append(sfile)

	else:
		skylist={}
		for skyfits in skyls:
			sky_data, sky_hdr = pyfits.getdata(skyfits, header=True)
			## sky_hdr=sky_hdulist[0].header
			#hdr=pyfits.getheader(skyfits)
			skylist[skyfits]=fnmatch.filter(sky_hdr['history'],"*.fits")
			if type(sky_med_frame)==None:
				sky_med_frame=sky_data
			else:
				sky_med_frame=np.dstack((sky_med_frame,sky_data))
			## sky_hdulist.close()
		if not sky_med_frame==None:
			sky_med_frame=bottleneck.nanmedian(sky_med_frame, axis=2)
		if d>2:
			print("skylist: "+str(skylist))

	if sky_med_frame is None:
		graphic_mpi_lib.dprint(d>1,'Warning: sky_med_frame is empty')
		sky_med_frame=0
	comm.bcast(skylist, root=0)
	comm.bcast(sky_med_frame, root=0)
	# Create directory to store reduced data
	if not os.path.isdir(target_dir):
		os.mkdir(target_dir)

if not rank==0:
	dirlist=comm.recv(source = 0)
	if dirlist==None:
		sys.exit(0)

	start=int(comm.recv(source = 0))

	cube_list=comm.bcast(None, root=0)
	skylist=comm.bcast(None, root=0)
	sky_med_frame=comm.bcast(None, root=0)

	if skylist==None:
		sys.exit(1)

skyfile=None
t0=MPI.Wtime()

# If a flat field was provided, load it
if flat_filename:
	flat=pyfits.getdata(flat_filename)

for i in range(len(dirlist)):
	targetfile="no"+string.replace(sky_pattern,'_','')+"_"+dirlist[i]
	if os.access(target_dir+os.sep+targetfile, os.F_OK | os.R_OK):
		print('Already processed: '+targetfile)
		skipped=skipped+1
		continue

	###################################################################
	#
	# Read cube header and data
	#
	###################################################################

	print(str(rank)+': ['+str(start+i)+'/'+str(len(dirlist)+start)+"] "+dirlist[i]+" Remaining time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)*(len(dirlist)-i)/(i+1-skipped)))
	cube,header=pyfits.getdata(dirlist[i], header=True)
	## hdulist = fits.open(dirlist[i])
	## header=hdulist[0].header
	## cube=hdulist[0].data

	found=False

	if not args.noinfo:
		all_info=cube_list['info'][cube_list['cube_filename'].index(dirlist[i])]
	else:
		all_info='empty'

	for e in range(13): # e will shorten the filename progressively until it finds a sky that was made with a file taken at a matching time.
		if d>3:
			print(str(rank)+"e "+str(e)+": "+str(dirlist[i][-28:-5-e+1]))
		if found:
			if e > 1:
				print(str(rank)+": No sky found for "+dirlist[i][-28:]+", using closest sky file: "+str(skyfile)+" matching "+str(fnmatch.filter(skylist[skyfile],"*"+dirlist[i][-28:-5-e+1]+"*.fits")))
			break

		for key in skylist.keys(): # loop through the sky files
			if len(fnmatch.filter(skylist[key],"*"+dirlist[i][-28:-5-e]+"*.fits"))>0 or len(skylist)==1:
				if key==skyfile:
					found=True
					if d >1:
						print(str(rank)+": key="+str(key)+", skyfile="+str(skyfile))
				elif len(skylist)==1:
					# Only one sky file, no need to search further
					found=True
					skyfile=key
				else:
					skyfile=key
					if not os.access(skyfile, os.F_OK ): # Check if file exists
						print("Error: cannot access file "+skyfile)
						skipped=skipped+1
						continue


				sky,sky_header=pyfits.getdata(skyfile, header=True)
				## sky_hdulist = fits.open(skyfile)
				## sky=sky_hdulist[0].data

				if cube.shape[1]==1024:
					# Strip overscan
					sky=sky[:1024,:]

				elif cube.shape[1]==512:
					# Strip overscan
					sky=sky[:512,:]

				if args.normalise:
					skymed=bottleneck.nanmedian(1.*sky)
					if skymed==0:
						skymed=1
					graphic_mpi_lib.dprint(d>1,str(rank)+": skymed="+str(skymed))
					break
	if found==False:
		continue

	if args.normalise:
		for frame in range(cube.shape[0]):
			if all_info=='empty' or not all_info[frame,6]==-1:
				graphic_mpi_lib.dprint(d>1, 'Normalising factors: skymed='+str(skymed)+' frame_median='+str(bottleneck.nanmedian(1.*cube[frame])))
				cube[frame]=cube[frame]-(bottleneck.nanmedian(1.*cube[frame])/skymed)*np.where(np.isnan(sky), sky_med_frame, sky)
			else:
				graphic_mpi_lib.dprint(d>0,'Skipping frame '+str(frame))
				continue
	else:
		graphic_mpi_lib.dprint(d>1, "Not normalising")
		graphic_mpi_lib.dprint(d>2, "len(np.where(np.isnan(sky)[0])):"+str(len(np.where(np.isnan(sky))[0])))
		## cube[frame]=cube[frame]-np.where(np.isnan(sky), sky_med_frame, sky)
		cube=cube-np.where(np.isnan(sky), sky_med_frame, sky)

	## sky_hdulist.close()

	skyref=string.split(skyfile,os.sep)[-1] # Strip directory away.
	## if len(skyref)>59: # value to long for FITS (limited to 80 chars - 18 for keyword name and 3 for ' = '
		## skyref

	# Flat field the data if a filename was provided
	if flat_filename:
		cube=cube/flat # this should work if cube is 2d or 3d
		header['HIERARCH GC FLAT_FIELD']=(flat_filename,'Filename of flat field')

	if args.normalise:
		header['HIERARCH GC SUB_SK_MED']=(skymed,'Median of sub. sky')
	header['HIERARCH GC SUB_SKY']=( __version__+'.'+__subversion__, '')
	header['HIERARCH GC SKYREF']=( skyref[-58:], '')
	## graphic_nompi_lib.save_fits( targetfile,  target_dir, cube, header )
	## graphic_nompi_lib.save_fits(targetfile,cube,hdr=header, backend='pyfits')
	graphic_nompi_lib.save_fits(targetfile, cube, hdr=header, backend='pyfits', verify='warn')

	## graphic_nompi_lib.save_fits(targetfile, hdulist, backend='astropy', verify='warn')

	## hdulist.close()

if rank==0:
	if not header==None:
		## if 'ESO OBS TARG NAME' in header.keys():
			## log_file=log_file+"_"+string.replace(header['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
		## else:
			## log_file=log_file+"_"+string.replace(header['OBJECT'],' ','')+"_"+str(__version__)+".log"
		## graphic_nompi_lib.write_log((MPI.Wtime()-t_init),log_file)

		graphic_nompi_lib.write_log_hdr((MPI.Wtime()-t_init), log_file, header, comments=None, 	nprocs=nprocs)

print(str(rank)+": Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
sys.exit(0)
