#!/usr/bin/python2.6
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.
Its purpose is to generate medians of "n" cubes.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__='3.3'
__subversion__='0'

import numpy, scipy, glob, shutil, os, sys, fnmatch, string, time
import numpy as np
from scipy import stats
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
from mpi4py import MPI
import argparse
sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
import bottleneck
from graphic_mpi_lib_330 import dprint
from astropy.io import fits as pyfits

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

backup_dir = "prev"

parser = argparse.ArgumentParser(description='This program generates medians of a given number \"n\" of cubes. This can be used to generate flats.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0, help='Debug level, 0 is no debug')
parser.add_argument('--pattern', action="store", dest="pattern",  help='Filename pattern')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",  default='all*', help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
parser.add_argument('--win', action="store", dest="win",  type=int, default=40,  help='Window length in pixels (40)')
parser.add_argument('--sky_dir', action="store", dest="target_dir", default="sky-num", help='Info directory')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('--num', action="store",  dest="num",  default=-1, type=int, help='Number offset positions to use')
parser.add_argument('-s', dest='stat', action='store_const',
				   const=True, default=False,
				   help='Print benchmarking statistics')
parser.add_argument('-nofit', dest='fit', action='store_const',
				   const=False, default=True,
				   help='Do not use PSF fitting values.')
parser.add_argument('-mean', dest='mean', action='store_const',
				   const=True, default=False,
				   help='Use mean instead of median to calculate the sky value.')
parser.add_argument('-nici', dest='nici', action='store_const',
				   const=True, default=False,
				   help='Switch for GEMINI/NICI data')
parser.add_argument('-quad', dest='noquad', action='store_const',
				   const=False, default=True,
				   help='Ensure that all four quadrants are covered. Disbled by default as it is unreliable')
## parser.add_argument('-hdf5', dest='hdf5', action='store_const',
				   ## const=True, default=False,
				   ## help='Switch to use HDF5 tables')

args = parser.parse_args()
d=args.d
pattern=args.pattern
info_pattern=args.info_pattern
R=args.win
log_file=args.log_file
info_dir=args.info_dir
target_dir=args.target_dir
log_file=args.log_file
num=args.num
fit=args.fit
nici=args.nici
noquad=args.noquad
## hdf5=args.hdf5

header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
	'frame_num', 'frame_time', 'paralactic_angle']

if rank==0:
	graphic_nompi_lib.print_init()

	t_init=MPI.Wtime()
	dirlist=glob.glob(pattern+'*.fits')
	dirlist.sort() # Sort the list alphabetically
	## infolist=glob.glob(centroids_dir+os.sep+c_pattern+'*.hdf5')
	## infolist.sort() # Sort the list alphabetically

	## if hdf5:
		## infolist=glob.glob(info_dir+os.sep+'*'+info_pattern+'*.hdf5')
	## else:
	infolist=glob.glob(info_dir+os.sep+'*'+info_pattern+'*.rdb')
	infolist.sort() # Sort the list alphabetically

	cube_list,dirlist=graphic_nompi_lib.create_megatable(dirlist,infolist,keys=header_keys,nici=nici,fit=fit)

	if d>1:
		print("Dirlist:")
		print(dirlist)

	if len(dirlist)==0:
		print("No files found!")
		comm.Abort()
		sys.exit(1)

	if len(infolist)==0:
		print("No centroid lists found!")
		comm.Abort()
		sys.exit(1)

	if num==-1:
		print('No --num argument given. Creating one single median cube.')
		num=len(dirlist)
	## cube_list,dirlist, skipped=create_megatable(dirlist,infolist,0)

	t0=MPI.Wtime()
	# Create directory to store reduced data
	if not os.path.isdir(target_dir):
		os.mkdir(target_dir)

	print("Reading headers")
	# Initialise
	skylist = {}
	obstimes=[]
	mean_obstimes={}

	# Loop over cubes to organise the sky frames into groups (stored in skylist)
	for cube_ix in range(len(cube_list['cube_filename'])):
		header=pyfits.getheader(cube_list['cube_filename'][cube_ix])
		obstimes.append(header['MJD-OBS'])

	# Now sort them by observation time
	obstimes=np.array(obstimes)
	sorted_ix=np.argsort(obstimes)

	# Break them into groups of num with a for-loop
	for ix in range(len(cube_list['cube_filename'])):
		# Work out what cube number we're up to
		cube_ix=sorted_ix[ix]

		# First loop we have to set up an array to track the filenames used for the current sky
		if (ix % num) ==0:
			this_skylist=[]
			these_obstimes=[]

		# Add the current filename to the list
		this_skylist.append(cube_list['cube_filename'][sorted_ix[ix]])
		these_obstimes.append(obstimes[cube_ix])

		# If we have enough, save it and the mean obstime
		if len(this_skylist) == num:
			skylist[cube_list['cube_filename'][cube_ix]]=this_skylist
			mean_obstimes[cube_list['cube_filename'][cube_ix]]=np.mean(these_obstimes)

	if d>1:
		print('skylist: '+str(skylist))

	skipped=0

	# Now go through the lists of sky frames and combine them.
	for k in skylist.keys():
		skyfile="med"+str(num)+"_"+k
		# Check if already processed
		if os.access(target_dir+os.sep+skyfile, os.F_OK | os.R_OK):
			print('Already processed: '+skyfile)
			skipped=skipped+1
			continue

		header=pyfits.getheader(k)
		header["HIERARCH GC NAN_N_MED"]=( __version__+'.'+__subversion__, "")
		header.add_history("This sky is made from masked median of:")
		header["HIERARCH GC SKY_OBSTIME"]=(mean_obstimes[k],"mean MJD of sky frame")

		i=1

		for cube_name in skylist[k]:
			ci=cube_list['cube_filename'].index(cube_name)
			sys.stdout.write('\r\r\r Reading cube '+str(i)+' of '+str(len(skylist[k]))+' : '+str(cube_name))
			sys.stdout.flush()
			i=i+1
			cube_in,header_in=pyfits.getdata(cube_name, header=True)

			if nici:
				# Dithering within cubes can happen for NICI data
				for f in range(len(cube_in.shape)):
					cube_in[f]=graphic_nompi_lib.nanmask_frame_nici(cube_list['info'][ci][f,4], cube_list['info'][ci][f,5], cube_in[f], R, d)
			else:
				valid=np.where(cube_list['info'][ci][:,4]>0)
				if len(valid[0])==0:
					dprint(d>2, "No valid frame in: "+str(cube_name))
					continue
				## cube_in=graphic_nompi_lib.nanmask_cube(cube_list['info'][ci][int(header_in['NAXIS3'])/2,4], cube_list['info'][ci][int(header_in['NAXIS3'])/2,5], cube_in, R, d)
				else:
					cube_in=graphic_nompi_lib.nanmask_cube(np.mean(cube_list['info'][ci][valid,4]), np.mean(cube_list['info'][ci][valid,5]), cube_in, R, d)
			dprint(d>1, "cube_in.shape: "+str(cube_in.shape))
			graphic_mpi_lib.send_chunks(cube_in, d)
			del cube_in
			header.add_history(cube_name)

		# send compute
		print("")
		print("Calculating median...")

		for p in range(nprocs-1):
			comm.send("compute", dest=p+1)
			comm.send("compute", dest=p+1)

		for p in range(nprocs-1):
			r=comm.recv(source = p+1)
			graphic_mpi_lib.dprint(d>0, "Received median reduced chunk from "+str(p+1))
			if p==0: #initialise
				sky=r
			else:
				sky=np.concatenate((sky,r), axis=0)

		## graphic_nompi_lib.save_fits(skyfile,  target_dir, sky, header )
		graphic_nompi_lib.save_fits(skyfile, sky, hdr=header, target_dir=target_dir,backend='pyfits')

	graphic_nompi_lib.write_log_hdr((MPI.Wtime()-t_init), log_file, header, comments=None, 	nprocs=nprocs)

	## send over
	for p in range(nprocs-1):
		comm.send("over", dest=p+1)
		comm.send("over", dest=p+1)
	MPI.Finalize()

	sys.exit(0)



else: #if not rank == 0

	start=comm.recv(source = 0)
	if start=="over":
		dprint(d>0, "Received \'over\' command, leaving.")
		sys.exit(0)

	data_in=comm.recv(source = 0)
	if d >2:
		print(data_in)
	if not type(data_in)==type("compute"):
		my_cube=data_in
	if d>1:
		print(str(rank)+" received cube shape: "+str(my_cube.shape))
	if d>1:
		print(str(rank)+" received chunk")

	start=comm.recv(source = 0)
	if d>1:
		print(str(rank)+": start: "+str(start))
	data_in=comm.recv(source = 0)
	if d>2:
		print(str(rank)+": data_in: "+str(data_in))
	if d>1:
		print(str(rank)+" received masked cube shape: "+str(data_in.shape))

	while not start=="over":
		if isinstance(start,str) or isinstance(data_in,str):
			if start=="compute" or data_in=="compute":
				dprint(d>0, "calculating median.")
				try:
					my_cube.shape
				except:
					print(my_cube)
				if args.mean:
					my_cube=bottleneck.nanmean(1.*my_cube,axis=0)
				else:
					my_cube=bottleneck.nanmedian(1.*my_cube,axis=0)
				comm.send(my_cube, dest = 0)
				my_cube=None
				data_in=None
				start=comm.recv(source = 0)
				data_in=comm.recv(source = 0)
				try:
					data_in.shape
					dprint(d>1," received cube shape: "+str(data_in.shape)+", start: "+str(start))
				except:
					dprint(d>1," instead of cube chunk received: "+str(data_in))

		elif type(start)==type(None):
			print("Received None... nothing yet implemented to handle this case.")
			sys.exit(1)

		else:
			dprint(d>2,"data_in: "+str(data_in))
			if type(data_in)==type(None) or type(my_cube)==type(None):
				dprint(d>1,"start: "+str(start)+", my_cube: "+str(my_cube)+", data_in: "+str(data_in))
			else:
				 dprint(d>1,"start: "+str(start)+", my_cube.shape: "+str(my_cube.shape)+", data_in.shape: "+str(data_in.shape))

			if type(my_cube)==type(None):
				my_cube=data_in
			else:
				my_cube=np.concatenate((my_cube,data_in),axis=0)
			start=comm.recv(source = 0)
			data_in=comm.recv(source = 0)
			if d>1:
				if start==None or start=="over" or start=="compute":
					print(str(rank)+": Received: "+str(start))
				else:
					print(str(rank)+": Received cube shape: "+str(data_in.shape))

	else:
		dprint(d>0, "Received \'over\' command, leaving.")
		sys.exit(0)
