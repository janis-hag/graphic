#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline for
High-contrast Imaging of planetary Companions".
It creates a list containing information on each frame: frame quality,
gaussian fitted position of star, parallactic angle.

The output tables contain the following columns:
frame_number, psf_barycenter_x, psf_barycenter_y, psf_pixel_size,
psf_fit_center_x, psf_fit_center_y, psf_fit_height, psf_fit_width_x,
psf_fit_width_y,  frame_number, frame_time, paralactic_angle

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__='3.1'
__subversion__='0'

import numpy, scipy, glob, shutil, os, sys,argparse, time
 ## pickle, tables, argparse
from mpi4py import MPI
#from gaussfit_nosat import fitgaussian_nosat
#from gaussfit import fitgaussian, i_fitmoffat, moments
import gaussfit_310
from scipy import stats
import graphic_lib_310
import numpy as np
from astropy.io import fits as pyfits
import bottleneck
from scipy import ndimage

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser(description='GRAPHIC:\n The Geneva Reduction and Analysis Pipeline for High-contrast Imaging of planetary Companions.\n\n\
This program creates a list containing information on each frame: frame quality, gaussian fitted position of star, parallactic angle.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern", help='Filename pattern')
parser.add_argument('--win', action="store", dest="win", type=float, help='Gaussian fit window size')
parser.add_argument('--rmin', action="store", dest="rmin", type=float, default=5, help='The minimal radius in pixels to consider')
parser.add_argument('--rmax', action="store", dest="rmax", type=float, default=300, help='The Maximal spot size')
parser.add_argument('--t', action="store", dest="t", type=float, default=20, help='The threshold value')
parser.add_argument('--t_max', action="store", dest="t_max", type=float, help='The maximum deviation value')
parser.add_argument('--ratio', action="store", dest="ratio", type=float, help='PSF ellipticity ratio')
parser.add_argument('-deviation', dest='deviation', action='store_const',
				   const=True, default=False,
				   help='Use a threshold based on standard deviation instead of an absolute threshold')
parser.add_argument('-sat', dest='nosat', action='store_const',
				   const=False, default=True,
				   help='Discard saturated pixels OPTION CURRENTLY IGNORED')
parser.add_argument('--recurs', dest='recurs', action='store',
				   type=int, default=2500,
				   help='Change the recursion limit for the centroid search')
parser.add_argument('-no_neg', dest='no_neg', action='store_const',
				   const=True, default=False,
				   help='Null negative pixels before PSF fitting')
parser.add_argument('-moffat', dest='moffat', action='store_const',
				   const=True, default=False,
				   help='Use moffat fitting instead of gaussian')
parser.add_argument('-nofit', dest='nofit', action='store_const',
				   const=True, default=False,
				   help='No PSF fitting performed.')
parser.add_argument('-stat', dest='stat', action='store_const',
				   const=True, default=False,
				   help='Print benchmarking statistics')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('--size', action="store", dest="size", type=int, default=-1, help='Filter size')


args = parser.parse_args()
d=args.d
pattern=args.pattern
D=args.win
min_size=args.rmin
max_size=args.rmax
thres_coefficient=args.t
max_deviation=args.t_max
ratio=args.ratio
log_file=args.log_file
no_neg=args.no_neg
nosat=args.nosat
moffat=args.moffat
nofit=args.nofit
deviation=args.deviation
## hdf5=args.hdf5
recurs=int(args.recurs)
window_size=args.size

if moffat:
	header_keys=['frame_number', 'psf_barycenter_x', 'psf_barycenter_y', 'psf_pixel_size', 'psf_fit_center_x', 'psf_fit_center_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
		'frame_num', 'frame_time', 'paralactic_angle']
else:
	header_keys=['frame_number', 'psf_barycenter_x', 'psf_barycenter_y', 'psf_pixel_size', 'psf_fit_center_x', 'psf_fit_center_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
		'frame_num', 'frame_time', 'paralactic_angle']

target_dir = "."
backup_dir = "prev"
positions_dir = "cube-info"
iterations = 1
args=6
comments=None

sys.setrecursionlimit(recurs)
t_init=MPI.Wtime()

if rank==0:  # Master process
	# try:
	t0=MPI.Wtime()
	skipped=0

	print(thres_coefficient)

	dirlist=glob.glob(pattern+'*.fits')
	dirlist.sort() # Sort the list alphabetically

	if len(dirlist)==0:
		print("No files found!")
		for n in range(nprocs-1):
			comm.send("over", dest = n+1 )
			comm.send("over", dest = n+1 )
		sys.exit(1)

	for i in range(len(dirlist)):
		# Read cube header and data
		#header_in=pyfits.getheader(dirlist[i])
		t_cube=MPI.Wtime()
		#check if already processed
		## if hdf5:
			## filename='all_info_'+str(thres_coefficient)+"_"+str(min_size)+"_"+str(max_size)+"_"+dirlist[i][:-5]+'.hdf5'
		## else:
		filename='all_info_'+str(thres_coefficient)+"_"+str(min_size)+"_"+str(max_size)+"_"+dirlist[i][:-5]+'.rdb'

		if os.access(positions_dir + os.sep +filename, os.F_OK ):
			print("["+str(i+1)+"/"+str(len(dirlist))+"]: "+filename+" already exists. SKIPPING")
			skipped=skipped+1
			continue

		print("["+str(i+1)+"/"+str(len(dirlist))+"]: Processing "+str(dirlist[i]))

		if not os.access(dirlist[i], os.F_OK ): # Check if file exists
			print "Error: cannot access file "+dirlist[i]
			skipped=skipped+1
			continue
		else:
			cube,cube_header=pyfits.getdata(dirlist[i], header=True)

		# Get saturation level
		sat=graphic_lib_310.get_saturation(cube_header)
		comm.bcast(sat, root=0)
		# send_frames...
		graphic_lib_310.send_frames(cube)
		del cube

		# Creates a 2D array [frame_number, frame_time, paralactic_angle]
		parang_list=graphic_lib_310.create_parang_list_ada(cube_header)

		# Prepare the centroid array:
		# [frame_number, psf_barycenter_x, psf_barycenter_y, psf_pixel_size, psf_fit_center_x, psf_fit_center_y, psf_fit_height, psf_fit_width_x, psf_fit_width_y]
		cent_list=None

		# Receive data back from slaves
		for n in range(nprocs-1):
			data_in=None
			data_in=comm.recv(source = n+1)
			if data_in==None:
				continue
			elif cent_list == None:
				cent_list=data_in.copy()
			else:
				cent_list=np.vstack((cent_list,data_in))

		if not os.path.isdir(positions_dir): # Check if positions dir exists
			os.mkdir(positions_dir)

		if cent_list==None:
			print("No centroids list generated for "+str(dirlist[i]))
			continue

		if d>2:
			print("parang_list "+str(parang_list.shape)+" : "+str(parang_list))
			print("cent_list "+str(cent_list.shape)+" :" +str(cent_list))


		#Create the final list:
		# [frame_number, psf_barycenter_x, psf_barycenter_y, psf_pixel_size, psf_fit_center_x, psf_fit_center_y, psf_fit_height, psf_fit_width_x, psf_fit_width_y  ,  frame_number, frame_time, paralactic_angle]

		cent_list=np.hstack((cent_list,parang_list))

		# Set last frame to invalid if it's the cube-median
		if cube_header['NAXIS3']==cube_header['ESO DET NDIT']:
			cent_list[-1]=-1

		# Set first frame to invalid for L_prime band due to cube reset effects
		if cube_header['ESO INS OPTI6 ID']=='L_prime':
			cent_list[0]=-1

		if comments==None and not 'ESO ADA PUPILPOS' in cube_header.keys():
			comments="Warning! No ESO ADA PUPILPOS keyword found. Is it ADI? Using 89.44\n"

		## if hdf5:
			## # Open a new empty HDF5 file
			## f = tables.openFile(positions_dir+os.sep+filename, mode = "w")
			## # Get the root group
			## hdfarray = f.createArray(f.root, 'centroids', cent_list, "List of centroids for cube "+str(dirlist[i]))
			## f.close()
		## else:
		graphic_lib_310.write_array2rdb(positions_dir+os.sep+filename,cent_list,header_keys)

		if d>2:
			print("saved cent_list "+str(cent_list.shape)+" :" +str(cent_list))

		sys.stdout.write('\r\r\r')
		sys.stdout.flush()

		bad=np.where(cent_list[:,6]==-1)[0]
		print(dirlist[i]+" total frames: "+str(cent_list.shape[0])+", rejected: "+str(len(bad))+" in "+str(MPI.Wtime()-t_cube)+" seconds.")
		del cent_list

		t_cube=MPI.Wtime()-t_cube
		# print(" ETA: "+humanize_time(t_cube*(len(dirlist)-i-1)))
		print(" Remaining time: "+graphic_lib_310.humanize_time((MPI.Wtime()-t0)*(len(dirlist)-i-1)/(i+1-skipped)))

	if len(dirlist)==skipped: # Nothing to be done.
		MPI.Finalize()
		sys.exit(0)

	print("")
	print(" Total time: "+graphic_lib_310.humanize_time(MPI.Wtime()-t0))
	if not len(dirlist)-skipped==0:
		print(" Average time per cube: "+graphic_lib_310.humanize_time((MPI.Wtime()-t0)/(len(dirlist)-skipped))+" = "+str((MPI.Wtime()-t0)/(len(dirlist)-skipped))+" seconds.")

	log_file=log_file+"_"+cube_header['ESO OBS TARG NAME']+"_"+str(__version__)+".log"
	graphic_lib_310.write_log((MPI.Wtime()-t_init), log_file, comments)
	# Stop slave processes
	comm.bcast("over", root=0)
	for n in range(nprocs-1):
		comm.send("over", dest = n+1 )
		comm.send("over", dest = n+1 )
	sys.exit(0)

#except:
	print "Unexpected error:", sys.exc_info()[0]
	comm.bcast("over", root=0)
	for n in range(nprocs-1):
		comm.send("over", dest = n+1 )
		comm.send("over", dest = n+1 )

	for n in range(nprocs-1):
		check=comm.recv(source = n+1)
		if not check=="OK":
			print("Unexpected reply from slave("+str(n+1)+"). Expected OK, recieved:"+str(check))
			print("Ignoring error.")

	sys.exit(1)


else: # Slave processes

	sat=comm.bcast(None, root=0)
	startframe=comm.recv(source = 0) # get number of first frame
	data_in=comm.recv(source = 0)
	cube_count=1
	center=None
	x0_i=0
	y0_i=0

	while not data_in=="over":
		if not data_in==None:
			for frame in range(data_in.shape[0]):
				sys.stdout.write('\r\r\r [Rank '+str(rank)+', cube '+str(cube_count)+']  Frame '+str(frame+startframe)+' of '+str(startframe+data_in.shape[0]))
				sys.stdout.flush()

				if window_size>0:
					data_in[frame]=data_in[frame]-ndimage.filters.median_filter(data_in[frame],size=(window_size,window_size),mode='reflect')
				if deviation:
					sigma = data_in[frame].std()
					median = bottleneck.median(data_in[frame])
					threshold = sigma*thres_coefficient
					graphic_lib_310.dprint(d>1,"Sigma: "+str(sigma)+", median: "+str(median)+", threshold: "+str(threshold))
					data_in[frame] = data_in[frame]-median
					data_in[frame] = np.where(data_in[frame]>sigma*max_deviation, median, data_in[frame])
				else:
					threshold=thres_coefficient

				# Starting rough center search and quality check
				cluster_array_ref, ref_ima, count =graphic_lib_310.cluster_search(data_in[frame], threshold, min_size, max_size, x0_i , y0_i,d=d)
				if d>3:
					print("cluster_search, on frame["+str(frame)+"]: "+str(cluster_array_ref))
				if not count == 1: # Check if one and only one star has been found
					cluster_array_ref=np.array([-1 , -1, -1, -1 , -1 , -1, -1 , -1])
				elif nofit: # Skip psf fitting procedure, and copy values
					cluster_array_ref=np.append(cluster_array_ref,[cluster_array_ref[0],cluster_array_ref[1],cluster_array_ref[2], 1, 1])
				else:
					x0_i = np.ceil(cluster_array_ref[0])
					y0_i = np.ceil(cluster_array_ref[1])
					if d ==1:
						print('\r [Rank '+str(rank)+', cube '+str(cube_count)+'] Number of spots detected in frame ('+str(frame+1)+'): '+str(count)+'\n')

					# Check if D needs to be redefined
					if D+x0_i>data_in[frame].shape[0]:
						if d>2:
							print("Redefined D because: D+x0_i>data_in[frame].shape[0] "+str(D)+"+"+str(x0_i)+"="+str(D+x0_i)+">"+str(data_in[frame].shape[0]))
						D=data_in[frame].shape[0]-x0_i
					elif D+y0_i>data_in[frame].shape[1]:
						if d>2:
							print("Redefined D because: D+y0_i>data_in[frame].shape[1] "+str(D)+"+"+str(y0_i)+"="+str(D+y0_i)+">"+str(data_in[frame].shape[1]))
						D=data_in[frame].shape[1]-y0_i
					elif  x0_i-D<0:
						if d>2:
							print("Redefined D because: D>y0_i "+str(D)+">"+str(x0_i))
						D=x0_i
					elif y0_i-D<0:
						if d>2:
							print("Redefined D because: D>y0_i "+str(D)+">"+str(y0_i))
						D=y0_i

					# Starting gaussian fitting for better center determination
					## print(x0_i-D,x0_i+D,y0_i-D,y0_i+D)
					center_win=data_in[frame,x0_i-D:x0_i+D,y0_i-D:y0_i+D]
					if no_neg: # put negative values to zero
						center_win=np.where(center_win<0,0,center_win)

					## print('\n'+str(frame+startframe)+': '+str(moments(center_win))+', '+str(center_win))

					if d>1:
						print("")
						print("D: "+str(D)+", x0_i: "+str(x0_i)+", y0_i: "+str(y0_i))
						print("center_win.shape: "+str(center_win.shape))
					try:
					## if True:
						if moffat:
							g_param=gaussfit_310.i_fitmoffat(center_win)
						## elif nosat:
							## g_param=gaussfit_247.fitgaussian_nosat(center_win,sat)
						else:
							g_param=gaussfit_310.fitgaussian(center_win)
					except:
						print("\n Something went wrong with the PSF fitting!")
						g_param=np.array([-1 , -1, -1, -1 , -1 ])
					if d>2:
						print("")
						print("height: "+str(g_param[0])+" x, y: "+str(g_param[1])+", "+str(g_param[2])+", width_x, width_y: "+str(g_param[3])+", "+str(g_param[4]))

					x0_g=x0_i+g_param[1]-D
					y0_g=y0_i+g_param[2]-D

					if moffat:
						cluster_array_ref=np.append(cluster_array_ref,[x0_g, y0_g, g_param[0], g_param[3], g_param[4]])
					elif g_param[3]/g_param[4] < 1./ratio and g_param[3]/g_param[4] > ratio:
						cluster_array_ref=np.append(cluster_array_ref,[x0_g, y0_g, g_param[0], g_param[3], g_param[4]])
					else:
						print("\n PSF shape rejection: "+str(g_param[3]/g_param[4]))
						cluster_array_ref=np.append(cluster_array_ref,[-1 , -1, -1, -1 , -1 ])


				cluster_array_ref=np.append(frame+startframe,cluster_array_ref)


				# if rank==1:
				if center==None:
					center=cluster_array_ref
				else:
					center=np.vstack((center,cluster_array_ref))
			if d > 3:
				print(str(rank)+": "+str(center))
			comm.send(center, dest = 0)
		else:
			if d > 3:
				print(str(rank)+": "+str(center))
			comm.send(None,dest=0)

		cube_count=cube_count+1
		sat=comm.bcast(None, root=0)
		startframe=comm.recv(source = 0) # get number of first frame
		data_in=comm.recv(source = 0)
		center=None
		x_0=0
		y_0=0


	else:
		comm.send("OK", dest = 0)
		sys.exit(0)
