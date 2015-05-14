#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline for
High-contrast Imaging of planetary Companions".

Its purpose is to recentre the frames in the cubes, and collapse them to create
smaller cubes with less frames. It is part of the pipeline's quick-look branch.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__='3.2'
__subversion__='0'

import numpy, scipy, glob,  os, sys, subprocess, string, time
import numpy as np
import graphic_lib_320
from scipy import ndimage
from mpi4py import MPI
import argparse
from graphic_lib_320 import dprint
import astropy.io.fits as pyfits

## sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
import bottleneck


nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD


parser = argparse.ArgumentParser(description='Creates cubes with less frames by median-combining frames.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern",  default='*', help='Filename pattern')
## parser.add_argument('--info_pattern', action="store", dest="info_pattern", required=True, help='Info filename pattern')
## parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('--naxis3', action="store", dest="naxis3", required=True, type=int, help='The number of frames to bin')
## parser.add_argument('--lmax', action="store", dest="l_max", type=float, default=0,
					## help='Shape of the final image. If not specified it will be calculated to fit all the images.')
parser.add_argument('-nofft', dest='nofft', action='store_const',
					const=True, default=False,
					help='Use interpolation instead of Fourier shift')
## parser.add_argument('-nofit', dest='fit', action='store_const',
				   ## const=False, default=True,
				   ## help='Do not use PSF fitting values.')
parser.add_argument('-recentre', dest='recentre', action='store_const',
				   const=True, default=False,
				   help='Recentre the frames before binning.')
parser.add_argument('-s', dest='stat', action='store_const',
					const=True, default=False,
					help='Print benchmarking statistics')
parser.add_argument('-nici', dest='nici', action='store_const',
				   const=True, default=False,
				   help='Switch for GEMINI/NICI data')
parser.add_argument('-bottleneck', dest='use_bottleneck', action='store_const',
				   const=True, default=False,
				   help='Use bottleneck module instead of numpy for nanmedian.')

args = parser.parse_args()
d=args.d
pattern=args.pattern
## info_pattern=args.info_pattern
## info_dir=args.info_dir
log_file=args.log_file
nofft=args.nofft
## fit=args.fit
recentre=args.recentre
nici=args.nici
naxis3=args.naxis3
use_bottleneck=args.use_bottleneck

if use_bottleneck:
	from bottleneck import median as median
	from bottleneck import nanmedian as nanmedian
else:
	from numpy import nanmedian
	from numpy import median as median

header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
	'frame_num', 'frame_time', 'paralactic_angle']

target_dir = "."
## if collapse:
	## target_pattern="binc"
## else:
target_pattern="bin"


def read_recentre_cube(rcn, cube, rcube_list, l_max):
	t0_trans=MPI.Wtime()
	# Send the cube to be recentreed
	comm.bcast("recentre",root=0)
	comm.bcast(l_max,root=0)
	comm.bcast(rcube_list['info'][rcn],root=0)
	graphic_lib_320.send_frames_async(cube)
	cube=None
	if args.stat==True:
		print("\n STAT: Data upload took: "+humanize_time(MPI.Wtime()-t0_trans))
		t0_trans=MPI.Wtime()

	# Recover data from slaves
	for n in range(nprocs-1):
		data_in=comm.recv(source = n+1)
		if data_in == None:
			continue
		sys.stdout.write('\r\r\r Recentreed data from '+str(n+1)+' received									   =>')
		sys.stdout.flush()

		if cube==None:
			cube=data_in
		else:
			cube=np.concatenate((cube,data_in))

	return cube, t0_trans

t_init=MPI.Wtime()

if rank==0:
## if True:
	graphic_lib_320.print_init()

	hdr=None

	dirlist=graphic_lib_320.create_dirlist(pattern)

	## infolist=glob.glob(info_dir+os.sep+'*'+info_pattern+'*.rdb')
	## infolist.sort() # Sort the list alphabetically
##
##
	## cube_list,dirlist=graphic_lib_320.create_megatable(dirlist,infolist,keys=header_keys,nici=nici,fit=fit)

	skipped=0

		# 0: frame_number, 1: psf_barycentre_x, 2: psf_barycentre_y, 3: psf_pixel_size,
		# 4: psf_fit_centre_x, 5: psf_fit_centre_y, 6: psf_fit_height, 7: psf_fit_width_x, 8: psf_fit_width_y,
		# 9: frame_number, 10: frame_time, 11: paralactic_angle
	## l_max=0

	for c in range(len(dirlist)): # Loop over the cubes
		t0_cube=MPI.Wtime()
		## if nici: # Convert JD to unix time to get rid of day change issues
			## cube_list['info'][c][:,10]=(cube_list['info'][c][:,10]-2440587.5)*86400

		bin_filename=target_pattern+str(naxis3)+"_"+dirlist[c]
		## info_filename="all_info_"+bin_filename[:-5]+".rdb"

		# Check if already processed
		if os.access(target_dir+os.sep+bin_filename, os.F_OK | os.R_OK):
			print('Already processed: '+bin_filename)
			skipped=skipped+1
			continue
		# Check if already processed
		elif os.access(target_dir+os.sep+bin_filename+'.EMPTY', os.F_OK | os.R_OK):
			print('Already processed, but no cube created: '+bin_filename)
			skipped=skipped+1
			continue

		new_info=None
		cube,hdr=pyfits.getdata(dirlist[c],header=True)
		cube=cube*1.
		new_cube=np.ones((cube.shape[0]/naxis3, cube.shape[1],cube.shape[2]))*1.

		sys.stdout.write("\n Processing cube ["+str(c+1)+"/"+str(len(dirlist))+"]: "+str(dirlist[c])+"\n")
		sys.stdout.flush()

		## for i in range(0,new_cube.shape[0],new_cube.shape[0]/naxis3)	:
		for i,j in enumerate(range(0,cube.shape[0],naxis3)):
			if j+naxis3>cube.shape[0] or i+1==new_cube.shape[0]: # Taking all remaining frames
				print(j, j+naxis3, new_cube.shape[0], i, cube.shape[0])
				new_cube[i]=median(cube[j:],axis=0)
				break
			else:
				## print(i, i+naxis3, cube.shape[0])
				new_cube[i]=median(cube[j:j+naxis3],axis=0)



		hdr['HIERARCH GC BINNING']=(str(__version__)+'.'+(__subversion__), "")
		hdr['HIERARCH GC FRAMES/BIN']=(str(naxis3), "Number of franes in each bin")
		graphic_lib_320.save_fits(bin_filename, new_cube, target_dir=target_dir,  hdr=hdr, backend='pyfits')

		sys.stdout.write("\n Saved: {name} .\n Processed in {human_time} at {rate:.2f} MB/s \n"
						 .format(name=bin_filename, human_time=graphic_lib_320.humanize_time(MPI.Wtime()-t0_cube) ,
								 rate=os.path.getsize(bin_filename)/(1048576*(MPI.Wtime()-t0_cube))))
		sys.stdout.write("Remaining time: "+graphic_lib_320.humanize_time((MPI.Wtime()-t_init)*(len(dirlist)-c)/(c-skipped+1))+"\n")
		sys.stdout.flush()

		del cube


	print("\n Program finished, killing all the slaves...")
	print("Total time: "+graphic_lib_320.humanize_time((MPI.Wtime()-t_init)))
	comm.bcast("over", root=0)
	if skipped==len(dirlist):
		sys.exit(0)

	if not hdr==None:
		if 'ESO OBS TARG NAME' in hdr.keys():
			log_file=log_file+"_"+string.replace(hdr['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
		else:
			log_file=log_file+"_"+string.replace(hdr['OBJECT'],' ','')+"_"+str(__version__)+".log"
		graphic_lib_320.write_log((MPI.Wtime()-t_init),log_file)
	sys.exit(0)

#################################################################################
#
# SLAVES
#
# slaves need to:
# receive stack and frame
# recentre frames in stack
# calculate median
# subtract median from frame
# improvement could be done by somehow keeping recentreed frames
else: #Nothing to do...
	sys.exit(0)
	#nofft=comm.bcast(None,root=0)
	## todo=comm.bcast(None,root=0)
##
##
	## while not todo=="over":
		## if todo=="median":
##
			## # Receive number of first column
			## start_col=comm.recv(source=0)
			## # Receive stack to median
			## stack=comm.recv(source=0)
			## if d>5:
				## print("")
				## print(str(rank)+" stack.shape: "+str(stack.shape))
			## # Mask out the NaNs
			## psf=bottleneck.nanmedian(stack, axis=0)
			## del stack
			## comm.send(psf, dest=0)
			## del psf
##
		## elif todo=="recentre":
			## # Receive padding dimension
			## l_max=comm.bcast(None,root=0)
			## # Receive info table used for recentreing
			## info_stack=comm.bcast(None,root=0)
			## if d >2:
				## print(str(rank)+" received info_stack: "+str(info_stack))
##
			## # Receive number of first frame in cube
			## s=comm.recv(source=0)
			## if d >2:
				## print(str(rank)+" received first frame number: "+str(s))
##
			## # Receive cube
			## stack=comm.recv(source=0)
			## if d >2:
				## if stack==None:
					## print(str(rank)+" received stack: "+str(stack))
				## else:
					## print(str(rank)+" received stack, shape="+str(stack.shape))
##
##
			## if not stack==None:
				## bigstack=np.zeros((stack.shape[0],l_max*2,l_max*2))
				## bigstack[:,
				## l_max-stack.shape[1]/2:l_max+stack.shape[1]/2,
					## l_max-stack.shape[2]/2:l_max+stack.shape[2]/2]=stack
				## for fn in range(bigstack.shape[0]):
					## graphic_lib_320.dprint(d>2, "recentreing frame: "+str(fn)+" with shape: "+str(bigstack[fn].shape))
					## if info_stack[s+fn,4]==-1 or info_stack[s+fn,5]==-1:
						## bigstack[fn]=np.NaN
						## continue
					## # Shift is given by (image centre position)-(star position)
					## if nofft==True: # Use interpolation
						## bigstack[fn]=ndimage.interpolation.shift(bigstack[fn], (stack.shape[1]/2.-info_stack[s+fn,4], stack.shape[2]/2.-info_stack[s+fn,5]), order=3, mode='constant', cval=np.NaN, prefilter=False)
					## else: # Shift in Fourier space
						## bigstack[fn,np.ceil(l_max - info_stack[s+fn,4]):np.ceil(l_max - info_stack[s+fn,4]+stack.shape[1]),np.ceil(l_max - info_stack[s+fn,5])]=stack[fn,:,0]/2.
						## bigstack[fn,np.ceil(l_max - info_stack[s+fn,4]):np.ceil(l_max - info_stack[s+fn,4]+stack.shape[1]),np.floor(l_max - info_stack[s+fn,5]+stack.shape[2])]=stack[fn,:,-1]/2.
						## bigstack[fn,np.ceil(l_max - info_stack[s+fn,4]),np.ceil(l_max - info_stack[s+fn,5]):np.ceil(l_max - info_stack[s+fn,5]+stack.shape[2])]=stack[fn,0,:]/2.
						## bigstack[fn,np.floor(l_max - info_stack[s+fn,4]+stack.shape[1]),np.ceil(l_max - info_stack[s+fn,5]):np.ceil(l_max - info_stack[s+fn,5]+stack.shape[2])]=stack[fn,-1,:]/2.
						## bigstack[fn]=graphic_lib_320.fft_shift(bigstack[fn], stack.shape[1]/2.-info_stack[s+fn,4], stack.shape[2]/2.-info_stack[s+fn,5])
					## bigstack[fn,:np.ceil(l_max - info_stack[s+fn,4]),:]=np.NaN
					## bigstack[fn,np.floor(l_max - info_stack[s+fn,4]+stack.shape[1]):,:]=np.NaN
					## bigstack[fn,:,:np.ceil(l_max - info_stack[s+fn,5])]=np.NaN
					## bigstack[fn,:,np.floor(l_max - info_stack[s+fn,5]+stack.shape[2]):]=np.NaN
				## graphic_lib_320.dprint(d>2, "Sending back bigstack, shape="+str(bigstack.shape))
				## comm.send(bigstack, dest = 0)
				## del bigstack
			## else:
				## comm.send(None, dest = 0 )
##
			## del stack
##
		## else:
			## print(str(rank)+": received "+str(todo)+". Leaving....")
			## comm.send(None, dest = 0 )
##
		## todo=comm.bcast(None,root=0)
