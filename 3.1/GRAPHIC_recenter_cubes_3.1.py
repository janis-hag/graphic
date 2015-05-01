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
import graphic_lib_310
from scipy import ndimage
from mpi4py import MPI
import argparse
from graphic_lib_310 import dprint
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
parser.add_argument('--info_pattern', action="store", dest="info_pattern", required=True, help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('--naxis3', action="store", dest="naxis3", default=4, type=int, help='The number of frames per cube')
parser.add_argument('-nofft', dest='nofft', action='store_const',
					const=True, default=False,
					help='Use interpolation instead of Fourier shift')
parser.add_argument('-nofit', dest='fit', action='store_const',
				   const=False, default=True,
				   help='Do not use PSF fitting values.')
parser.add_argument('-nocollapse', dest='collapse', action='store_const',
				   const=False, default=True,
				   help='Do not collapse cubes.')
parser.add_argument('-s', dest='stat', action='store_const',
					const=True, default=False,
					help='Print benchmarking statistics')
parser.add_argument('-nici', dest='nici', action='store_const',
				   const=True, default=False,
				   help='Switch for GEMINI/NICI data')


args = parser.parse_args()
d=args.d
pattern=args.pattern
info_pattern=args.info_pattern
info_dir=args.info_dir
log_file=args.log_file
nofft=args.nofft
fit=args.fit
collapse=args.collapse
nici=args.nici
naxis3=args.naxis3

header_keys=['frame_number', 'psf_barycenter_x', 'psf_barycenter_y', 'psf_pixel_size', 'psf_fit_center_x', 'psf_fit_center_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
	'frame_num', 'frame_time', 'paralactic_angle']

target_dir = "."
if collapse:
	target_pattern="rc"
else:
	target_pattern="r"


def read_recenter_cube(rcn, cube, rcube_list, l_max):
	t0_trans=MPI.Wtime()
	# Send the cube to be recentered
	comm.bcast("recenter",root=0)
	comm.bcast(l_max,root=0)
	comm.bcast(rcube_list['info'][rcn],root=0)
	graphic_lib_310.send_frames_async(cube)
	cube=None
	if args.stat==True:
		print("\n STAT: Data upload took: "+humanize_time(MPI.Wtime()-t0_trans))
		t0_trans=MPI.Wtime()

	# Recover data from slaves
	for n in range(nprocs-1):
		data_in=comm.recv(source = n+1)
		if data_in == None:
			continue
		sys.stdout.write('\r\r\r Recentered data from '+str(n+1)+' received									   =>')
		sys.stdout.flush()

		if cube==None:
			cube=data_in
		else:
			cube=np.concatenate((cube,data_in))

	return cube, t0_trans

t_init=MPI.Wtime()

if rank==0:
	print(sys.argv[0]+' started on '+ time.strftime("%c"))
	hdr=None

	dirlist=graphic_lib_310.create_dirlist(pattern)

	infolist=glob.glob(info_dir+os.sep+'*'+info_pattern+'*.rdb')
	infolist.sort() # Sort the list alphabetically


	cube_list,dirlist=graphic_lib_310.create_megatable(dirlist,infolist,keys=header_keys,nici=nici,fit=fit)

	skipped=0

		# 0: frame_number, 1: psf_barycenter_x, 2: psf_barycenter_y, 3: psf_pixel_size,
		# 4: psf_fit_center_x, 5: psf_fit_center_y, 6: psf_fit_height, 7: psf_fit_width_x, 8: psf_fit_width_y,
		# 9: frame_number, 10: frame_time, 11: paralactic_angle
	l_max=0

	for c in range(0, len(cube_list['cube_filename']), naxis3): # Loop over the cubes
		t0_cube=MPI.Wtime()
		if nici: # Convert JD to unix time to get rid of day change issues
			cube_list['info'][c][:,10]=(cube_list['info'][c][:,10]-2440587.5)*86400

		psf_sub_filename=target_pattern+"_"+cube_list['cube_filename'][c]
		info_filename="all_info_"+psf_sub_filename[:-5]+".rdb"

		# Check if already processed
		if os.access(target_dir+os.sep+psf_sub_filename, os.F_OK | os.R_OK):
			print('Already processed: '+psf_sub_filename)
			skipped=skipped+1
			continue
		# Check if already processed
		elif os.access(target_dir+os.sep+psf_sub_filename+'.EMPTY', os.F_OK | os.R_OK):
			print('Already processed, but no cube created: '+psf_sub_filename)
			skipped=skipped+1
			continue

		new_cube=None
		new_info=None
		for n in range(naxis3):
			if c+n==len(cube_list['cube_filename']):
				break
			sys.stdout.write("\n Processing cube ["+str(c+n+1)+"/"+str(len(cube_list['cube_filename']))+"]: "+str(cube_list['cube_filename'][c])+"\n")
			sys.stdout.flush()
			cube,hdr=pyfits.getdata(cube_list['cube_filename'][c+n],header=True)
			## if np.max(cube_list['info'][c+n][:][11])==0: # In order to work in field tracking mode too.
				## p0=-1
			## else:
			p0=float(cube_list['info'][c+n][1][11])
			if l_max==0:
				for i in range(len(cube_list['info'])):
					if not cube_list['info'][i][len(cube_list['info'][i])/2,4]==-1:
						l=graphic_lib_310.get_max_dim(cube_list['info'][i][len(cube_list['info'][i])/2,4], cube_list['info'][i][len(cube_list['info'][i])/2,5], int(hdr['NAXIS1']), cube_list['info'][i][:,11]-p0)
						if l>l_max: l_max=l

				graphic_lib_310.dprint(d>1, 'l_max: '+str(l_max))
				l_max=np.floor(l_max)
			if l_max==0:
				l_max=2*int(hdr['NAXIS1'])
			cube, t0_trans=read_recenter_cube(c+n, cube, cube_list, l_max)
			if collapse:
				if new_cube==None:
					new_cube=np.ones((naxis3,cube.shape[1],cube.shape[2]))
					new_info=np.NaN*np.ones((naxis3,len(cube_list['info'][c+n][0])))
				new_cube[n]=bottleneck.nanmedian(cube, axis=0)
				new_info[n]=bottleneck.nanmedian(np.where(cube_list['info'][c+n]==-1,np.NaN,cube_list['info'][c+n]), axis=0)
				new_info[n]=np.where(np.isnan(new_info[n]),-1,new_info[n])
			else:
				psf_sub_filename=target_pattern+"_"+cube_list['cube_filename'][c+n]
				hdr.set("HIERARCH GC RECENTER",str(__version__)+'.'+(__subversion__), "")
				hdr.set("HIERARCH GC LMAX",l_max,"")
				hdr.set('CRPIX1','{0:14.7G}'.format(cube.shape[1]/2.+hdr['CRPIX1']-cube_list['info'][c+n][hdr['NAXIS3']/2,4]), "")
				hdr.set('CRPIX2','{0:14.7G}'.format(cube.shape[2]/2.+hdr['CRPIX2']-cube_list['info'][c+n][hdr['NAXIS3']/2,5]), "")

				hdr.add_history("Updated CRPIX1, CRPIX2")
				graphic_lib_310.save_fits(psf_sub_filename, cube, target_dir=target_dir,  hdr=hdr, backend='pyfits')
				graphic_lib_310.write_array2rdb(info_dir+os.sep+info_filename,cube_list['info'][c+n],header_keys)

		if collapse:
			hdr.set("HIERARCH GC RECENTER",str(__version__)+'.'+(__subversion__), "")
			hdr.set("HIERARCH GC LMAX",l_max,"")
			hdr.set('CRPIX1','{0:14.7G}'.format(cube.shape[1]/2.+hdr['CRPIX1']-cube_list['info'][c][hdr['NAXIS3']/2,4]), "")
			hdr.set('CRPIX2','{0:14.7G}'.format(cube.shape[2]/2.+hdr['CRPIX2']-cube_list['info'][c][hdr['NAXIS3']/2,5]), "")

			hdr.add_history("Updated CRPIX1, CRPIX2")
			graphic_lib_310.save_fits(psf_sub_filename, new_cube, target_dir=target_dir,  hdr=hdr, backend='pyfits')
			graphic_lib_310.write_array2rdb(info_dir+os.sep+info_filename,new_info,header_keys)

		sys.stdout.write("\n Saved: {name} .\n Processed in {human_time} at {rate:.2f} MB/s \n"
						 .format(name=psf_sub_filename, human_time=graphic_lib_310.humanize_time(MPI.Wtime()-t0_cube) ,
								 rate=os.path.getsize(psf_sub_filename)/(1048576*(MPI.Wtime()-t0_cube))))
		sys.stdout.write("Remaining time: "+graphic_lib_310.humanize_time((MPI.Wtime()-t_init)*(len(cube_list['cube_filename'])-c)/(c-skipped+1))+"\n")
		sys.stdout.flush()

		del cube


	print("\n Program finished, killing all the slaves...")
	print("Total time: "+graphic_lib_310.humanize_time((MPI.Wtime()-t_init)))
	comm.bcast("over", root=0)
	if skipped==len(cube_list['cube_filename']):
		sys.exit(0)

	if not hdr==None:
		if 'ESO OBS TARG NAME' in hdr.keys():
			log_file=log_file+"_"+string.replace(hdr['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
		else:
			log_file=log_file+"_"+string.replace(hdr['OBJECT'],' ','')+"_"+str(__version__)+".log"
		graphic_lib_310.write_log((MPI.Wtime()-t_init),log_file)
	sys.exit(0)

#################################################################################
#
# SLAVES
#
# slaves need to:
# receive stack and frame
# recenter frames in stack
# calculate median
# subtract median from frame
# improvement could be done by somehow keeping recentered frames
else:
	## nofft=comm.bcast(None,root=0)
	todo=comm.bcast(None,root=0)


	while not todo=="over":
		if todo=="median":

			# Receive number of first column
			start_col=comm.recv(source=0)
			# Receive stack to median
			stack=comm.recv(source=0)
			if d>5:
				print("")
				print(str(rank)+" stack.shape: "+str(stack.shape))
			# Mask out the NaNs
			psf=bottleneck.nanmedian(stack, axis=0)
			del stack
			comm.send(psf, dest=0)
			del psf

		elif todo=="recenter":
			# Receive padding dimension
			l_max=comm.bcast(None,root=0)
			# Receive info table used for recentering
			info_stack=comm.bcast(None,root=0)
			if d >2:
				print(str(rank)+" received info_stack: "+str(info_stack))

			# Receive number of first frame in cube
			s=comm.recv(source=0)
			if d >2:
				print(str(rank)+" received first frame number: "+str(s))

			# Receive cube
			stack=comm.recv(source=0)
			if d >2:
				if stack==None:
					print(str(rank)+" received stack: "+str(stack))
				else:
					print(str(rank)+" received stack, shape="+str(stack.shape))


			if not stack==None:
				bigstack=np.zeros((stack.shape[0],l_max*2,l_max*2))
				bigstack[:,
				l_max-stack.shape[1]/2:l_max+stack.shape[1]/2,
					l_max-stack.shape[2]/2:l_max+stack.shape[2]/2]=stack
				for fn in range(bigstack.shape[0]):
					graphic_lib_310.dprint(d>2, "recentering frame: "+str(fn)+" with shape: "+str(bigstack[fn].shape))
					if info_stack[s+fn,4]==-1 or info_stack[s+fn,5]==-1:
						bigstack[fn]=np.NaN
						continue
					# Shift is given by (image center position)-(star position)
					if nofft==True: # Use interpolation
						bigstack[fn]=ndimage.interpolation.shift(bigstack[fn], (stack.shape[1]/2.-info_stack[s+fn,4], stack.shape[2]/2.-info_stack[s+fn,5]), order=3, mode='constant', cval=np.NaN, prefilter=False)
					else: # Shift in Fourier space
						bigstack[fn,np.ceil(l_max - info_stack[s+fn,4]):np.ceil(l_max - info_stack[s+fn,4]+stack.shape[1]),np.ceil(l_max - info_stack[s+fn,5])]=stack[fn,:,0]/2.
						bigstack[fn,np.ceil(l_max - info_stack[s+fn,4]):np.ceil(l_max - info_stack[s+fn,4]+stack.shape[1]),np.floor(l_max - info_stack[s+fn,5]+stack.shape[2])]=stack[fn,:,-1]/2.
						bigstack[fn,np.ceil(l_max - info_stack[s+fn,4]),np.ceil(l_max - info_stack[s+fn,5]):np.ceil(l_max - info_stack[s+fn,5]+stack.shape[2])]=stack[fn,0,:]/2.
						bigstack[fn,np.floor(l_max - info_stack[s+fn,4]+stack.shape[1]),np.ceil(l_max - info_stack[s+fn,5]):np.ceil(l_max - info_stack[s+fn,5]+stack.shape[2])]=stack[fn,-1,:]/2.
						bigstack[fn]=graphic_lib_310.fft_shift(bigstack[fn], stack.shape[1]/2.-info_stack[s+fn,4], stack.shape[2]/2.-info_stack[s+fn,5])
					bigstack[fn,:np.ceil(l_max - info_stack[s+fn,4]),:]=np.NaN
					bigstack[fn,np.floor(l_max - info_stack[s+fn,4]+stack.shape[1]):,:]=np.NaN
					bigstack[fn,:,:np.ceil(l_max - info_stack[s+fn,5])]=np.NaN
					bigstack[fn,:,np.floor(l_max - info_stack[s+fn,5]+stack.shape[2]):]=np.NaN
				graphic_lib_310.dprint(d>2, "Sending back bigstack, shape="+str(bigstack.shape))
				comm.send(bigstack, dest = 0)
				del bigstack
			else:
				comm.send(None, dest = 0 )

			del stack

		else:
			print(str(rank)+": received "+str(todo)+". Leaving....")
			comm.send(None, dest = 0 )

		todo=comm.bcast(None,root=0)
