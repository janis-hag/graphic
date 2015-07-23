"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of the Parallel Angular Differential Imaging Pipeline (PADIP).
These are function common to different programs of the pipeline.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

import numpy, os, shutil, sys, glob, math
import numpy as np
## from scipy.signal import correlate2d
from mpi4py import MPI
from gaussfit_320 import fitgaussian
from scipy import ndimage, fftpack
#import astropy.io.fits as pyfits
from astropy.io import fits

## sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
## import bottleneck

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD


def bcast_window_coord(x0=None,y0=None,xs=None,xe=None,ys=None,ye=None,ref_window=None):
	'''
	Get broadcast the reference window limits and the reference window

	To send x0,y0,xs,xe,ys,ye,ref_window call bcast_window_coord(x0=x0,y0=y0,xs=xs,xe=xe,ys=ys,ye=ye,ref_window=ref_window)
	To receive x0,y0,xs,xe,ys,ye,ref_window call x0,y0,xs,xe,ys,ye,ref_window=bcast_window_coord()

	Returns x0,y0,xs,xe,ys,ye,ref_window
	'''

	# Broadcast the reference position, reference window
	x0 = comm.bcast(x0, root=0)
	y0 = comm.bcast(y0, root=0)
	# Broadcast the reference window limits and the reference window
	xs = comm.bcast(xs, root=0)
	xe = comm.bcast(xe, root=0)
	ys = comm.bcast(ys, root=0)
	ye = comm.bcast(ye, root=0)
	ref_window = comm.bcast(ref_window, root=0)

	return x0,y0,xs,xe,ys,ye,ref_window


## def calc_parang(hdr):
##	 """
##	 Read a header and calculates the paralactic angle
##	 """
##	 from numpy import sin, cos, tan, arctan, pi
##	 geolat_rad=float(hdr['HIERARCH ESO TEL GEOLAT'])*pi/180.
##	 dec_rad=float(hdr['DEC'])*pi/180.
##	 ra_seconds=float(hdr['RA'])*3600./15.
##	 ha_seconds=float(hdr['LST'])-ra_seconds
##	 ha_rad=(ha_seconds*15.*pi)/(3600*180)
##	 parang_deg=arctan((-sin(ha_rad)*cos(ha_rad))/sin(dec_rad)-cos(dec_rad)*tan(geolat_rad))*180/pi

##	 return parang_deg


def dprint(D, text):
	"""
	Prints debug text prepended by rank number

	"""
	if D:
		from mpi4py import MPI
		from sys import stdout
		stdout.flush()
		print(str(MPI.COMM_WORLD.Get_rank())+": "+text)




## def get_shift(rw, shift_im ,x_start, x_end, y_start, y_end,R):
	## """
	## Calculate the shift between the reference window (rw) and the shifted image (shift_im)
##
	## -xs,ys x position of the reference window start
	## -xe,ye y position of the reference window end
	## -R the window size
	## """
##
	## # Cut out the window
	## sw = shift_im[x_start:x_end,y_start:y_end]
	## # Set to zero saturated and background pixels
	## sw = numpy.where(sw>0.28,0,sw) #desaturate
	## sw = numpy.where(sw<0.02,0,sw) #set background to 0
	## rw = numpy.where(rw>0.28,0,rw) #desaturate
	## rw = numpy.where(rw<0.02,0,rw) #set background to 0
##
	## cor=correlate2d(rw,sw,mode='same')
	## mass=cor.sum()
	## xc=0
	## yc=0
	## for i in range(cor.shape[0]):
		## xc=xc+(i+1-cor.shape[0]/2.)*cor[i,:].sum()
	## for i in range(cor.shape[1]):
		## yc=yc+(i+1-cor.shape[1]/2.)*cor[:,i].sum()
##
## ##	 for i in range(cor.shape[0]):
## ##		 xc=xc+(i+1)*cor[i,:].sum()
## ##	 for i in range(cor.shape[1]):
## ##		 yc=yc+(i+1)*cor[:,i].sum()
## ##	 x_shift=xc/mass-cor.shape[0]/2.
## ##	 y_shift=yc/mass-cor.shape[1]/2.
##
		## x_shift=xc/mass
		## y_shift=yc/mass
##
	## return x_shift, y_shift


def read_rdb(file, h=0, comment=None):
	"""
	Reads an rdb file

	Input:
	 file: rdb_filename
	 h: line number of header
	 comment: comment char, for lines to be ignored

	Output: content of the file in form of a list
	"""

	import string, os
	# Check if file exists
	if not os.access(file, os.R_OK):
		return None

	f = open(file,'r');
	data = f.readlines()
	f.close()

	# take the second line to define the list keys.
	key = string.split(data[h][:-1],'\t')
	data_list = {}
	for i in range(len(key)): data_list[key[i]] = []

	for line in data[h+2:]:
		if not line[0]==comment or line[0:2]=='--':
			qq = string.split(line[:-1],'\t')
			for i in range(len(key)):
				try: value = float(qq[i])
				except ValueError: value = qq[i]
				data_list[key[i]].append(value)

	return data_list



def send_chunks(cub_in,d):
	"""
	Dispatches the _cubes_ to parallel processes.

	-cub_in the data_cube to send
	-d debug flag
	"""

	if cub_in.shape[1]/2.<(nprocs-1): # Send only two or three columns to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[1]:
				 comm.send(cub_in.shape[1], dest=n+1)
				 comm.send(None, dest=n+1)
			elif 2*n+3==cub.shape[1]: # Reaching the end of the cube, send all the remaining frames.
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[:,2*n:,:], dest = n+1 )
			else:
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[:,2*n:2*n+2,:], dest = n+1 )
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			if n+1 == nprocs-1:
				end=cub_in.shape[1]
			comm.send(start, dest = n+1 )
			comm.send(cub_in[:,start:end,:], dest = n+1)

			if d>3:
				print("Chunk "+str(n)+" sent, shape: "+str(cub_in[:,start:end,:].shape))
	del cub_in

def send_chunks_2D(cub_in,d):
	"""
	Dispatches the _frame_ to parallel processes.

	-cub_in the frame to send
	-d debug flag
	"""

	if cub_in.shape[1]/2.<(nprocs-1): # Send only two or three columns to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[1]:
				 comm.send(cube_in.shape[1], dest=n+1)
				 comm.send(None, dest=n+1)
			elif 2*n+3==cube.shape[1]: # Reaching the end of the cube, send all the remaining frames.
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[:,2*n:], dest = n+1 )
			else:
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[:,2*n:2*n+2], dest = n+1 )

			# check=comm.recv(source = n+1)
			# print("Chunk "+str(n)+" reception confirmed")
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			if n+1 == nprocs-1:
				end=cub_in.shape[1]
			comm.send(start, dest = n+1 )
			comm.send(cub_in[:,start:end], dest = n+1)

			if d>3:
				print("Chunk "+str(n)+" sent, shape: "+str(cub_in[:,start:end].shape))

	del cub_in


def send_dirlist(dirlist):
	"""
	Dispatches the dirlist to slaves, telling them what position in the dirlist they are starting at.
	It keeps the last part for the master.
	"""
	if len(dirlist) < nprocs:
		for n in range(nprocs):
			if n+1 == len(dirlist):
				start=n
				dirlist=dirlist[n:] # take the list to the end
				if n+1 < nprocs:
					for k in range(n+1,nprocs): # kill useless slaves
						comm.send(None, dest=k)
				break
			else:
				comm.send(dirlist[n:n+1], dest = n+1 )
				comm.send(n, dest=n+1)
	else:
		for n in range(nprocs):
			start=int(n*numpy.floor(float(len(dirlist))/nprocs))
			end=int((n+1)*numpy.floor(float(len(dirlist))/nprocs))
			#if end >= cub_in.shape[0]:
			if n == nprocs-1:
				dirlist=dirlist[start:] # take the list to the end
				break
			comm.send(dirlist[start:end], dest = n+1 )
			comm.send(start, dest=n+1)

	return start,dirlist

def send_dirlist_slaves(dirlist):
	"""
	Dispatches the dirlist to slaves, telling them what position in the dirlist they are starting at.

	TODO: change behaviour in if len(dirlist) < nprocs-1. Instead of giving last proc more files, send last proc less files (with min=2).
	"""
	## if len(dirlist) < nprocs-1:
		## for n in range(nprocs-1):
			## if n+1 == len(dirlist):
				## # take the list to the end
				## start=n
				## comm.send(dirlist[n:], dest = n+1 )
				## comm.send(n, dest=n+1)
				## if n+1 < nprocs:
					## for k in range(n+1,nprocs-1): # kill useless slaves
						## comm.send(None, dest=k+1)
				## break
			## else:
				## comm.send(dirlist[n:n+1], dest = n+1 )
				## comm.send(n, dest=n+1)
	if len(dirlist) < nprocs-1:
		for n in range(nprocs-1):
			if n+1 == len(dirlist):
				# take the list to the end
				start=n
				comm.send(dirlist[n:], dest = n+1 )
				comm.send(n, dest=n+1)
				if n+1 < nprocs:
					for k in range(n+1,nprocs-1): # kill useless slaves
						comm.send(None, dest=k+1)
				break
			else:
				comm.send(dirlist[n:n+1], dest = n+1 )
				comm.send(n, dest=n+1)
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.ceil(float(len(dirlist))/nprocs))
			end=int((n+1)*numpy.ceil(float(len(dirlist))/nprocs))
			#if end >= cub_in.shape[0]:
			if n+1 == nprocs-1:
				comm.send(dirlist[start:], dest = n+1 )
				## dirlist=dirlist[start:] # take the list to the end
				## break
			else:
				comm.send(dirlist[start:end], dest = n+1 )
			comm.send(start, dest=n+1)

	# return start,dirlist

def send_dirlist_slaves_prev(dirlist):
	"""
	Dispatches the dirlist to slaves, telling them what position in the dirlist they are starting at.

	Previous version. Well tested but not optimised.
	"""
	if len(dirlist) < nprocs-1:
		for n in range(nprocs-1):
			if n+1 == len(dirlist):
				# take the list to the end
				start=n
				comm.send(dirlist[n:], dest = n+1 )
				comm.send(n, dest=n+1)
				if n+1 < nprocs:
					for k in range(n+1,nprocs-1): # kill useless slaves
						comm.send(None, dest=k+1)
				break
			else:
				comm.send(dirlist[n:n+1], dest = n+1 )
				comm.send(n, dest=n+1)
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(len(dirlist))/nprocs))
			end=int((n+1)*numpy.floor(float(len(dirlist))/nprocs))
			#if end >= cub_in.shape[0]:
			if n+1 == nprocs-1:
				comm.send(dirlist[start:], dest = n+1 )
				## dirlist=dirlist[start:] # take the list to the end
				## break
			else:
				comm.send(dirlist[start:end], dest = n+1 )
			comm.send(start, dest=n+1)


def send_frames(cub_in):
	"""
	Dispatches the cubes to parallel processes, dividing the cube in cubes with less frames.

	If the number of process is higher than the number of frames, send only one frame to each process
	and None to the remainig ones.
	-cub_in the data_cube to send
	"""

	if cub_in.shape[0]/2.<(nprocs-1): # Send only two or three frames to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[0]:
				comm.send(cub_in.shape[0], dest=n+1)
				comm.send(None, dest=n+1)
			elif 2*n+3==cub_in.shape[0]: # Reaching the end of the cube, send all the remaining frames.
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[2*n:,:,:], dest = n+1 )
			else:
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[2*n:2*n+2,:,:], dest = n+1 )
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[0])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[0])/(nprocs-1)))
			#if end >= cub_in.shape[0]:
			if n+1 == nprocs-1:
				end=cub_in.shape[0]
			comm.send(start, dest = n+1 )
			comm.send(cub_in[start:end,:,:], dest = n+1 )

	del cub_in

def send_frames_async(cub_in):
	"""
	Dispatches the cubes to parallel processes, dividing the cube in cubes with less frames.

	If the number of process is higher than the number of frames, send only one frame to each process
	and None to the remainig ones.
	-cub_in the data_cube to send
	"""

	if cub_in.shape[0]/2.<(nprocs-1): # Send only two or three frames to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[0]:
				r1=comm.isend(cub_in.shape[0], dest=n+1)
				r2=comm.isend(None, dest=n+1)
			elif 2*n+3==cub_in.shape[0]: # Reaching the end of the cube, send all the remaining frames.
				r1=comm.isend(2*n, dest = n+1 )
				r2=comm.isend(cub_in[2*n:,:,:], dest = n+1 )
			else:
				r1=comm.isend(2*n, dest = n+1 )
				r2=comm.isend(cub_in[2*n:2*n+2,:,:], dest = n+1 )
			if n==0:
				request=[r1,r2]
			else:
				request.append(r1)
				request.append(r2)
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[0])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[0])/(nprocs-1)))
			#if end >= cub_in.shape[0]:
			if n+1 == nprocs-1:
				end=cub_in.shape[0]
			r1=comm.isend(start, dest = n+1 )
			r2=comm.isend(cub_in[start:end,:,:], dest = n+1 )
			if n==0:
				request=[r1,r2]
			else:
				request.append(r1)
				request.append(r2)
	# Wait for all transfers to be finished
	MPI.Request.Waitall(request)
	del cub_in

def send_masked_chunks(cub_in,d):
	"""
	Dispatches the cubes to parallel processes.

	-cub_in the data_cube to send
	-d debug flag
	"""

	if cub_in.shape[1]/2.<(nprocs-1): # Send only two or three columns to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[1]:
				 comm.send(cube_in.shape[1], dest=n+1)
				 comm.send(None, dest=n+1)
				 comm.send(None, dest=n+1)

			elif 2*n+3==cube.shape[1]: # Reaching the end of the cube, send all the remaining frames.
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in.data[:,2*n:,:], dest = n+1 )
				comm.send(cub_in.mask[:,2*n:,:], dest = n+1 )
			else:
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in.data[:,2*n:2*n+2,:], dest = n+1 )
				comm.send(cub_in.mask[:,2*n:2*n+2,:], dest = n+1 )

			# check=comm.recv(source = n+1)
			# print("Chunk "+str(n)+" reception confirmed")
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			if n+1 == nprocs-1:
				end=cub_in.shape[1]
			comm.send(start, dest = n+1 )
			print(str(rank)+" start: "+str(start))
			print(str(rank)+" end: "+str(end))
			print(str(rank)+" cub_in: "+str(cub_in))
			print(str(rank)+" cub_in.mask: "+str(cub_in.mask))

			comm.send(cub_in.data[:,start:end,:], dest = n+1)
			comm.send(cub_in.mask[:,start:end,:], dest = n+1)

			if d>1:
				print("Chunk "+str(n)+" sent, shape: "+str(cub_in[:,start:end,:].shape))

	del cub_in





