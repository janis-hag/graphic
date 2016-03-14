#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.
First preparation step for ADI reduction.

Removes overscan. Finally save new images in reduction directory
"""

__version__='3.3'
__subversion__='0'

import numpy, scipy, glob, shutil, os, sys, fnmatch, time
from scipy.signal import correlate2d
from mpi4py import MPI
from string import split
import argparse
import numpy as np
import graphic_mpi_lib_330 as graphic_mpi_lib
import graphic_nompi_lib_330 as graphic_nompi_lib
from astropy.io import fits as pyfits

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser(description='Subtracts a combined PSF from each single frame.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0,help='Debug level, 0 is no debug')
parser.add_argument('--pattern', action="store", dest="pattern",  help='Filename pattern')
parser.add_argument('--target_dir', action="store", dest="target_dir",  default='.', help='Reduced files target directory')
parser.add_argument('--source_dir', action="store", dest="source_dir",  default='.', help='Directory containing files to reduce')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('-s', dest='stat', action='store_const',
				   const=True, default=False,
				   help='Print benchmarking statistics')

parser.add_argument('--trim', dest='trim', action='store_const',
				   const=True, default=False,
				   help='Trim images.')
parser.add_argument('--l_max', dest='l_max', action='store', type=int, default=None, help='The output image size (default is to not change the shape)')
parser.add_argument('--centre_offset', dest='centre_offset',action='store',type=int,default=0,nargs=2,
					help='Offset the image centre when using l_max, so that the image is cut centred on this position (in x,y)')
parser.add_argument('--offset', dest='offset', action='store_const',
				   const=True, default=False,
				   help='Add an offset to the pixel values. Usefull in case of negative images.')

args = parser.parse_args()
d=args.d
pattern=args.pattern
target_dir=args.target_dir
log_file=args.log_file
source_dir=args.source_dir
stat=args.stat
offset=args.offset
trim=args.trim
l_max=args.l_max
centre_offset=args.centre_offset

#target_dir = "RED"
#backup_dir = "prev"
file_prefix = "o_"

t_init=MPI.Wtime()

if rank==0:  # Master process
	print(sys.argv[0]+' started on '+ time.strftime("%c"))
	dirlist=graphic_nompi_lib.create_dirlist(source_dir+os.sep+pattern,target_dir=target_dir, target_pattern=file_prefix)

	if dirlist==None:
		print("No files found")
		for n in range(nprocs-1):
			comm.send("over",dest =n+1)
		sys.exit(1)

	for n in range(nprocs):
		start=int(n*np.floor(float(len(dirlist)/nprocs)))
		end=int((n+1)*np.floor(float(len(dirlist)/nprocs)))
		if n == nprocs-1:
			dirlist=dirlist[start:] # take the list to the end
			startframe=start
			break
		comm.send(dirlist[start:end], dest = n+1 )
		comm.send(start, dest=n+1)

	# Create directory to store reduced data
	if not os.path.isdir(target_dir):
		os.mkdir(target_dir)

if not rank==0:
	dirlist=comm.recv(source = 0)
	if dirlist=="over":
		sys.exit(1)

	startframe=int(comm.recv(source = 0))

t0=MPI.Wtime()
skipped=0

for i in range(len(dirlist)):
	targetfile=file_prefix+split(dirlist[i], os.sep)[-1]

	print(str(rank)+': ['+str(i+1)+"/"+str(len(dirlist))+"] "+dirlist[i]+" Remaining time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)*(len(dirlist)-i)/(i+1-skipped)))
	cube,header=pyfits.getdata(dirlist[i], header=True)

	cube_shape=cube.shape

	if trim:
		cube=cube[:,100:-100,100:-100]
	elif type(l_max) != type(None):
		if cube.ndim ==3:
			mindim=1
		elif cube.ndim==2:
			mindim=0
		if type(centre_offset) ==type(None):
			cenx=np.min(cube.shape[mindim:])/2
			ceny=np.min(cube.shape[mindim:])/2
		else:
			cenx=centre_offset[0]
			ceny=centre_offset[1]
		if cube.ndim==3:	
			cube=cube[:,cenx-l_max/2:cenx+l_max/2,ceny-l_max/2:ceny+l_max/2] 
		elif cube.ndim==2:	
			cube=cube[cenx-l_max/2:cenx+l_max/2,ceny-l_max/2:ceny+l_max/2] 

	# This used to only work if the difference in size was 2 pixels, but this is no longer the case for NACO. ACC edit Feb 2016
	elif cube.shape[1]>cube.shape[2]:
		overscan_limit=cube.shape[2]
		cube=cube[:,:overscan_limit,:]
	elif cube.shape[2]>cube.shape[1]:
		overscan_limit=cube.shape[1]
		cube=cube[:,:overscan_limit,:]
	else:
		print('Error! No overscan detected for: '+str(dirlist[i]))
		sys.exit(1)

	if offset:
		cube=cube-np.median(cube)
		header["HIERARCH GC RM_OVERSCAN_OFFSET"]=('T', "")
	else:
		header["HIERARCH GC RM_OVERSCAN_OFFSET"]=('T', "")

	#header.add_history('Overscan ['+str(overscan_limit)+','+str(cube_shape[1])+'] removed.')
	header["HIERARCH GC RM_OVERSCAN"]=(__version__+'.'+__subversion__, "")
	graphic_nompi_lib.save_fits(targetfile, cube, hdr=header, backend='pyfits' )

if 'ESO OBS TARG NAME' in header.keys():
	log_file=log_file+"_"+header['ESO OBS TARG NAME']+"_"+str(__version__)+".log"
else:
	log_file=log_file+"_"+str(__version__)+".log"

graphic_nompi_lib.write_log((MPI.Wtime()-t_init),log_file)
sys.exit(0)
