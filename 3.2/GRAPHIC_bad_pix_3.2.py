#!/usr/bin/python
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

Its purpose is to clean bad pixels from each frame in a cube list.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__='3.2'
__subversion__='0'

import numpy, scipy, glob, shutil, os, sys, string
import numpy as np
import graphic_lib_320
from mpi4py import MPI
import argparse
from graphic_lib_320 import dprint
import astropy.io.fits as pyfits

#sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
## import bottleneck

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

target_dir = "."
target_pattern ="cl_"
coefficient= 1

parser = argparse.ArgumentParser(description='Puts bad pixels to median value, based on darks.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern",  default='*', help='Filename pattern')
parser.add_argument('--dark_pattern', action="store", dest="dark_pattern", required=True, help='Darks filename pattern')
parser.add_argument('--dark_dir', action="store", dest="dark_dir", required=True, help='Directory containing the darks')
parser.add_argument('--coef', action="store", dest="coef", type=float, default=5, help='The sigma threshold for pixels to be rejected')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('-s', dest='stat', action='store_const',
				   const=True, default=False,
				   help='Print benchmarking statistics')
parser.add_argument('-interactive', dest='interactive', action='store_const',
				   const=True, default=False,
				   help='Switch to set execution to interactive mode')
parser.add_argument('-bottleneck', dest='use_bottleneck', action='store_const',
				   const=True, default=False,
				   help='Use bottleneck module instead of numpy for nanmedian.')

args = parser.parse_args()
d=args.d
pattern=args.pattern
dark_pattern=args.dark_pattern
dark_dir=args.dark_dir
coef=args.coef
log_file=args.log_file
use_bottleneck=args.use_bottleneck

if use_bottleneck:
	from bottleneck import median as median
	from bottleneck import nanmedian as nanmedian
else:
	from numpy import nanmedian
	from numpy import median as median


comments=[]

def gen_badpix(sky ,coef, comments):
	""" Create badpixel map by searching for pixels

	Input:
	-sky: an array containing a sky
	-sky_head: FITS header of the sky
	"""
	global median

	sigma = sky.std()
	med = median(sky)
	print("Sigma: "+str(sigma)+", median: "+str(med))

	#Creates a tuple with the x-y positions of the dead pixels
	deadpix=np.where(sky < med-sigma*coef )

	#Creates a tuple with the x-y positions of the dead pixels
	hotpix=np.where(sky > med+sigma*coef )

	#negativepix=np.where(sky < 0 )

	# Warning! x and y inverted with respect to ds9
	dprint(d>2, "sky.shape: "+str(sky.shape)+", sky.size: "+str(sky.size))
	# if np.shape(deadpix)[1]+np.shape(negativepix)[1]+np.shape(hotpix)[1]==0:
	if np.shape(deadpix)[1]+np.shape(hotpix)[1]==0:
		c="No bad pixels found. Aborting"
		dprint(d>2, 'No bad pixels found. Aborting!')
		## print(c)
		comments.append(c)
		for n in range(nprocs-1):
			comm.send(None,dest =n+1)
		sys.exit(1)
	else:
		c="Found "+str(np.shape(deadpix)[1])+" = "+str(100.*np.shape(deadpix)[1]/sky.size)+"% dead, "+\
		"and "+str(np.shape(hotpix)[1])+" = "+str(100.*np.shape(hotpix)[1]/sky.size)+"% hot pixels."
		## print(c)
		comments.append(c)


	badpix=tuple(np.append(np.array(deadpix),np.array(hotpix), axis=1))
	## badpix=tuple(np.append(np.array(badpix),np.array(negativepix), axis=1))

	return badpix, comments
	#	return deadpix, hotpix, negativepix


def clean_bp(badpix, cub_in):
	global nanmedian
	#(deadpix, hotpix, negativepix, cub_in):
	cub_in[:,badpix[0],badpix[1]]=np.NaN
	for f in range(cub_in.shape[0]):
		if args.interactive:
			sys.stdout.write('\r Frame '+str(f+1)+' of '+str(cub_in.shape[0]))
			sys.stdout.flush()

		for j in range(len(badpix[0])):
			y=badpix[0][j]
			x=badpix[1][j]
			## print(j,y,x)
			if y == cube.shape[1]-1: # At the image edge !!!
				if x == cube.shape[2]-1: # In a corner
					cub_in[f,y,x]=nanmedian([cub_in[f,y-1,x-1],cub_in[f,y-1,x],
												cub_in[f,y,x-1]])
				elif x == 0:
					cub_in[f,y,x]=nanmedian([cub_in[f,y-1,x],cub_in[f,y-1,x+1],
												cub_in[f,y,x+1]])
				else: # Along the edge
					cub_in[f,y,x]=nanmedian([cub_in[f,y-1,x-1],cub_in[f,y-1,x],cub_in[f,y-1,x+1],
												cub_in[f,y,x-1],cub_in[f,y,x+1]])
			elif y == 0: # At the image edge !!!
				if x == cube.shape[2]-1: # In a corner
					cub_in[f,y,x]=nanmedian([cub_in[f,y,x-1],
												cub_in[f,y+1,x-1],cub_in[f,y+1,x]])
				elif x == 0:  # In a corner
					cub_in[f,y,x]=nanmedian([cub_in[f,y,x+1],
												cub_in[f,y+1,x],cub_in[f,y+1,x+1]])
				else: # Along the edge
					cub_in[f,y,x]=nanmedian([cub_in[f,y,x-1],cub_in[f,y,x+1],
												cub_in[f,y+1,x-1],cub_in[f,y+1,x],cub_in[f,y+1,x+1]])
			elif x == 0: # Along the edge
				cub_in[f,y,x]=nanmedian([cub_in[f,y-1,x],
												cub_in[f,y-1,x+1],cub_in[f,y,x+1],
												cub_in[f,y+1,x],
												cub_in[f,y+1,x+1]])
			elif x == cub_in.shape[1]-1: # Along the edge
				cub_in[f,y,x]=nanmedian([cub_in[f,y-1,x-1],cub_in[f,y-1,x],
										  cub_in[f,y,x-1],
										  cub_in[f,y+1,x-1],cub_in[f,y+1,x]])
			else: # Usual case, not on an edge
				cub_in[f,y,x]=nanmedian([cub_in[f,y-1,x-1],cub_in[f,y-1,x],cub_in[f,y-1,x+1],
											cub_in[f,y,x-1],cub_in[f,y,x+1],
											cub_in[f,y+1,x-1],cub_in[f,y+1,x],cub_in[f,y+1,x+1]])
	return cub_in

t_init=MPI.Wtime()
if rank==0:
	graphic_lib_320.print_init()

	t_init=MPI.Wtime()

	print("Searching cubes...")
	dirlist=graphic_lib_320.create_dirlist(pattern, target_pattern=target_pattern)
	print("Searching reference cubes...")
	darklist=graphic_lib_320.create_dirlist(dark_dir+os.sep+dark_pattern)

	if dirlist is None or darklist is None:
		print('Missing files, leaving...')
		MPI.Finalize()
		sys.exit(1)

	start,dirlist=graphic_lib_320.send_dirlist(dirlist)
	dprint(d>2, 'Dirlist sent to slaves')

	dark_cube=None
	for file_name in darklist:
		## dark_hdulist = fits.open(file_name)
		## data=dark_hdulist[0].data
		data=pyfits.getdata(file_name, header=False)
		if dark_cube is None:
			dark_cube=data #[np.newaxis,...]
			## print(dark_cube.shape)
		elif len(data.shape)==3: # CUBE
			## print(data.shape)
			## dark_cube=np.concatenate((dark_cube,data[np.newaxis,...]),axis=0)
			dark_cube=np.concatenate((dark_cube,data),axis=0)
			## print(dark_cube.shape)
		elif len(data.shape)==2: # FRAME
			if len(dark_cube.shape)==3:
				dark_cube=np.rollaxis(dark_cube,0,3)
			## print(data.shape, dark_cube.shape)
			dark_cube=np.rollaxis(np.dstack((dark_cube,data)), 2)

	if len(np.where(np.isnan(dark_cube))[0]):
		print("Found NaNs: "+str(np.where(np.isnan(dark_cube))))
		dark_cube=np.where(np.isnan(dark_cube),0,dark_cube)

	dprint(d>2, "dark_cube.shape "+str(dark_cube.shape))
	dark_cube=dark_cube*1.
	if len(dark_cube.shape)==3:
		dark=median(dark_cube, axis=0)
	elif len(dark_cube.shape)==2:
		dark=dark_cube
	del dark_cube
	bad_pix,comments = gen_badpix(dark, coefficient, comments)
	comm.bcast(bad_pix, root=0)

if not rank==0:
	dirlist=comm.recv(source = 0)
	if dirlist is None:
		print('Received None dirlist. Leaving...')
		bad_pix=comm.bcast(None, root=0)
		sys.exit(1)

	start=int(comm.recv(source = 0))
	bad_pix=comm.bcast(None, root=0)
	dprint(d>2, 'Received dirlist, start, and bad_pix')


t0=MPI.Wtime()

for i in range(len(dirlist)):
	print(str(rank)+': ['+str(start+i)	+'/'+str(len(dirlist)+start-1)+"] "+dirlist[i]+" Remaining time: "+graphic_lib_320.humanize_time((MPI.Wtime()-t0)*(len(dirlist)-i)/(i+1)))

	# Read cube header and data
	## hdulist = fits.open(dirlist[i],memmap=True)
	## header=hdulist[0].header
	## cube=hdulist[0].data
	cube,header=pyfits.getdata(dirlist[i], header=True)
	cube=clean_bp(bad_pix, cube)

	header["HIERARCH GC BAD_PIX"]=(__version__+'.'+__subversion__, "")
	## graphic_lib_320.save_fits( target_pattern+dirlist[i],  target_dir, cube, header )
	graphic_lib_320.save_fits(target_pattern+dirlist[i], cube, header=header,backend='pyfits')
	## graphic_lib_320.save_fits(target_pattern+dirlist[i], hdulist, backend='astropy', verify='fix')

if 'ESO OBS TARG NAME' in header.keys():
	log_file=log_file+"_"+string.replace(header['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
else:
	log_file=log_file+"_"+string.replace(header['OBJECT']+"_"+str(__version__),' ','')+".log"

print(str(rank)+": Total time: "+graphic_lib_320.humanize_time((MPI.Wtime()-t0)))
graphic_lib_320.write_log((MPI.Wtime()-t_init), log_file, comments)
sys.exit(0)
