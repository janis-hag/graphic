#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline for
High-contrast Imaging of planetary Companions".

Creates cubes with by combining single frame files

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__='3.3'
__subversion__='0'

import scipy, glob,  os, sys, subprocess, string, time
import numpy as np
import graphic_nompi_lib_330 as graphic_nompi_lib
from scipy import ndimage
## from mpi4py import MPI
import argparse
## from graphic_nompi_lib import dprint
import astropy.io.fits as pyfits

## sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
## import bottleneck


## nprocs = MPI.COMM_WORLD.Get_size()
## rank   = MPI.COMM_WORLD.Get_rank()
## procnm = MPI.Get_processor_name()
## comm = MPI.COMM_WORLD


parser = argparse.ArgumentParser(description='Creates cubes with by combining single frame files.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern",  default='*', help='Filename pattern')
## parser.add_argument('--info_pattern', action="store", dest="info_pattern", required=True, help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('--naxis2', action="store", dest="naxis2", default=0, type=int, help='The size of the of frames. No frame trimming if not set.')
parser.add_argument('--naxis3', action="store", dest="naxis3", default=4, type=int, help='The number of frames per cube')
parser.add_argument('-nofft', dest='nofft', action='store_const',
					const=True, default=False,
					help='Use interpolation instead of Fourier shift')
parser.add_argument('-nofit', dest='fit', action='store_const',
				   const=False, default=True,
				   help='Do not use PSF fitting values.')
parser.add_argument('-collapse', dest='collapse', action='store_const',
				   const=True, default=False,
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
## info_pattern=args.info_pattern
info_dir=args.info_dir
log_file=args.log_file
nofft=args.nofft
fit=args.fit
collapse=args.collapse
nici=args.nici
naxis2=args.naxis2
naxis3=args.naxis3

header_keys=['frame_number', 'psf_barycenter_x', 'psf_barycenter_y', 'psf_pixel_size', 'psf_fit_center_x', 'psf_fit_center_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
	'frame_num', 'frame_time', 'paralactic_angle']

target_dir = "."
if collapse:
	target_pattern="tpc"
else:
	target_pattern="tp"


t_init=time.time()

## if rank==0:
graphic_nompi_lib.print_init()

hdr=None

dirlist=graphic_nompi_lib.create_dirlist(pattern)
print(dirlist)
## infolist=glob.glob(info_dir+os.sep+'*'+info_pattern+'*.rdb')
## infolist.sort() # Sort the list alphabetically
## cube_list,dirlist=graphic_lib_310.create_megatable(dirlist,infolist,keys=header_keys,nici=nici,fit=fit)

skipped=0

	# 0: frame_number, 1: psf_barycenter_x, 2: psf_barycenter_y, 3: psf_pixel_size,
	# 4: psf_fit_center_x, 5: psf_fit_center_y, 6: psf_fit_height, 7: psf_fit_width_x, 8: psf_fit_width_y,
	# 9: frame_number, 10: frame_time, 11: paralactic_angle
## l_max=0

for j in range(int(np.ceil(1.*len(dirlist)/naxis3))):
	if naxis3*j>len(dirlist):
		n3=naxis3*j-len(dirlist)
	else:
		n3=naxis3
	c=j*naxis3

	ndata,hdr=pyfits.getdata(dirlist[c], header=True)
	# Creating a double-cube with both channels
	if naxis2==-1:
		naxis2=ndata.shape[1]
	cube=np.zeros((n3,naxis2,naxis2))
	# Creating a double-list with both channels
	cent_list=np.ones((n3,12))*-1.

## for c in range(0, len(dirlist), naxis3): # Loop over the cubes
	t0_cube=time.time()

	trimpack_filename=target_pattern+"_"+dirlist[c]
	info_filename="scexao_parang_"+trimpack_filename[:-5]+".rdb"

	# Check if already processed
	if os.access(target_dir+os.sep+trimpack_filename, os.F_OK | os.R_OK):
		print('Already processed: '+trimpack_filename)
		skipped=skipped+1
		continue
	# Check if already processed
	elif os.access(target_dir+os.sep+trimpack_filename+'.EMPTY', os.F_OK | os.R_OK):
		print('Already processed, but no cube created: '+trimpack_filename)
		skipped=skipped+1
		continue

	new_cube=None
	new_info=None
	parang_list=None
	for n in range(n3):

		if c+n==len(dirlist):
			break
		sys.stdout.write("\n Processing cube ["+str(c+n+1)+"/"+str(len(dirlist))+"]: "+str(dirlist[c+n])+"\n")
		sys.stdout.flush()
		frame, header=pyfits.getdata(dirlist[c+n], header=True)
		## if np.max(cube_list['info'][c+n][:][11])==0: # In order to work in field tracking mode too.
		### frame_num       frame_time      paralactic_angle

		## jdate=float(header['MJD'])+2400000.5
		if parang_list is None:
			parang_list=np.array(np.hstack((n,graphic_nompi_lib.create_parang_scexao(header))))
			## utcstart=datetime2jd(dateutil.parser.parse(hdr['DATE']+"T"+hdr['UT']))
		else:
			parang_list=np.vstack((parang_list,np.hstack((n,graphic_nompi_lib.create_parang_scexao(header)))))

		# Trimming the frame
		cube[n]=frame[frame.shape[0]/2-naxis2/2:frame.shape[0]/2+naxis2/2,frame.shape[1]/2-naxis2/2:frame.shape[1]/2+naxis2/2]


	cent_list=np.ones((n3,9))
	cent_list[:,0]=np.arange(n3) # Frame number
	cent_list[:,1]=naxis2/2. # X center
	cent_list[:,2]=naxis2/2. # Y center
	cent_list=np.hstack((cent_list,parang_list))

	## trimpack_filename=target_pattern+"_"+dirlist[c+n]
	hdr['HIERARCH GC TRIMPACK']=str(__version__)+'.'+str(__subversion__)
	hdr['CRPIX1']='{0:14.7G}'.format(-frame.shape[0]/2.+hdr['CRPIX1']+naxis2/2.)
	hdr['CRPIX2']='{0:14.7G}'.format(-frame.shape[1]/2.+hdr['CRPIX2']+naxis2/2.)

	hdr['history']='Updated CRPIX1, CRPIX2'
	graphic_nompi_lib.save_fits(trimpack_filename, cube, target_dir=target_dir,  hdr=hdr, backend='pyfits')
	if not os.path.isdir(info_dir): # Check if info dir exists
		os.mkdir(info_dir)
	graphic_nompi_lib.write_array2rdb(info_dir+os.sep+info_filename,cent_list,header_keys)


	sys.stdout.write("\n Saved: {name} .\n Processed in {human_time} at {rate:.2f} MB/s \n"
					 .format(name=trimpack_filename, human_time=graphic_nompi_lib.humanize_time(time.time()-t0_cube) ,
							 rate=os.path.getsize(trimpack_filename)/(1048576*(time.time()-t0_cube))))
	sys.stdout.write("Remaining time: "+graphic_nompi_lib.humanize_time((time.time()-t_init)*(len(dirlist)-c)/(c-skipped+1))+"\n")
	sys.stdout.flush()

	del cube


## print("\n Program finished, killing all the slaves...")
print("Total time: "+graphic_nompi_lib.humanize_time((time.time()-t_init)))
## comm.bcast("over", root=0)
## if skipped==len(dirlist):
	## sys.exit(0)

if not hdr==None:
	if 'ESO OBS TARG NAME' in hdr.keys():
		log_file=log_file+"_"+string.replace(hdr['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
	else:
		log_file=log_file+"_"+string.replace(hdr['OBJECT'],' ','')+"_"+str(__version__)+".log"
	graphic_nompi_lib.write_log((time.time()-t_init),log_file,  comments=None)
sys.exit(0)

