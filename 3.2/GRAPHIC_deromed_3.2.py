#!python2.7
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

This is the final step of the reduction. Frames are all derotated to correct for
parlactic angle variation, and a temporal median is then calculated for each
pixel.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__='3.2'
__subversion__='0'

import scipy, glob,  os, sys, subprocess, string
from mpi4py import MPI
from scipy import ndimage
import argparse
import graphic_lib_320
import numpy as np
from graphic_lib_320 import dprint
import astropy.io.fits as fits

## sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
## import bottleneck


nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

target_dir = "."
d=0

parser = argparse.ArgumentParser(description='Derotates each single frame with respect to the parallactic angle.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0,help='Debug level, 0 is no debug')
parser.add_argument('--pattern', action="store", dest="pattern",  help='Filename pattern')
parser.add_argument('--naxis1', action="store",  dest="naxis1",  default=0, type=int, help='Original frame shape (e.g. 1024)')
parser.add_argument('--steps', action="store",  dest="steps",  default=2, type=int, help='Number of serialisation steps')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",  default='all*', help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
parser.add_argument('--info_type', action="store", dest="info_type",  default='rdb', help='Info directory')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('-nomask', dest='nomask', action='store_const',
				   const=False, default=True,
				   help='Produce only one image with masked out centre.')
parser.add_argument('--interpolate', dest='interpolate', action='store_const',
				   const=True, default=False,
				   help='Use interpolation instead of "3 shear FFT" for rotation')
parser.add_argument('-s', dest='stat', action='store_const',
				   const=True, default=False,
				   help='Print benchmarking statistics')
parser.add_argument('-nici', dest='nici', action='store_const',
				   const=True, default=False,
				   help='Switch for GEMINI/NICI data')
parser.add_argument('-sphere', dest='sphere', action='store_const',
				   const=True, default=False,
				   help='Switch for VLT/SPHERE data')
parser.add_argument('-scexao', dest='scexao', action='store_const',
				   const=True, default=False,
				   help='Switch for Subaru/SCExAO data')
parser.add_argument('-interactive', dest='interactive', action='store_const',
				   const=True, default=False,
				   help='Switch to set execution to interactive mode')
parser.add_argument('-nofit', dest='fit', action='store_const',
				   const=False, default=True,
				   help='Do not use PSF fitting values.')
parser.add_argument('-save_stack', dest='save_stack', action='store_const',
				   const=True, default=False,
				   help='Save intermediate derotation stacks.')
parser.add_argument('-bottleneck', dest='use_bottleneck', action='store_const',
				   const=True, default=False,
				   help='Use bottleneck module instead of numpy for nanmedian.')

args = parser.parse_args()
d=args.d
pattern=args.pattern
steps=args.steps
naxis1=args.naxis1
info_pattern=args.info_pattern
info_dir=args.info_dir
log_file=args.log_file
info_type=args.info_type
nici=args.nici
sphere=args.sphere
scexao=args.scexao
interpolate=args.interpolate
nomask=args.nomask
interactive=args.interactive
fit=args.fit
use_bottleneck=args.use_bottleneck

if use_bottleneck:
	from bottleneck import median as median
	from bottleneck import nanmedian as nanmedian
else:
	from numpy import nanmedian
	from numpy import median as median

med_tot=None

header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
	'frame_num', 'frame_time', 'paralactic_angle']

if rank==0:
	## sys.stdout = open('GC_derotate'+str(__version__)+"."+str(__subversion__)+".log", 'w')
	## sys.sterr = open('error_GC_derotate'+str(__version__)+"."+str(__subversion__)+".log", 'w')

	t_init=MPI.Wtime()
	skipped=0

	dirlist=graphic_lib_320.create_dirlist(pattern)
	if dirlist==None:
		print("No files found. Check --pattern option!")
		comm.bcast("over",root=0)
		sys.exit(1)

	infolist=graphic_lib_320.create_dirlist(info_dir+os.sep+info_pattern, extension='.rdb')
	## infolist=glob.glob(info_dir+os.sep+info_pattern+'*.'+info_type)
	## infolist.sort() # Sort the list alphabetically
	if infolist==None:
		print("No info files found, check your --info_pattern and --info_dir options.")
		sys.exit(1)

	cube_list, dirlist=graphic_lib_320.create_megatable(dirlist,infolist,keys=header_keys,nici=nici, sphere=sphere, scexao=scexao, fit=fit)

	comm.bcast(cube_list,root=0)

	# Search for the first valid angle to align all the frames to
	for cube_number in xrange(len(cube_list['info'])):
		for frame_number in xrange(cube_list['info'][cube_number].shape[0]):
			p0=float(cube_list['info'][cube_number][frame_number][11])
			if not p0==-1:
				break
		if not p0==-1:
			break


	comm.bcast(p0,root=0)

	hdulist = fits.open(dirlist[0])
	hdr=hdulist[0].header
	cube=hdulist[0].data
	## cube,hdr=pyfits.getdata(dirlist[0],header=True)
	if interpolate:
		fil ='interp_'
	else:
		fil=''
	if 'ESO OBS TARG NAME' in hdr.keys():
		log_file=log_file+"_"+string.replace(hdr['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
		fil=fil+str(hdr['ESO INS OPTI6 ID'])
		finalname='final_image_'+string.upper(string.replace(hdr['ESO OBS TARG NAME'],' ',''))+'_'+fil+'_'+dirlist[0]
	elif 'OBJECT' in hdr.keys():
		if not 'CHANNEL' in hdr.keys():
			fil=fil+''
		elif hdr['CHANNEL']=='BLUE':
			fil=fil+string.replace(hdr['FILTER_B'],'%','')
		elif hdr['CHANNEL']=='RED':
			fil=fil+string.replace(hdr['FILTER_R'],'%','')
		log_file=log_file+"_"+string.replace(hdr['OBJECT'],' ','')+"_"+str(__version__)+".log"
		finalname='final_image_'+string.upper(string.replace(hdr['OBJECT'],' ',''))+'_'+fil+'_'+dirlist[0]
	else:
		log_file=log_file+"_UNKNOW_TARGET_"+str(__version__)+".log"
		finalname='final_image_UNKNOW_TARGET_'+dirlist[0]
	# check if already done io.access
	if 'GC ORIG NAXIS1' in hdr.keys():
		naxis1=int(hdr['GC ORIG NAXIS1'])
		if 'GC LMAX' in hdr.keys():
			if naxis1>int(hdr['GC LMAX']):
				naxis1=0
	## elif 'GC ORIG NAXIS1' in hdr.keys():
		## naxis1=int(hdr['GC ORIG NAXIS1'])
		## if 'GC LMAX' in hdr.keys():
			## if naxis1>int(hdr['GC LMAX']):
				## naxis1=0

	cub_shape=cube.shape
	comm.bcast(cub_shape,root=0)
	comm.bcast(naxis1,root=0)
	del cube

	med_tot=None
	end=None

	hdr["HIERARCH GC DER FIT"]=(fit*1, "")
	hdr["HIERARCH GC DER VERS"]=(__version__+'.'+__subversion__, "")
	hdr.add_history("Following files used:")
	for input_filename in dirlist:
		hdr.add_history(input_filename)
	hdr.add_history("Final product of GRAPHIC reduction pipeline")

	# loop through the serie, broadcast range at each step
	for step in range(steps):
		step_filename=str(step)+"."+str(steps)+"_"+finalname
		if os.access( step_filename, os.F_OK ): # Check if file already exists
			## med_tot=pyfits.getdata(step_filename).byteswap().newbyteorder()
			## med_tot=pyfits.getdata(step_filename).byteswap().newbyteorder()
			print("Using already processed file: "+step_filename)
			hdulist_med = fits.open(step_filename)
			## hdr=hdulist[0].header
			med_tot=hdulist_med[0].data
			if step==steps-1:
				print('Already processed, nothing else to do, leaving...')
				sys.exit(0)
			continue
		if end==cub_shape[2]:
			break

		t0_step=MPI.Wtime()
		stack=None
		print('\n Step ['+str(step+1)+'/'+str(steps)+'] of derotation...')
		# define a range and broadcast it
		start=int(step*np.ceil(float(cub_shape[2])/(steps)))
		end=int((step+1)*np.ceil(float(cub_shape[2])/(steps)))
		if end >= cub_shape[2]:
			end=cub_shape[2]

		comm.bcast("derotate",root=0)
		comm.bcast(start, root=0)
		comm.bcast(end, root=0)
		graphic_lib_320.send_dirlist_slaves(dirlist)
		#stack the result
		#send stack chunks to slaves
		# Recover data from slaves
		for n in range(nprocs-1):
			data_in=comm.recv(source = n+1)
			if data_in==None:
				continue
			elif stack==None:
				stack=data_in
			else:
				if d > 0:
					print("data_in.shape: "+str(data_in.shape))
					print("stack.shape: "+str(stack.shape))
				stack=np.concatenate((stack, data_in),axis=0)
		if args.save_stack:
			graphic_lib_320.save_fits("derot_stack_"+step_filename, stack, hdr=hdr , backend='pyfits' )

		print(' Step ['+str(step+1)+'/'+str(steps)+'] of median calculation')
		comm.bcast("median", root=0)
		graphic_lib_320.send_chunks(stack,d)
		#gather and concat the result
		med=None
		for p in range(nprocs-1):
			r=comm.recv(source = p+1)
			if d>0:
				print("Received median reduced chunk from "+str(p+1))
			if med==None: #initialise
				med=r
			else:
				med=np.concatenate((med,r), axis=0)
		if med_tot==None:
			med_tot=med
		else:
			med_tot=np.concatenate((med_tot,med), axis=1)
		print("Saving: "+step_filename)
		graphic_lib_320.save_fits(step_filename, med_tot, hdr=hdr, backend='pyfits' )
		print("Step time: "+graphic_lib_320.humanize_time((MPI.Wtime()-t0_step)))

	# Put a cross in the centre
	# print(np.where(cube_list['info'][:][:][7]>0),7)
	if "GC RMIN" in hdr.keys():
		fwhm=hdr["GC RMIN"]
	elif "GC FWHM" in hdr.keys():
		fwhm=hdr["GC FWHM"]
	else:
		fwhm=median(cube_list['info'][0][np.where(cube_list['info'][0][:,7]>0),7])
	if nomask:
		med_nomask=med_tot.copy()
	med_tot=graphic_lib_320.mask_centre(med_tot, fwhm/2., med_tot.shape[0]/2., med_tot.shape[0]/2. )

	print("Saving: "+str(finalname))
	graphic_lib_320.save_fits(finalname, med_tot, hdr=hdr, backend='pyfits' )
	if nomask:
		graphic_lib_320.save_fits('nomask_'+finalname, med_nomask, hdr=hdr, backend='pyfits' )
	comm.bcast("over",root=0)
	MPI.Finalize()
	print("Total time: "+str(MPI.Wtime()-t_init)+" s = "+graphic_lib_320.humanize_time((MPI.Wtime()-t_init)))

	## log_file=log_file+"_"+hdr['HIERARCH ESO OBS TARG NAME']+"_"+str(__version__)+".log"
	graphic_lib_320.write_log((MPI.Wtime()-t_init),log_file)
	sys.exit(0)

if not rank==0:
	cube_list=comm.bcast(None,root=0)
	if cube_list=="over":
		sys.exit(1)
	p0=comm.bcast(None,root=0)
	cub_shape=comm.bcast(None,root=0)
	naxis1=comm.bcast(None,root=0)
	todo=comm.bcast(None,root=0)
	while not todo=="over":
		if todo=="derotate":
			# Get cube range to consider
			start=comm.bcast(None, root=0)
			end=comm.bcast(None, root=0)
			dirlist=comm.recv(source=0)

			dprint(d>1, "start: "+str(start)+", end: "+str(end))
			if dirlist==None: # Nothing todo, send back None and restart loop.
				comm.send(None, dest=0)
				todo=comm.bcast(None,root=0)
				continue

			filenumber=comm.recv(source=0)
			## p0=float(cube_list['info'][0][0][11])

			## s_stack=None
			## rs_cube=None
			full_stack=None
			for i in range(len(dirlist)):
				## graphic_lib_320.iprint(interactive, '\r\r\r '+str(rank)+': Derotating cube '+str(i+1)+' of '+str(len(dirlist))+' : '+str(dirlist[i]))
				graphic_lib_320.iprint(interactive, '\r\r\r '+str(rank)+': Derotating cube '+str(i+1)+' of '+str(len(dirlist))+' : '+str(dirlist[i])+'\n')

				hdulist_s = fits.open(dirlist[i],memmap=True)
				s_cube=hdulist_s[0].data
				rs_cube=np.ones((s_cube.shape[0],s_cube.shape[1],end-start))*np.NaN
				cn=cube_list['cube_filename'].index(dirlist[i])
				## if s_cube.shape[0]==1:
					## graphic_lib_320.dprint(d>1, 'Skipping corrupt frame. cube '+str(cn)+', frame '+str(fn))
					## continue
				for fn in range(s_cube.shape[0]):
					if cube_list['info'][cn][fn,5]==-1: # Invalid frame, skip
						graphic_lib_320.dprint(d>1, 'Skipping invalid frame, no PSf found: cube '+str(cn)+', frame '+str(fn))
						s_cube=s_cube[1:]
						continue
					elif interpolate:
						rs_cube[fn]=ndimage.interpolation.rotate(s_cube[0], p0-cube_list['info'][cn][fn,11],reshape=False, order=3, mode='constant', cval=np.NaN, prefilter=False)[:,start:end]
						s_cube=s_cube[1:]
					## elif not naxis1==0:
					else:
						rs_cube[fn]=graphic_lib_320.fft_3shear_rotate_pad(
							s_cube[0],p0-cube_list['info'][cn][fn,11],
							pad=2,x1=cube_list['info'][cn][fn,4],
							y1=cube_list['info'][cn][fn,5])[:,start:end]
						s_cube=s_cube[1:]
					## else:
					if False:
						## if rank==2:
							## print("")
							## print("s_cube[0] "+str(bottleneck.nanmax(s_cube[0])))
						## rs_cube[fn]=graphic_lib_320.fft_3shear_rotate_pad(s_cube[0],p0-cube_list['info'][cn][fn,11], x1=-1, pad=2)[:,start:end]
						rs_cube[fn]=graphic_lib_320.fft_3shear_rotate_pad(s_cube[0],p0-cube_list['info'][cn][fn,11], pad=2)[:,start:end]
						## temp=graphic_lib_320.fft_3shear_rotate_pad(s_cube[0],p0-cube_list['info'][cn][fn,11], pad=2)
						## if rank==2:
							## print("temp "+str(bottleneck.nanmax(temp)))
						## rs_cube[fn]=temp[:,start:end]
						## if rank==2:
							## print("rs_cube[fn] "+str(bottleneck.nanmax(rs_cube[fn])))
						s_cube=s_cube[1:]
				if d>2:
					graphic_lib_320.save_fits('rs_cube_'+str(rank)+'_'+dirlist[i], rs_cube, hdr=hdulist_s[0].header , backend='pyfits' )

				if full_stack==None:
					full_stack=rs_cube.copy()
				else:
					full_stack=np.concatenate((full_stack,rs_cube),axis=0)

			## comm.send(rs_cube,dest=0)
			comm.send(full_stack,dest=0)
			todo=comm.bcast(None,root=0)

		elif todo=="median":
			graphic_lib_320.iprint(interactive, "\r\r\r Process "+str(rank)+" of "+str(nprocs-1)+" calculating median.")

			start_col=comm.recv(source=0)
			my_cube=comm.recv(source=0)
			my_cube=nanmedian(my_cube,axis=0)
			comm.send(my_cube, dest = 0)
			del my_cube
			todo=comm.bcast(None,root=0)
		else:
			print(str(rank)+": Unexpected command received. Leaving...")
			sys.exit(1)
