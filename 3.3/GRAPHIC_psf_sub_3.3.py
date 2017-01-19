#!python2.7
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

Its purpose is to substract a generated PSF from each frame in a cube.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__='3.3'
__subversion__='0'

import numpy, scipy, glob, os, sys, fnmatch, string, time
import numpy as np
from scipy import ndimage
from mpi4py import MPI
import argparse
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
import random
import astropy.io.fits as fits

sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
import bottleneck


nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

target_dir = "."
target_pattern="ps_"+__version__+"."+__subversion__+"_"



parser = argparse.ArgumentParser(description='Subtracts a combined PSF from each single frame.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern",  default='*', help='Filename pattern')
parser.add_argument('--info_pattern', action="store", dest="info_pattern", required=True, help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
parser.add_argument('--n_fwhm', action="store", dest="n_fwhm", type=float, default=0.75, help='Number of FWHM displacements')
parser.add_argument('--dfwhm', action="store", dest="dfwhm", type=float, default=0.25, help='Number of FWHM displacements between PSF frames, useful for disks')
parser.add_argument('--rmin', action="store", dest="rmin", type=float, default=5, help='The minimal size in pixels to consider')
parser.add_argument('--tmax', action="store", dest="tmax", type=float, default=300, help='The maximum time difference in second between the frames')
parser.add_argument('--fmax', action="store", dest="fmax", type=int, default=300, help='The maximum number of frames to use to generate the PSF')
parser.add_argument('--fwhm', action="store", dest="fwhm", type=float, default=np.NaN, help='Give a custom PSF FWHM value, overriding the fitted value.')
parser.add_argument('--lmax', action="store", dest="l_max", type=float, default=0, help='Shape of the final image. If not specified it will be calculated to fit all the images.')
parser.add_argument('--ndither', action="store", dest="ndither", type=int, default=4, help='Number of dithering steps.')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('--max_in_mem', action="store", dest="max_in_mem",  default=None, help='Maximum number of cubes in memory')
parser.add_argument('-centred', dest='centred', action='store_const',
					const=True, default=False,
					help='Use this switch if the frames don\'t need to be recentred.')
parser.add_argument('-nofft', dest='nofft', action='store_const',
					const=True, default=False,
					help='Use interpolation instead of Fourier shift')
parser.add_argument('-nofit', dest='fit', action='store_const',
				   const=False, default=True,
				   help='Do not use PSF fitting values.')
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
parser.add_argument('-disk', dest='disk', action='store_const',
				   const=True, default=False,
				   help='Switch to use disk optimisations')
parser.add_argument('-interactive', dest='interactive', action='store_const',
				   const=True, default=False,
				   help='Switch to set execution to interactive mode')
parser.add_argument('-nomjd', dest='mjd', action='store_const',
				   const=False, default=True,
				   help='Switch to use LST instead of MJD dates')
parser.add_argument('-median', dest='combine', action='store_const',
				   const='median', default='mean',
				   help='Switch to use median instead of mean for frame combination.')


args = parser.parse_args()
d=args.d
pattern=args.pattern
info_pattern=args.info_pattern
info_dir=args.info_dir
n_fwhm=args.n_fwhm
dfwhm=args.dfwhm
rmin=args.rmin
tmax=args.tmax
fmax=args.fmax
fwhm=args.fwhm
ndither=args.ndither
log_file=args.log_file
centred=args.centred
nofft=args.nofft
fit=args.fit
nici=args.nici
sphere=args.sphere
scexao=args.scexao
disk=args.disk
interactive=args.interactive
mjd=args.mjd
max_in_mem=args.max_in_mem
combine=args.combine

## print(tmax)
#Convert tmax to MJD format until more clever way gets set
if mjd: # Convert second to MJD
	ctmax=float(tmax)/86400.
else: # Convert seconds to LST hours NO NEED to convert LST already given in seconds
	ctmax=tmax # no conversion needed

l_max=float(args.l_max)/2.

header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height',
	'psf_fit_width_x', 'psf_fit_width_y', 'frame_num', 'frame_time', 'paralactic_angle']

def read_recentre(rcn, rcubes, rcube_list, l_max):
	hdulist = fits.open(rcube_list['cube_filename'][rcn])
	rhdr=hdulist[0].header
	rcubes[rcn]=hdulist[0].data

	t0_trans=MPI.Wtime()
	if "GC RECENTER" in rhdr.keys():
		graphic_nompi_lib.iprint(interactive, '\r\r\r Already recentred: '+str(rcube_list['cube_filename'][rcn]))
		return rcubes, t0_trans
	# Send the cube to be recentreed
	graphic_nompi_lib.iprint(interactive, '\r\r\r Recentring '+str(rcube_list['cube_filename'][rcn]))

	comm.bcast("recentre",root=0)
	comm.bcast(l_max,root=0)
	comm.bcast(rcube_list['info'][rcn],root=0)
	graphic_mpi_lib.send_frames_async(rcubes[rcn])
	## rcubes.pop(rcn)
	rcubes[rcn]=None
	#rcube_list['info'][rcn]=None

	if args.stat==True:
		print("\n STAT: Data upload took: "+str(MPI.Wtime()-t0_trans)+" s = "+graphic_nompi_lib.humanize_time(MPI.Wtime()-t0_trans))
		t0_trans=MPI.Wtime()

	# Recover data from slaves
	for n in range(nprocs-1):
		data_in=comm.recv(source = n+1)
		if data_in is None:
			continue
		#info_in=comm.recv(source = n+1)
		graphic_nompi_lib.iprint(interactive, '\r\r\r Recentreed data from '+str(n+1)+' received									   =>')

		## if not rcn in rcubes.keys():
		if rcubes[rcn] is None:
			rcubes[rcn]=data_in
			#rcube_list['info'][rcn]=info_in
		else:
			rcubes[rcn]=np.concatenate((rcubes[rcn],data_in))
			#rcube_list['info'][rcn]=np.concatenate((rcube_list['info'][rcn],info_in),axis=0)
		## print(rcubes[rcn].shape)

	return rcubes, t0_trans

# Define ufunc for quicker recentreing

t_init=MPI.Wtime()

if rank==0:
	graphic_nompi_lib.print_init()

	if centred:
		target_pattern=target_pattern+'nc_'

	naxis1=0
	naxis2=0

	## dirlist=glob.glob(pattern+'*.fits')
	## dirlist.sort() # Sort the list alphabetically
	dirlist=graphic_nompi_lib.create_dirlist(pattern, target_dir='.', extension='.fits')
	if len(dirlist) is None:
		print("No files found")
		for n in range(nprocs-1):
			comm.send("over",dest =n+1)
		sys.exit(1)

	## # Check values in dirlist and remove dodgy files.
	## for i in range(len(dirlist)):
		## if not os.access(dirlist[i], os.F_OK | os.R_OK): # Check if file exists
			print("")
			## print(str(rank)+': Error, cannot access: '+dirlist[i])
			## dirlist[i]=None
			## continue

	# Clean dirlist of discarded values:
	dirlist.sort()
	skipped=0
	for i in range(len(dirlist)):
		if dirlist[0] is None:
			dirlist.pop(0)
			skipped=skipped+1
		else: break

	print("Found "+str(len(dirlist))+" files.")

	infolist=graphic_nompi_lib.create_dirlist(info_dir+os.sep+'*'+info_pattern, target_dir=info_dir, extension='.rdb')

	## infolist=glob.glob(info_dir+os.sep+'*'+info_pattern+'*.rdb')
	## infolist.sort() # Sort the list alphabetically
	print("Found "+str(len(infolist))+" info files.")

	cube_list, dirlist, skipped=graphic_nompi_lib.create_megatable(dirlist, infolist, skipped, keys=header_keys, nici=nici, sphere=sphere, scexao=scexao, fit=fit)
	if dirlist==[]:
		for n in range(nprocs-1):
			comm.send("over",dest =n+1)
		sys.exit(1)

	cubes={}
	if len(dirlist)==skipped or len(dirlist)==0 or len(infolist)==0:
		sys.exit(1)

	print("")

	## # Clean dirlist of discarded values:
	## dirlist.sort() # Put all the None at the beginning
	## for i in range(len(dirlist)):
		## if dirlist[0] is None:
			## dirlist.pop(0)
			## skipped=skipped+1
		## else: break

	if skipped>0:
		print(" Skipped "+str(skipped)+" cubes.")

	skipped=0
	## l_max=0
		# 0: frame_number, 1: psf_barycentre_x, 2: psf_barycentre_y, 3: psf_pixel_size,
		# 4: psf_fit_centre_x, 5: psf_fit_centre_y, 6: psf_fit_height, 7: psf_fit_width_x, 8: psf_fit_width_y,
		# 9: frame_number, 10: frame_time, 11: paralactic_angle
	t0_cube=MPI.Wtime()
	## if nici: # Convert JD to unix time to get rid of day change issues
		## for c in range(len(cube_list['cube_filename'])): # Loop over the cubes
			## cube_list['info'][c][:,10]=(cube_list['info'][c][:,10]-2440587.5)*86400

	if l_max==0:
		for i in range(len(cube_list['info'])):
			## print(cube_list['info'][i][len(cube_list['info'][i])/2,4])
			if not cube_list['info'][i][len(cube_list['info'][i])/2,4]==-1:
				hdulist = fits.open(cube_list['cube_filename'][i])
				l=graphic_nompi_lib.get_max_dim(cube_list['info'][i][len(cube_list['info'][i])/2,4], cube_list['info'][i][len(cube_list['info'][i])/2,5], hdulist[0].header['NAXIS1'], cube_list['info'][i][:,11])
				if l>l_max: l_max=l
		graphic_mpi_lib.dprint(d>1, 'l_max: '+str(l_max))
		l_max=int(np.floor(l_max))

	time_treshold=0

	for c in range(len(cube_list['cube_filename'])): # Loop over the cubes
		t0_cube=MPI.Wtime()
		# take the median psf x_fwhm in the cube
		if np.isnan(fwhm):
			fwhm=bottleneck.median(cube_list['info'][c][np.where(cube_list['info'][c][:,7]>0),7])

		if l_max==0:
			if disk:
				psf_sub_filename=target_pattern+'disk_'+str(fwhm)+'_'+str(n_fwhm)+"_"+str(dfwhm)+"_"+str(rmin)+"_"+str(tmax)+"_"+cube_list['cube_filename'][c]
			else:
				psf_sub_filename=target_pattern+str(fwhm)+'_'+str(n_fwhm)+"_"+str(rmin)+"_"+str(tmax)+"_"+cube_list['cube_filename'][c]
		else:
			if disk:
				psf_sub_filename=target_pattern+'_disk_'+str(fwhm)+'_'+str(n_fwhm)+"_"+str(dfwhm)+"_"+str(rmin)+"_"+str(tmax)+"_"+str(np.ceil(2*l_max))+'_'+cube_list['cube_filename'][c]
			else:
				psf_sub_filename=target_pattern+str(fwhm)+'_'+str(n_fwhm)+"_"+str(rmin)+"_"+str(tmax)+"_"+str(np.ceil(2*l_max))+'_'+cube_list['cube_filename'][c]


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

		graphic_nompi_lib.iprint(interactive, "Processing cube ["+str(c+1)+"/"+str(len(cube_list['cube_filename']))+"]: "+str(cube_list['cube_filename'][c])+"\n")


		# Initialise new cube to contain the psf subtracted frames
		final_cube=None

		# Initialise new tables to keep track of all frame informations
		new_info=None
		info_filename="all_info_"+psf_sub_filename[:-5]+".rdb"

		## alpha=2*np.sin((n_fwhm*fwhm)/(2.*rmin)) # n*fwhm/rmin
		alpha=np.rad2deg(np.arctan((n_fwhm*fwhm)/rmin)) # n*fwhm/rmin
		if disk:
			## dalpha=2*np.sin((dfwhm*fwhm)/(2.*rmin))
			dalpha=np.rad2deg(np.arctan((dfwhm*fwhm)/rmin))
		hdulist = fits.open(cube_list['cube_filename'][c])
		hdr=hdulist[0].header

		if 'NAXIS1' in hdr:
			naxis1=hdr['NAXIS1']
		if 'NAXIS2' in hdr:
			naxis2=hdr['NAXIS2']
		## if l_max==0:
			## l_max=int(hdr['NAXIS1']) # The final cube size is doubled

		empty_frame=0

		for f in range(len(cube_list['info'][c])): # Loop over the frames in the cube
			t0_frame=MPI.Wtime()

			graphic_nompi_lib.iprint(interactive, "\r\r\r Processing cube ["+str(c+1)+"/"+str(len(cube_list['cube_filename']))+"]: "+str(cube_list['cube_filename'][c])+", frame "+str(f+1)+"/"+str(len(cube_list['info'][c])))

			# Create stack to hold frames for PSF generation
			stack=np.ones((fmax,l_max*2,l_max*2))*np.NaN
			cube_count=0

			# Skip bad frames
			if cube_list['info'][c][f][6]==-1:
				graphic_mpi_lib.dprint(d>1, "Skipping bad frame (c,f): "+str(c)+","+str(f)+", info: "+str(cube_list['info'][c][f]))
				continue

			p0=float(cube_list['info'][c][f][11])
			t0=float(cube_list['info'][c][f][10])
			#cube_timespan=np.abs(cube_list['info'][c][1][10]-cube_list['info'][c][-1][10])
			cube_timespan=np.abs(cube_list['info'][c][0][10]-cube_list['info'][c][-1][10])
			valid_count=0
			valid_cubes=0
			valid={}

			if not max_in_mem is None and len(cubes.keys())>max_in_mem:
				print('Too many cubes loaded in memory, temporarly adapting ctmax, removing '+str(cube_timespan))
				time_treshold=time_treshold+cube_timespan
			elif time_treshold>0:
				print('Less cubes loaded in memory, re-adapting ctmax, adding '+str(cube_timespan))
				time_treshold=time_treshold-cube_timespan

			if len(cubes.keys())>0:
				for cn in cubes.keys():
					if np.min(np.abs(cube_list['info'][cn][:,10]-t0))>ctmax:
					# pop this cube from list, since it's not needed anymore
						cubes.pop(cn)
						graphic_nompi_lib.iprint(interactive, "\n Removing cube "+str(cn)+" from stored cubes.")


			if args.stat==True:
				tb=MPI.Wtime()


			TooLate=False
			TooEarly=False

			# Loop through cubes to look for frames compatible with rotation and time conditions
			for n in range(-1,2*len(cube_list['cube_filename'])):
				# Alternate between cubes before and after ref-cube, prevent cn from becoming negative (cube-wrap)
				cn=int(c+((-1)**(n+c))*np.ceil((n+1)/2.))
				if cn<0:
					graphic_mpi_lib.dprint(d>3,"Wrapping (c,n,cn): "+str(c)+","+str(n)+","+str(cn))
					graphic_mpi_lib.dprint(d>1, "TooEarly")
					TooEarly=True
					continue
				elif cn+1>len(cube_list['cube_filename']):
					graphic_mpi_lib.dprint(d>3,"Wrapping (c,n,cn): "+str(c)+","+str(n)+","+str(cn))
					graphic_mpi_lib.dprint(d>1, "TooLate")
					TooLate=True
					continue
				#if np.min(cube_list['info'][cn][1:,10])>t0+ctmax-time_treshold:
				if np.min(cube_list['info'][cn][0:,10])>t0+ctmax-time_treshold:
					# Need to use [1:,10] as older cube-info had the first date in LST
					TooLate=True
					#graphic_mpi_lib_330.dprint(d>1, "TooLate: np.min(cube_list['info'][cn][1:,10])"+str(np.min(cube_list['info'][cn][1:,10]))+">"+str(t0)+'+'+str(ctmax)+'='+str(t0+ctmax)+'=t0+ctmax')
					graphic_mpi_lib.dprint(d>1, "TooLate: np.min(cube_list['info'][cn][0:,10])"+str(np.min(cube_list['info'][cn][0:,10]))+">"+str(t0)+'+'+str(ctmax)+'='+str(t0+ctmax)+'=t0+ctmax')
				#elif np.max(cube_list['info'][cn][1:,10])<t0-ctmax+time_treshold:
				elif np.max(cube_list['info'][cn][0:,10])<t0-ctmax+time_treshold:
					# Need to use [1:,10] as older cube-info had the first date in LST
					TooEarly=True
					#graphic_mpi_lib_330.dprint(d>1, "TooEarly: np.max(cube_list['info'][cn][1:,10])"+str(np.max(cube_list['info'][cn][1:,10]))+"<"+str(t0)+'-'+str(ctmax)+'='+str(t0-ctmax)+'=t0-ctmax')
					graphic_mpi_lib.dprint(d>1, "TooEarly: np.max(cube_list['info'][cn][0:,10])"+str(np.max(cube_list['info'][cn][0:,10]))+"<"+str(t0)+'-'+str(ctmax)+'='+str(t0-ctmax)+'=t0-ctmax')

				if TooEarly and TooLate:
					# No need to look further, remaing cubes are too far in time
					break

				if d >0:
					print("")
					print("debug 3: stacking condition: alpha="+str(alpha)+", p0="+str(p0)+", t0="+str(t0)+", ctmax="+str(ctmax))
					print("cn: "+str(cn))
					if d>1:
						print("angle-p0: "+str(cube_list['info'][cn][:,11]-p0))
						print("time-t0: "+str(cube_list['info'][cn][:,10]-t0))
					else:
						print("angle>alpha: "+str(np.where(np.abs(cube_list['info'][cn][:,11]-p0)>alpha)))
						print("time<ctmax: "+str(np.where(np.abs(cube_list['info'][cn][:,10]-t0)<ctmax)))

				graphic_mpi_lib.dprint(d>2, "cube_list['info'][cn]"+str(cube_list['info'][cn]))
				graphic_mpi_lib.dprint(d>3,cube_list['info'][cn][:,11])

				if d==-123:
					print(cube_list['cube_filename'][cn])

				p1=p0
				#Check if it's worth reading the cube
				######################
				# Format of the valid list. For each (n,cn) doublet, there is a list of valid frames for that specific cube number (cn).
				############
				valid[(n+1,cn)]=np.where((np.abs(cube_list['info'][cn][:,11]-p0)>alpha) & (np.abs(cube_list['info'][cn][:,10]-t0)<ctmax) & np.not_equal(cube_list['info'][cn][:,6],-1))[0] #(not cube_list['info'][cn][:,2]<0))]
				## valid_rotation[(n+1,cn)]=np.where(np.abs(cube_list['info'][cn][:,11]-p0)>alpha)[0]
				## valid_time[(n+1,cn)]=np.where(np.abs(cube_list['info'][cn][:,10]-t0)<ctmax)[0]
				## valid_frames[(n+1,cn)]=np.where(np.not_equal(cube_list['info'][cn][:,6],-1))[0]
				graphic_mpi_lib.dprint(d>0, "Valid: "+str(cube_list['cube_filename'][cn])+', '+str(valid)+', '+str(len(valid)))
				if len(valid[(n+1,cn)])==0: # Remove empty list
					valid.pop((n+1,cn))
				else:
					valid_cubes+=1
					if not cn in cubes.keys(): # Check if cube already loaded
						if centred: # Check if cube already centred (quicklook)
							## hdulist = fits.open(cube_list['cube_filename'][cn])
							## data=hdulist[0].data
							data=fits.getdata(cube_list['cube_filename'][cn])
							bigstack=np.ones((data.shape[0],l_max*2,l_max*2))*np.NaN
							print(l_max,bigstack.shape, data.shape)
							# ACC: be careful with the array indices when putting one arbitrary sized array into another arbitrary sized array
							mindim=np.min([l_max*2,data.shape[1]]) # This ensures that l_max < data.shape[1] and l_max > data.shape[1] both work.
							xstack_min=np.max([0.,l_max-mindim/2])
							xstack_max=np.min([l_max*2,l_max+mindim/2])
							ystack_min=np.max([0.,l_max-mindim/2])
							ystack_max=np.min([l_max*2,l_max+mindim/2])
							xdata_min=np.max([0.,data.shape[1]/2-mindim/2])
							xdata_max=np.min([data.shape[1],data.shape[1]/2]+mindim/2)
							ydata_min=np.max([0.,data.shape[2]/2-mindim/2])
							ydata_max=np.min([data.shape[2],data.shape[2]/2]+mindim/2)
							bigstack[:,xstack_min:xstack_max,ystack_min:ystack_max]=data[:,xdata_min:xdata_max,ydata_min:ydata_max]
							cubes[cn]=bigstack
							del bigstack, data
						else: # Send cube for recentring
							cubes, t0_trans=read_recentre(cn, cubes, cube_list, l_max)
						if args.stat==True:
							print("\n STAT: Data download took: "+str(MPI.Wtime()-t0_trans)+" s = "+graphic_nompi_lib.humanize_time(MPI.Wtime()-t0_trans))
							print("\n STAT: Recentreing took: "+str(MPI.Wtime()-t0_trans)+" s = "+graphic_nompi_lib.humanize_time(MPI.Wtime()-tb))


			# Check if the two first cubes are not contiguous.
			## if (valid.keys()[0][0]+valid.keys()[1][0]) % 2:

			si=0 # Stack position incrementer
			if not len(valid.keys())==0:
				sample=fmax/len(valid.keys())
				for j,k in valid.keys():
					print(sample, fmax, len(valid.keys()), len(valid[(j,k)]))
					print(cubes[k].shape, valid[(j,k)])
					if len(valid[(j,k)])<sample:
						stack[si:si+len(valid[(j,k)])]=cubes[k][valid[(j,k)]]
						si+=len(valid[(j,k)])
						cube_count+=1
					else:
						# Could add PCA step here
						print(si, sample, j, k)
						stack[si:si+sample]=cubes[k][random.sample(valid[(j,k)],sample)]
						si+=sample
						cube_count=cube_count+1

				## mask=np.empty(len(valid),dtype=bool)
				## mask[:]=True

				## valid_count=valid_count+len(valid)
				## if disk:
					## for i in range(len(valid)):
						## fn=valid[i]
						## if np.abs(cube_list['info'][cn][fn,11]-p1)>dalpha:
							## p1=cube_list['info'][cn][fn,11]
						## else:
							## mask[i]=False
					## stack=np.concatenate((stack,cubes[cn][valid[mask]]))

				## if stack is None:
					## graphic_mpi_lib.dprint(d>19, "stack: "+str(stack)+", cube_list['info'][cn]: "+str(cube_list['info'][cn]))
					## # Create frame stack
					## stack=cubes[cn][valid[mask]] #(not cube_list['info'][cn][:,2]<0))]
					## cube_count=cube_count+1

				## elif stack.shape[0]<fmax:
					## stack=np.concatenate((stack,cubes[cn][valid[mask]]))
					## cube_count=cube_count+1
				## elif stack.shape[0]>fmax:
					## stack=np.concatenate((stack,cubes[cn][valid[mask]]))
					## cube_count=cube_count+1

			if args.stat==True:
				print("\n STAT: Stack preparation took: "+str(MPI.Wtime()-t0_trans)+" s = "+graphic_nompi_lib.humanize_time(MPI.Wtime()-tb))

			#graphic_nompi_lib.iprint(interactive, "\r\r\r Processing cube ["+str(c+1)+"/"+str(len(cube_list['cube_filename']))+"]: "+str(cube_list['cube_filename'][c])+", frame "+str(f+1)+"/"+str(len(cube_list['info'][c]))+" recentreing stack. Kept "+str(stack.shape[0])+" out of "+str(valid_count)"+ frames.")


			if stack is None or sum(len(i) for i in valid.itervalues())==0:
				empty_frame=empty_frame+1
				graphic_nompi_lib.iprint(interactive, "\r\r\r Processing cube ["+str(c+1)+"/"+str(len(cube_list['cube_filename']))+"]: "+str(cube_list['cube_filename'][c])+", frame "+str(f+1)+"/"+str(len(cube_list['info'][c]))+" .... no frames found to generate PSF. Valid_cubes="+str(valid_cubes)+" si="+str(si))
			else:
				graphic_nompi_lib.iprint(interactive, "\r\r\r Processing cube ["+str(c+1)+"/"+str(len(cube_list['cube_filename']))+"]: "+str(cube_list['cube_filename'][c])+", frame "+str(f+1)+"/"+str(len(cube_list['info'][c]))+" calculating "+combine+". Kept "+str(si)+" out of "+str(sum(len(i) for i in valid.itervalues()))+" valid frames.")




				tb=MPI.Wtime()
				t0_trans=MPI.Wtime()
				if d>0:
					graphic_nompi_lib.save_fits('temp_psf_stack.fits',stack,hdr=hdr, backend='pyfits')

				# Send chunks to get median (cut along x axis)
				comm.bcast(combine,root=0)
				graphic_mpi_lib.send_chunks(stack,d)
				if args.stat==True:
					graphic_nompi_lib.iprint(interactive, "\n STAT: Data upload took: "+str(MPI.Wtime()-t0_trans)+" s = "+graphic_nompi_lib.humanize_time(MPI.Wtime()-t0_trans)+"\n")


				stack=None
				chunk=None
				psf=None
				t0_trans=MPI.Wtime()


				## gather results and save
				for n in range(nprocs-1):
					chunk=comm.recv(source = n+1)
					try:
						chunk.shape
					except:
						print("Error, received: "+str(chunk))
						continue

					graphic_nompi_lib.iprint(interactive, '\r\r\r '+combine+' processed data from '+str(n+1)+' received									 =>')


					if psf is None: #initialise
						psf=chunk
					else:
						psf=np.concatenate((psf,chunk), axis=0)

				if args.stat==True:
					graphic_nompi_lib.iprint(interactive, "\n STAT: Data download took: "+str(MPI.Wtime()-t0_trans)+" s = "+graphic_nompi_lib.humanize_time(MPI.Wtime()-t0_trans))
					graphic_nompi_lib.iprint(interactive, "\n STAT: '+combine+' calculation took: "+str(MPI.Wtime()-tb)+" s = "+graphic_nompi_lib.humanize_time(MPI.Wtime()-tb))


				if d > 2:
					graphic_nompi_lib.iprint(interactive, "\n DEBUG: cubes.keys: "+str(cubes.keys()))


				if not c in cubes.keys():
					cubes, t0_trans=read_recentre(c,cubes, cube_list, l_max)
					graphic_nompi_lib.iprint(interactive, '\n '+str(len(cubes.keys()))+' stored in memory')
				if final_cube is None: # Check if a cube has already been started
					final_cube=cubes[c][f][np.newaxis,...]-psf.clip(0)
				else:
					final_cube=np.concatenate((final_cube,cubes[c][f][np.newaxis,...]-psf.clip(0)), axis=0)


				temp_info=cube_list['info'][c][f]
				## # Adjust centre value using "psf" frame for shape
				temp_info[1]=psf.shape[0]/2. #x centroid
				temp_info[2]=psf.shape[1]/2. #y centroid
				temp_info[4]=psf.shape[0]/2. #x psf fit
				temp_info[5]=psf.shape[1]/2.	#y psf fit

				if new_info is None:
					## new_info=cube_list['info'][c][f]
					new_info=temp_info
				else:
					new_info=np.vstack((new_info,temp_info))


				if args.stat==True:
					print("\n STAT: Frame processing took: "+str(MPI.Wtime()-t0_trans)+" s = "+graphic_nompi_lib.humanize_time(MPI.Wtime()-t0_frame))
				graphic_mpi_lib.dprint(d==-31 and not new_info is None, str(f)+" new_info "+str(len(new_info)))
				graphic_mpi_lib.dprint(d==-31 and not final_cube is None, "final_cube.shape: "+str(final_cube.shape))
				graphic_mpi_lib.dprint(d==-31 and final_cube is None, "final_cube: "+str(final_cube))

		if empty_frame==hdr['NAXIS3']:# cubes[c].shape[0]-1:  #All frames are empty.
			final_cube=None

		if final_cube is None:
			graphic_nompi_lib.iprint(interactive, "\n No cube generated for "+str(cube_list['cube_filename'][c])+"! You should consider relaxing n_fwhm, rmin, or tmax conditions.")

			open(psf_sub_filename+'.EMPTY', 'a').close()
		elif not len(final_cube.shape)==3:
			graphic_nompi_lib.iprint(interactive, "\n No cube generated for "+str(cube_list['cube_filename'][c])+"! You should consider relaxing n_fwhm, rmin, or tmax conditions.")

			open(psf_sub_filename+'.EMPTY', 'a').close()
		else:
			hdr["HIERARCH GC PSF_SUB"]=(__version__+'.'+__subversion__, "")
			hdr["HIERARCH GC PS FIT"]=(fit*1, "")
			hdr["HIERARCH GC FWHM"]=(fwhm, "")
			hdr["HIERARCH GC COMBINE"]=(combine, "")
			hdr["HIERARCH GC N_FWHM"]=(n_fwhm,"")
			hdr["HIERARCH GC TMAX"]=(tmax,"")
			hdr["HIERARCH GC RMIN"]=(rmin,"")
			hdr["HIERARCH GC FMAX"]=(fmax,"")
			hdr["HIERARCH GC LMAX"]=(l_max,"")
			hdr["HIERARCH GC ORIG NAXIS1"]=(hdr['NAXIS1'],"")
			hdr["HIERARCH GC ORIG NAXIS2"]=(hdr['NAXIS2'],"")
			hdr["HIERARCH GC DISK"]=(str(disk),"")
			if 'CRPIX1' in hdr.keys() and 'CRPIX2' in hdr.keys() :
				hdr["CRPIX1"]=('{0:14.7G}'.format(final_cube.shape[1]/2.+float(hdr['CRPIX1'])-cube_list['info'][c][hdr['NAXIS3']/2,4]), "")
				hdr["CRPIX2"]=('{0:14.7G}'.format(final_cube.shape[2]/2.+float(hdr['CRPIX2'])-cube_list['info'][c][hdr['NAXIS3']/2,5]), "")

			hdr['history']="Updated CRPIX1, CRPIX2"
			## graphic_mpi_lib.save_fits(psf_sub_filename, final_cube, hdr=hdr, backend='pyfits')
			graphic_nompi_lib.save_fits(psf_sub_filename, final_cube, hdr=hdr, backend='pyfits')

			## print(new_info[1][:])
			## # Adjust centre value using "psf" frame for shape
			## new_info[1][:]=psf.shape[0]/2. #x centroid
			## new_info[2][:]=psf.shape[1]/2. #y centroid
			## new_info[4][:]=psf.shape[0]/2. #x psf fit
			## new_info[5][:]=psf.shape[1]/2.	#y pas fit

			graphic_nompi_lib.write_array2rdb(info_dir+os.sep+info_filename,new_info,header_keys)

			graphic_nompi_lib.iprint(interactive, "\n Saved: {name} .\n Processed in {human_time} at {rate:.2f} MB/s \n {cubes_in_mem} loaded in memory \n"
							 .format(name=psf_sub_filename, human_time=graphic_nompi_lib.humanize_time(MPI.Wtime()-t0_cube) ,
									 rate=os.path.getsize(psf_sub_filename)/(1048576*(MPI.Wtime()-t0_cube)),
									 cubes_in_mem=len(cubes.keys())))
			graphic_nompi_lib.iprint(interactive, "Remaining time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t_init)*(len(cube_list['cube_filename'])-c)/(c-skipped+1))+"\n")

		del final_cube


	print("\n Program finished, killing all the slaves...")
	print("Total time: "+str(MPI.Wtime()-t_init)+" s = "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t_init)))
	comm.bcast("over", root=0)
	## if 'ESO OBS TARG NAME' in hdr.keys():
		## log_file=log_file+"_"+string.replace(hdr['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
	## elif 'OBJECT' in hdr.keys():
		## log_file=log_file+"_"+string.replace(hdr['OBJECT'],' ','')+"_"+str(__version__)+".log"
	## else:
		## log_file=log_file+"_UNKNOW_TARGET_"+str(__version__)+".log"
	## graphic_nompi_lib.write_log((MPI.Wtime()-t_init),log_file)
	graphic_nompi_lib.write_log_hdr((MPI.Wtime()-t_init), log_file, hdr, comments=None, nprocs=nprocs)

	if nici:
		print('--pattern '+string.split(psf_sub_filename, '_S')[0]+' --info_pattern '+string.split(info_filename,'_S')[0])
	elif sphere:
		print('--pattern '+string.split(psf_sub_filename, 'SPHER')[0]+' --info_pattern '+string.split(info_filename,'SPHER')[0])
	elif scexao:
		print('--pattern '+string.split(psf_sub_filename, 'HIC')[0]+' --info_pattern '+string.split(info_filename,'HIC')[0])
	else:
		print('--pattern '+string.split(psf_sub_filename, 'NACO')[0]+' --info_pattern '+string.split(info_filename,'NACO')[0])
	sys.exit(0)

#######################################################################
#
# SLAVES
#
# slaves need to:
# receive stack and frame
# recentre frames in stack
# calculate median
# subtract median from frame
# improvement could be done by somehow keeping recentreed frames
#
#######################################################################
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

		elif todo=="mean":

			# Receive number of first column
			start_col=comm.recv(source=0)
			# Receive stack to median
			stack=comm.recv(source=0)
			if d>5:
				print("")
				print(str(rank)+" stack.shape: "+str(stack.shape))
			# Mask out the NaNs
			psf=bottleneck.nanmean(stack, axis=0)
			del stack
			comm.send(psf, dest=0)
			del psf

		elif todo=="recentre":
			# Receive padding dimension
			l_max=comm.bcast(None,root=0)
			# Receive info table used for recentreing
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
				if stack is None:
					print(str(rank)+" received stack: "+str(stack))
				else:
					print(str(rank)+" received stack, shape="+str(stack.shape))


			if not (stack is None or len(stack.shape)<3):
				if 2*l_max<stack.shape[1]:
					bigstack=np.zeros((stack.shape[0],stack.shape[1]*2,stack.shape[2]*2))
					smallstack=np.zeros((stack.shape[0],l_max*2,l_max*2))
					bigstack[:,
						stack.shape[1]/2:3*stack.shape[1]/2,
						stack.shape[2]/2:3*stack.shape[2]/2]=stack
				else:
					bigstack=np.zeros((stack.shape[0],l_max*2,l_max*2))
					bigstack[:,
						l_max-stack.shape[1]/2:l_max+stack.shape[1]/2,
						l_max-stack.shape[2]/2:l_max+stack.shape[2]/2]=stack
					## print(bigstack[:,
						## l_max-stack.shape[1]/2:l_max+stack.shape[1]/2,
						## l_max-stack.shape[2]/2:l_max+stack.shape[2]/2].shape)
					## print(stack.shape)
					## sys.exit(1)
				stack_shape=stack.shape
				del stack
				for fn in range(bigstack.shape[0]):
					graphic_mpi_lib.dprint(d>2, "recentreing frame: "+str(fn)+" with shape: "+str(bigstack[fn].shape))
					if info_stack[s+fn,4]==-1 or info_stack[s+fn,5]==-1 or info_stack[s+fn,6]==-1:
						bigstack[fn]=np.NaN
						continue
					# Shift is given by (image centre position)-(star position)
					if nofft==True: # Use interpolation
						if 2*l_max<stack_shape[1]:
							smallstack[fn]=ndimage.interpolation.shift(bigstack[fn], (stack_shape[1]/2.-info_stack[s+fn,4], stack_shape[2]/2.-info_stack[s+fn,5]),
								order=3, mode='constant', cval=np.NaN, prefilter=False)[
								bigstack_shape[1]/2-l_max:l_max+bigstack.shape[1]/2,bigstack.shape[2]/2-l_max:l_max+bigstack.shape[2]/2]
						else:
							bigstack[fn]=ndimage.interpolation.shift(bigstack[fn], (stack_shape[1]/2.-info_stack[s+fn,4], stack_shape[2]/2.-info_stack[s+fn,5]), order=3, mode='constant', cval=np.NaN, prefilter=False)
					else: # Shift in Fourier space
						if 2*l_max<stack_shape[1]:
							smallstack[fn]=graphic_nompi_lib.fft_shift(bigstack[fn], stack_shape[1]/2.-info_stack[s+fn,4], stack_shape[2]/2.-info_stack[s+fn,5])[
								bigstack.shape[1]/2-l_max:l_max+bigstack.shape[1]/2,bigstack.shape[2]/2-l_max:l_max+bigstack.shape[2]/2]
						else:
							bigstack[fn]=graphic_nompi_lib.fft_shift(bigstack[fn], stack_shape[1]/2.-info_stack[s+fn,4], stack_shape[2]/2.-info_stack[s+fn,5])
					## if l_max<stack_shape[1]:
						## bigstack=smallstack
						## if bigstack.shape[1]>np.ceil(l_max - info_stack[s+fn,4]-0.5):
							## bigstack[fn,:np.ceil(l_max - info_stack[s+fn,4]-0.5),:]=np.NaN
						## if np.floor(l_max - info_stack[s+fn,4]+0.5+stack_shape[1])>0:
							## bigstack[fn,np.floor(l_max - info_stack[s+fn,4]+0.5+stack_shape[1]):,:]=np.NaN
						## if bigstack.shape[1]>np.ceil(l_max - info_stack[s+fn,5]-0.5):
							## bigstack[fn,:,:np.ceil(l_max - info_stack[s+fn,5]-0.5)]=np.NaN
						## if np.floor(l_max - info_stack[s+fn,5]+0.5+stack_shape[2])>0:
							## bigstack[fn,:,np.floor(l_max - info_stack[s+fn,5]+0.5+stack_shape[2]):]=np.NaN
					if 2*l_max>stack_shape[1]:
						bigstack[fn,:int(np.ceil(l_max - info_stack[s+fn,4]-0.5)),:]=np.NaN
						bigstack[fn,int(np.floor(l_max - info_stack[s+fn,4]+0.5+stack_shape[1])):,:]=np.NaN
						bigstack[fn,:,:int(np.ceil(l_max - info_stack[s+fn,5]-0.5))]=np.NaN
						bigstack[fn,:,int(np.floor(l_max - info_stack[s+fn,5]+0.5+stack_shape[2])):]=np.NaN
				if 2*l_max<stack_shape[1]:
					bigstack=smallstack.copy()
					del smallstack
				graphic_mpi_lib.dprint(d>2, "Sending back bigstack, shape="+str(bigstack.shape))
				comm.send(bigstack, dest = 0)
				del bigstack

			else:
				comm.send(None, dest = 0 )

		else:
			print(str(rank)+": received "+str(todo)+". Leaving....")
			comm.send(None, dest = 0 )

		todo=comm.bcast(None,root=0)
