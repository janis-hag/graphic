#!/usr/bin/python
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline for
High-contrast Imaging of planetary Companions".
Its purpose is to inject fake companions in a cube list, to get
detection limits.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__='3.2'
__subversion__='0'

import numpy, scipy, glob, shutil, os, sys, time, fnmatch, argparse, string
import graphic_lib_320
from mpi4py import MPI
from scipy import ndimage
import numpy as np
from astropy.io import fits

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD


parser = argparse.ArgumentParser(description='GRAPHIC:\n The Geneva Reduction and Analysis Pipeline forHigh-contrast Imaging of planetary Companions.\n\n\
This program injects fake companions in each frame following the field rotation.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern",  default='*', help='Filename pattern')
parser.add_argument('--prim_adu', action="store", dest="prim_adu", type=int, required=True, help='Flux of the primary star')
parser.add_argument('--no_info', dest='info', action='store_const',
				   const=False, default=True, help='Ignore info files')
parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
parser.add_argument('--info_pattern', action="store", dest="info_pattern", default='all_info', help='Info filename pattern')
parser.add_argument('--deltamag', dest='dm', type=float, nargs='+',required=True,
                    help='Fake companion magnitude differences')
parser.add_argument('--sepvect', dest='sep_vect', type=float, nargs='+',required=True,
                    help='Fake companion separations')
parser.add_argument('--noise', dest='noise', action='store_const',
				   const=False, default=True,
				   help='Switch to introduce additional photon noise')
parser.add_argument('-s', dest='stat', action='store_const',
				   const=True, default=False,
				   help='Print benchmarking statistics')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('-nici', dest='nici', action='store_const',
				   const=True, default=False,
				   help='Switch for GEMINI/NICI data')
parser.add_argument('-nofit', dest='fit', action='store_const',
				   const=False, default=True,
				   help='Do not use PSF fitting values.')

args = parser.parse_args()
d=args.d
pattern=args.pattern
## filter_size=args.size
info_dir=args.info_dir
info_pattern=args.info_pattern
stat=args.stat
log_file=args.log_file
nici=args.nici
prim_adu=args.prim_adu
dm=args.dm
sep_vect=args.sep_vect
fit=args.fit

skipped=0

t_init=MPI.Wtime()
## target_pattern="FP_"
## target_pattern="FP_dm"+re.sub('[\[\] ]','_',str(dm))+"_sv_"+re.sub('[\[\] ]','_',str(sep_vect))
target_pattern='FP_'+str(__version__)+'.'+str(__subversion__)+'_s_{smin:.3G}_{smax:.3G}_dm_{mmin:.3G}_{mmax:.3G}'.format(smin=np.min(sep_vect), smax=np.max(sep_vect),mmin=np.min(dm), mmax=np.max(dm))

header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y',
	'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y',
	'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
	'frame_num', 'frame_time', 'paralactic_angle']

if rank==0:
	graphic_lib_320.print_init()

	dirlist=graphic_lib_320.create_dirlist(pattern,target_pattern=target_pattern+"_")
	complete_dirlist=dirlist
	if dirlist==None:
		print("No files found. Check --pattern option!")
		for n in range(nprocs-1):
			comm.send(None,dest =n+1)
		sys.exit(1)
	else:
		start,dirlist=graphic_lib_320.send_dirlist(dirlist)


	if args.info:
		## infolist=glob.glob(info_dir+os.sep+info_pattern+'*.'+info_type)
		## infolist.sort() # Sort the list alphabetically
		infolist=graphic_lib_320.create_dirlist(info_dir+os.sep+info_pattern,extension='.rdb')
		if infolist==None:
			print("No info files found, check your --info_pattern and --info_dir options, or use -noinfo")
			comm.bcast(None, root=0)
			sys.exit(1)
		cube_list, complete_dirlist=graphic_lib_320.create_megatable(complete_dirlist,infolist,keys=header_keys,nici=nici,fit=fit)
		comm.bcast(cube_list, root=0)

	## # Create directory to store reduced data
	## if not os.path.isdir(target_dir):
		## os.mkdir(target_dir)

if not rank==0:

	## start=comm.bcast(None, root=0)
	dirlist=comm.recv(source=0)
	if dirlist==None:
		sys.exit(1)
	start=comm.recv(source = 0)

	if args.info:
		cube_list=comm.bcast(None, root=0)
		if cube_list==None:
			sys.exit(1)


t0=MPI.Wtime()


for i in range(len(dirlist)):
	targetfile=target_pattern+"_"+dirlist[i]

	##################################################################
	#
	# Read cube header and data
	#
	##################################################################

	print(str(rank)+': ['+str(start+i)+'/'+str(len(dirlist)+start)+"] "+dirlist[i]+" Remaining time: "+graphic_lib_320.humanize_time((MPI.Wtime()-t0)*(len(dirlist)-i)/(i+1-skipped)))
	## cube,header=pyfits.getdata(dirlist[i], header=True)
	hdulist = fits.open(dirlist[i])
	hdulist.verify(option='fix')
	header=hdulist[0].header
	## header.set('ESO DET CHIP PXSPACE', '{0:4G}'.format(header['ESO DET CHIP PXSPACE']))
	cube=hdulist[0].data

	if args.info:
		all_info=cube_list['info'][cube_list['cube_filename'].index(dirlist[i])]

	for frame in range(cube.shape[0]):
		if not args.info:
			print("Debug: no info")
			if nici:
				cube[frame]=graphic_lib_320.inject_FP_nici(cube[frame], sep_vect, prim_adu, dm, header, alpha=0, x0=0,y0=0, r_tel_prim=8.2, r_tel_sec=1.2, noise=args.noise, pad=2)
			else:
				cube[frame]=graphic_lib_320.inject_FP(cube[frame], sep_vect, prim_adu, dm, header, alpha=0, x0=0,y0=0, r_tel_prim=8.2, r_tel_sec=1.2, noise=args.noise, pad=2)
		elif all_info[frame,5]==-1:
			continue
		if nici:
			cube[frame]=graphic_lib_320.inject_FP_nici(cube[frame], sep_vect, prim_adu, dm, header, alpha=all_info[frame,11], x0=all_info[frame,4],y0=all_info[frame,5], r_tel_prim=8.2, r_tel_sec=1.0, noise=args.noise)
		else:
			## cube[frame]=graphic_lib_320.inject_FP(cube[frame], sep_vect, prim_adu, dm, header, alpha=all_info[frame,11], x0=all_info[frame,4]-cube.shape[1]/2,y0=all_info[frame,5]-cube.shape[2]/2, r_tel_prim=8.2, r_tel_sec=1.16, noise=args.noise)
			cube[frame]=graphic_lib_320.inject_FP(cube[frame], sep_vect, prim_adu, dm, header, alpha=all_info[frame,11], x0=all_info[frame,4],y0=all_info[frame,5], r_tel_prim=8.2, r_tel_sec=1.16, noise=args.noise, pad=2)

	## hdr["HIERARCH GC FP FIT"]=(fit.astype(int), "")
	header["HIERARCH GC FP FIT"]=(fit.numerator,'')
	header["HIERARCH GC PRIM ADU"]=(prim_adu, "Primary star ADU")
	header["HIERARCH GC MAG"]=(' '.join('%3G' %v for v in dm), "FP mag")
	header["HIERARCH GC SEPs"]=(' '.join('%3G' %v for v in sep_vect), "FP sep")
	header["HIERARCH GC FP VERS"]=(__version__+'.'+__subversion__, "")

	#Vector can be retrieved using following command
	#a=[float(s) for s in string.split(hdr['DM'][2:-1],' ')]

	## header.add_history(fnmatch.filter(sky_header.get_history(),"sky.median*" )[0])
	graphic_lib_320.save_fits( targetfile, hdulist )

print(str(rank)+": Total time: "+graphic_lib_320.humanize_time((MPI.Wtime()-t0)))

if rank==0:
	if 'ESO OBS TARG NAME' in header.keys():
		log_file=log_file+"_"+string.replace(header['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
	else:
		log_file=log_file+"_"+string.replace(header['OBJECT'],' ','')+"_"+str(__version__)+".log"

	graphic_lib_320.write_log((MPI.Wtime()-t_init),log_file, comments=None)
sys.exit(0)
