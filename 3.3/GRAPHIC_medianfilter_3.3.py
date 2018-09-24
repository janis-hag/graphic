#!/usr/bin/python
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

This step run a median filter of a given size on each single frame.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""
__version__='3.3'
__subversion__='0'

## import numpy, scipy, pyfits, glob, shutil, os, sys, time, fnmatch, tables, argparse, string
import glob, os, sys, argparse, string
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
from mpi4py import MPI
from scipy import ndimage
import astropy.io.fits as pyfits

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

target_dir = "."
backup_dir = "prev"

iterations = 1
coefficient = 0.95

parser = argparse.ArgumentParser(description='Runs a median filter on each frame of the cube.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern",  default='*', help='Filename pattern')
parser.add_argument('--size', action="store", dest="size", type=int, help='Filter size')
parser.add_argument('-noinfo', dest='no_info', action='store_const',
                   const=False, default=True,
                   help='Ignore info files')
parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
## parser.add_argument('--info_type', action="store", dest="info_type",  default='rdb', help='Info directory')
parser.add_argument('--info_pattern', action="store", dest="info_pattern", default='all_info', help='Info filename pattern')
parser.add_argument('-s', dest='stat', action='store_const',
                   const=True, default=False,
                   help='Print benchmarking statistics')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
## parser.add_argument('-hdf5', dest='hdf5', action='store_const',
                   ## const=True, default=False,
                   ## help='Switch to use HDF5 tables')
parser.add_argument('-nofit', dest='fit', action='store_const',
                   const=False, default=True,
                   help='Do not use PSF fitting values.')
parser.add_argument('-nici', dest='nici', action='store_const',
                   const=True, default=False,
                   help='Switch for GEMINI/NICI data')

args = parser.parse_args()
d=args.d
pattern=args.pattern
filter_size=args.size
info_dir=args.info_dir
## info_type=args.info_type
info_pattern=args.info_pattern
stat=args.stat
log_file=args.log_file
## hdf5=args.hdf5
fit=args.fit
nici=args.nici
window_size=args.size

skipped=0

t_init=MPI.Wtime()
target_pattern="mfs"+str(window_size)

header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
    'frame_num', 'frame_time', 'paralactic_angle']

if rank==0:
    graphic_nompi_lib.print_init()

    dirlist=graphic_nompi_lib.create_dirlist(pattern,target_dir=target_dir,target_pattern=target_pattern+"_")
    if dirlist==None:
        print("No files found. Check --pattern option!")
        for n in range(nprocs-1):
            comm.send(None,dest =n+1)
        sys.exit(1)

    if args.no_info:
        infolist=glob.glob(info_dir+os.sep+info_pattern+'*.rdb')
        infolist.sort() # Sort the list alphabetically
        if len(infolist)<2:
            print("No info files found, check your --info_pattern and --info_dir options.")
            for n in range(nprocs-1):
                comm.send(None,dest =n+1)
        cube_list, dirlist=graphic_nompi_lib.create_megatable(dirlist,infolist,keys=header_keys,nici=nici,fit=fit)
        comm.bcast(cube_list, root=0)

    start,dirlist=graphic_mpi_lib.send_dirlist(dirlist)


    # Create directory to store reduced data
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

if not rank==0:
    if args.no_info:
        cube_list=comm.bcast(None, root=0)

    dirlist=comm.recv(source = 0)
    if dirlist==None:
        sys.exit(1)

    start=int(comm.recv(source = 0))


t0=MPI.Wtime()


for i in range(len(dirlist)):
    targetfile=target_pattern+"_"+dirlist[i]

    ##################################################################
    #
    # Read cube header and data
    #
    ##################################################################

    print(str(rank)+': ['+str(start+i)+'/'+str(len(dirlist)+start)+"] "+dirlist[i]+" Remaining time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)*(len(dirlist)-i)/(i+1-skipped)))
    cube,header=pyfits.getdata(dirlist[i], header=True)
    ## hdulist = fits.open(dirlist[i],memmap=True)
    ## header=hdulist[0].header
    ## cube=hdulist[0].data

    if args.no_info:
        all_info=cube_list['info'][cube_list['cube_filename'].index(dirlist[i])]

    for frame in range(cube.shape[0]):
        if not args.no_info:
            cube[frame]=cube[frame]-ndimage.filters.median_filter(cube[frame],size=(window_size,window_size),mode='reflect')
        elif all_info[frame,6]==-1:
            continue
        else:
            cube[frame]=cube[frame]-ndimage.filters.median_filter(cube[frame],size=(window_size,window_size),mode='reflect')


    header["HIERARCH GC MEDIAN FILTER SIZE"]=(window_size, "")
    header["HIERARCH GC MED_FILT"]=( __version__+'.'+__subversion__, "")

    ## header.add_history(fnmatch.filter(sky_header.get_history(),"sky.median*" )[0])
    ## graphic_nompi_lib.save_fits( targetfile,  target_dir, cube, header )
    graphic_nompi_lib.save_fits(targetfile, cube, hdr=header,backend='pyfits')
    ## graphic_nompi_lib.save_fits(targetfile, hdulist, backend='astropy', verify='fix')

print(str(rank)+": Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))

if rank==0:
    if 'ESO OBS TARG NAME' in header.keys():
        log_file=log_file+"_"+string.replace(header['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
    else:
        log_file=log_file+"_"+string.replace(header['OBJECT'],' ','')+"_"+str(__version__)+".log"

    graphic_nompi_lib.write_log((MPI.Wtime()-t_init),log_file, comments=None)
sys.exit(0)
