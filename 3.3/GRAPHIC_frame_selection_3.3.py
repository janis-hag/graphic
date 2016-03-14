#!/usr/bin/python2.6
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline for
High-contrast Imaging of planetary Companions".

Its purpose is to look at the statistics of the frames within a cube and reject those that fall outside some bounds.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__='3.3'
__subversion__='0'

import numpy, scipy, glob,  os, sys, subprocess, string, time
import numpy as np
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
from scipy import ndimage
from mpi4py import MPI
import argparse
from graphic_mpi_lib_330 import dprint
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
parser.add_argument('--centering_nsigma',action="store", dest="centering_nsigma", type=float, default=5,help="Number of sigma used for rejection based on measured center position")
parser.add_argument('--flux_nsigma',action="store", dest="flux_nsigma", type=float, default=5,help="Number of sigma used for rejection based on measured flux")
parser.add_argument('--psf_width_nsigma',action="store", dest="psf_width_nsigma", type=float, default=5,help="Number of sigma used for rejection based on measured psf width")
parser.add_argument('-nici', dest='nici', action='store_const',const=True, default=False,help='Switch for GEMINI/NICI data')
parser.add_argument('-nofit', dest='fit', action='store_const',const=False, default=True,help='Do not use PSF fitting values.')

args = parser.parse_args()
d=args.d
pattern=args.pattern
info_pattern=args.info_pattern
info_dir=args.info_dir
log_file=args.log_file
centering_nsigma=args.centering_nsigma
flux_nsigma=args.flux_nsigma
psf_width_nsigma=args.psf_width_nsigma
nici=args.nici
fit=args.fit

header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
	'frame_num', 'frame_time', 'paralactic_angle']

target_dir = "."

if rank==0:
    print(sys.argv[0]+' started on '+ time.strftime("%c"))
    hdr=None

    dirlist=graphic_nompi_lib.create_dirlist(pattern)

    infolist=glob.glob(info_dir+os.sep+'*'+info_pattern+'*.rdb')
    infolist.sort() # Sort the list alphabetically


    cube_list,dirlist=graphic_nompi_lib.create_megatable(dirlist,infolist,keys=header_keys,nici=nici,fit=fit)

    ncubes=len(dirlist)    

    print("Performing frame selection on input cubes.")    
        
    # Loop through the cubes:
    for cube_ix in range(ncubes):
    	
        # This containts of all the information for this cube        
        cube_info=cube_list['info'][cube_ix]
        
        # How many frames are there in this cube:
        nframes=cube_info.shape[0]
        
        # Make an array to keep track of the frames we want to keep
        # Ignore already ignored frames (which should have columns 1-8 set to 1)
        valid_frames=cube_info[:,1] != -1
        n_invalid_initial=np.sum(valid_frames==False)


        #################
        # SELECTION 1 : Centering
        #################

        # First, get the centre positions
        if fit:
            xcen=cube_info[:,4]
            ycen=cube_info[:,5]
        else:
            xcen=cube_info[:,1]
            ycen=cube_info[:,2]
        
        # Calculate the scatter in the centre positions and ignore anything far away
        xcen_sigma=np.median(np.abs(xcen-np.median(xcen)))
        ycen_sigma=np.median(np.abs(ycen-np.median(ycen)))
        
        xcen_diff=np.abs(xcen-np.median(xcen))
        ycen_diff=np.abs(ycen-np.median(ycen))
        
        valid_frames[xcen_diff > (centering_nsigma*xcen_sigma)]=False
        valid_frames[ycen_diff > (centering_nsigma*ycen_sigma)]=False
        
        n_invalid_centering=np.sum(valid_frames==False)-n_invalid_initial
        
        #################
        # SELECTION 2 : Flux (only for fitted data)
        #################
        if fit:
            flux=cube_info[:,6]
            
            # Calculate the scatter in the flux and ignore anything far away
            flux_sigma=np.median(np.abs(flux-np.median(flux)))
            flux_diff=np.abs(flux-np.median(flux))
            valid_frames[flux_diff > (flux_nsigma*flux_sigma)]=False
            
        n_invalid_flux=np.sum(valid_frames==False)-(n_invalid_initial+n_invalid_centering)
        
        #################
        # SELECTION 3 : PSF width (only for fitted data)
        #################
        # First, get the widths
        if fit:
            xwidth=cube_info[:,4]
            ywidth=cube_info[:,5]
        
            # Calculate the scatter in the psf widths and ignore anything far away
            xwidth_sigma=np.median(np.abs(xwidth-np.median(xwidth)))
            ywidth_sigma=np.median(np.abs(ywidth-np.median(ywidth)))
            
            xwidth_diff=np.abs(xwidth-np.median(xwidth))
            ywidth_diff=np.abs(ywidth-np.median(ywidth))
            
            valid_frames[xwidth_diff > (psf_width_nsigma*xwidth_sigma)]=False
            valid_frames[ywidth_diff > (psf_width_nsigma*ywidth_sigma)]=False
        else:
            print "Not using fitted values. Ignoring flux and psf width!"
            
        n_invalid_psf_width=np.sum(valid_frames==False)-(n_invalid_initial+n_invalid_centering+n_invalid_flux)
        
        # Calculate how many frames were removed:
        n_invalid=np.sum(valid_frames==False)
        print "Cube:",cube_ix,": ",n_invalid," invalid frames"
        graphic_mpi_lib.dprint(d>2,"    Breakdown: (initial, centering, flux, width): ("+str(n_invalid_initial)+","+str(n_invalid_centering)+","+str(n_invalid_flux)+","+str(n_invalid_psf_width)+")")
        
        #################
        # Update the rdb file based on the frame selection
        #################
        cube_info[valid_frames==False,1:8]=-1
        info_filename='all_info_framesel_'+cube_list['cube_filename'][cube_ix].replace('.fits','.rdb')
        graphic_nompi_lib.write_array2rdb(info_dir+os.sep+info_filename,cube_info,header_keys)