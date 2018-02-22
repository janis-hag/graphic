#!/usr/bin/python
import numpy as np
import scipy, time, glob, sys, os, shutil
from scipy import fftpack
import pylab as py
import pyfftw
import multiprocessing
import astropy.io.fits as pyfits
from mpi4py import MPI
import graphic_nompi_lib_330 as graphic_nompi_lib
import argparse

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Apply the correction of distortion only for IRDIS data cubes')
parser.add_argument('--pattern', action="store", dest="pattern",  default="cl_nomed", help='cubes to apply the correction of distortion')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",default='all_info', help='Info filename pattern.')
parser.add_argument('--info_dir', action="store", dest="info_dir",default='cube-info', help='Info directory')
parser.add_argument('--distortion_factor', action="store", dest="distortion_factor",  default=1.006, type=float, help='Distortion factor to be applied to correct from distortion.')
parser.add_argument('-ifs', action="store_const", dest="ifs",  const=True, default=False, help='Switch for IFS data')
parser.add_argument('--ifs_lowpass_r', action="store", dest="ifs_lowpass_r",default=5, type=float, help='R value for the Butterworth low pass filter applied to IFS data')
parser.add_argument('--ifs_lowpass_cutoff', action="store", dest="ifs_lowpass_cutoff",default=120, type=float, help='Cutoff value for the Butterworth low pass filter applied to IFS data')
parser.add_argument('--ifs_angle', action="store", dest="ifs_angle",default=100.48, type=float, help='Rotation between IFS and IRDIS field of view. ')

args = parser.parse_args()
pattern = args.pattern
info_pattern = args.info_pattern
info_dir = args.info_dir
distortion_factor=args.distortion_factor
ifs = args.ifs
ifs_lowpass_r = args.ifs_lowpass_r
ifs_lowpass_cutoff = args.ifs_lowpass_cutoff
ifs_angle = args.ifs_angle

if rank==0:

    t0=MPI.Wtime()
    print("beginning of Distortion correction:")

    length=0

    sys.stdout.write('Application de la correction de la distortion sur:')
    sys.stdout.write('\n')
    #for allfiles in glob.iglob(key_word):
    for allfiles in glob.iglob(pattern+'*'):
        sys.stdout.write(allfiles)
        sys.stdout.write('\n')
        sys.stdout.write(allfiles.replace('left','right'))
        sys.stdout.write('\n')
        length+=1
    sys.stdout.flush()


    # Loop through the cubes and correct distortion
    for count,allfiles in enumerate(glob.iglob(pattern+'*')):
        sys.stdout.write('\n')
        sys.stdout.write('\r Cube ' + str(count+1) + '/' + str(length))
        sys.stdout.flush()

        # Load the cubes
        cube,hdr=pyfits.getdata(allfiles,header=True)
        orig_shape = cube.shape
        
        #correction factor to rescale cubes
        correction_factor_x=1.
        correction_factor_y=distortion_factor #the y direction is stretched by a factor distortion_factor so we compress this direction to compensate.

        if ifs:
            # Apply a light low-pass filter to remove any spikes that would cause problems with the rotation
            cube = graphic_nompi_lib.low_pass(cube,ifs_lowpass_r,2,ifs_lowpass_cutoff)   
            
            # We need to rotate the images to the same orientation as IRDIS first
            derot_cube = []
            for ix,frame in enumerate(cube):
                derot_cube.append(graphic_nompi_lib.fft_3shear_rotate_pad(cube[ix],ifs_angle,pad=2,return_full=False))
        
            cube = np.array(derot_cube)
            # Now remove the extra rows and columns that were added during the derotation
            if orig_shape[-1] != cube.shape[-1]:
                cube = cube[...,1:-1,1:-1]
            
        # Rescale all of the cubes to the correction_factor_x and correction_factor_y
        rescaled_cube = graphic_nompi_lib.rescale_image(cube,correction_factor_x,correction_factor_y)
        

        if ifs:        
            # We need to rotate the images to the same orientation as IRDIS first
            derot_cube = []
            for ix,frame in enumerate(rescaled_cube):
                derot_cube.append(graphic_nompi_lib.fft_3shear_rotate_pad(rescaled_cube[ix],-ifs_angle,pad=2,return_full=False))
            
            rescaled_cube = np.array(derot_cube)
            # Now remove the extra rows and columns that were added during the derotation
            if orig_shape[-1] != rescaled_cube.shape[-1]:
                rescaled_cube = rescaled_cube[...,1:-1,1:-1]


        # Write it out
        pyfits.writeto("dis_"+allfiles,rescaled_cube,header=hdr,clobber=True)
        

    # Copy the info files
    for allfiles in glob.iglob(info_dir+'/'+info_pattern+'*'):
        shutil.copyfile(allfiles,allfiles.replace(pattern,"dis_"+pattern))

    sys.stdout.write('\n')
    print("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
    print("Correction of distortion finished")
    sys.exit(0)
    ##sys.exit(0)
else:
    sys.exit(0)
