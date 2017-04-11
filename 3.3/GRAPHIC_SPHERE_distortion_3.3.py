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
parser.add_argument('--distortion_factor', action="store", dest="distortion_factor",  default=1.006, type=int, help='Distortion factor to be applied to correct from distortion.')


args = parser.parse_args()
pattern = args.pattern
info_pattern = args.info_pattern
info_dir = args.info_dir
distortion_factor=args.distortion_factor




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

    count=1

    # Loop through the cubes and correct distortion
    for allfiles in glob.iglob(pattern+'*'):
        sys.stdout.write('\n')
        sys.stdout.write('\r Cube ' + str(count) + '/' + str(length))
        sys.stdout.flush()

        # Load the cubes
        cube,hdr=pyfits.getdata(allfiles,header=True)
        
        #correction factor to rescale cubes
        correction_factor_x=1.
        correction_factor_y=1./distortion_factor #the y direction is stretched by a factor distortion_factor so we compress this direction to compensate.
        

        # Rescale all of the cubes to the correction_factor_x and correction_factor_y
        rescaled_cube = graphic_nompi_lib.rescale_image(cube,correction_factor_x,correction_factor_y)

        # Write it out
        pyfits.writeto("dis_"+allfiles,rescaled_cube,header=hdr,clobber=True)
        
        count+=1

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
