#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline for
High-contrast Imaging of planetary Companions".

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__ = '3.3'
__subversion__ = '0'

import numpy, scipy, glob, shutil, os, sys, time, fnmatch, argparse, string, re
from mpi4py import MPI
from scipy import ndimage
import numpy as np
import astropy.io.fits as pyfits

import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib

## sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
import bottleneck

nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

target_pattern = "ps_"
centroids_dir = "centroids"
info_dir = "cube-info"
file_prefix = "fmed_"

parser = argparse.ArgumentParser(
        description='Creates a median from all the frames.')
parser.add_argument('--debug', action="store", dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern", default='*',
                    help='Filename pattern')
parser.add_argument('--target_dir', action="store", dest="target_dir",
                    default='.', help='Reduced files target directory')
parser.add_argument('-nici', dest='nici', action='store_const', const=True,
                    default=False, help='Switch for GEMINI/NICI data')
parser.add_argument('-s', dest='stat', action='store_const', const=True,
                    default=False, help='Print benchmarking statistics')
parser.add_argument('--log_file', action="store", dest="log_file",
                    default='GRAPHIC', help='Log filename')

args = parser.parse_args()
d = args.d
pattern = args.pattern
log_file = args.log_file
target_dir = args.target_dir
nici = args.nici

if rank == 0:
    print(sys.argv[0] + ' started on ' + time.strftime("%c"))
    t_init = MPI.Wtime()
    dirlist = graphic_nompi_lib.create_dirlist(
            pattern, target_pattern=target_pattern + "_")
    complete_dirlist = dirlist
    if dirlist is None:
        print("No files found. Check --pattern option!")
        MPI.Finalize()
        sys.exit(1)

    skipped = 0
    # 0: frame_number, 1: psf_barycenter_x, 2: psf_barycenter_y, 3: psf_pixel_size,
    # 4: psf_fit_center_x, 5: psf_fit_center_y, 6: psf_fit_height, 7: psf_fit_width_x, 8: psf_fit_width_y,
    # 9: frame_number, 10: frame_time, 11: paralactic_angle

    hdr = pyfits.getheader(dirlist[0])
    med_tot = None
    median_filename = file_prefix + string.split(dirlist[0], os.sep)[-1]
    stack = None

    hdr.add_history("This median is made from the following files:")

    for c in range(len(dirlist)):
        t0_cube = MPI.Wtime()
        if args.stat == True:
            tb = MPI.Wtime()

        # Check if already processed
        if os.access(target_dir + os.sep + median_filename, os.F_OK | os.R_OK):
            print('Already processed: ' + median_filename)
            skipped = skipped + 1
            continue

        sys.stdout.write("Processing cube [" + str(c + 1) + "/" +
                         str(len(dirlist)) + "]: " + str(dirlist[c]) + "\n")
        sys.stdout.flush()
        s = 1
        restart = False
        total_frames = 0

        hdr.add_history(dirlist[c])

        ## hdulist = fits.open(dirlist[c])
        ## header.set('ESO DET CHIP PXSPACE', '{0:4G}'.format(header['ESO DET CHIP PXSPACE']))
        ## cube=hdulist[0].data
        cube = pyfits.getdata(dirlist[c])
        if stack is None:
            # Create frame stack
            stack = cube
        elif len(stack.shape
                 ) == 2:  # Actually working with frames instead of cubes
            stack = np.concatenate((stack[np.newaxis, ...], cube[np.newaxis]),
                                   axis=0)
        elif len(cube.shape) == 2:  # Add frames, not cubes
            stack = np.concatenate((stack, cube[np.newaxis]), axis=0)
        else:
            stack = np.concatenate((stack, cube), axis=0)
    if stack is None:
        print("No cubes found to generate median.")
        MPI.Finalize()
        sys.exit(0)

    if args.stat == True:
        print("\n STAT: Stack preparation took: " +
              graphic_nompi_lib.humanize_time(MPI.Wtime() - tb))
        tb = MPI.Wtime()
        t0_trans = MPI.Wtime()

    print(stack.shape)
    graphic_mpi_lib.send_chunks(stack, d)
    del stack

    if args.stat == True:
        print("\n STAT: Data upload took: " +
              graphic_nompi_lib.humanize_time(MPI.Wtime() - t0_trans))

    med = None
    t0_trans = MPI.Wtime()

    ## gather results and save
    for n in range(nprocs - 1):
        chunk = comm.recv(source=n + 1)
        ## if chunk == None:
        ##	 continue
        try:
            chunk.shape
        except:
            continue

        sys.stdout.write('\r\r\r Median processed data from ' + str(n + 1) +
                         ' received								 =>')
        sys.stdout.flush()

        if med is None:  #initialise
            med = chunk
        else:
            med = np.concatenate((med, chunk), axis=0)

    if args.stat == True:
        print("\n STAT: Data download took: " +
              graphic_nompi_lib.humanize_time(MPI.Wtime() - t0_trans))
        print("\n STAT: Median calculation took: " +
              graphic_nompi_lib.humanize_time(MPI.Wtime() - tb))

    #print("Saving: "+str(psf_sub_filename))
    ## save_fits(median_filename, target_dir, med, hdr)
    hdr["HIERARCH GC FREE_MED"] = (__version__ + '.' + __subversion__, "")
    graphic_nompi_lib.save_fits(median_filename, med, target_dir=target_dir,
                                hdr=hdr, backend='pyfits')

    sys.stdout.write(
            "\n Saved: {name} .\n Processed in {human_time} at {rate:.2f} MB/s \n"
            .format(
                    name=median_filename,
                    human_time=graphic_nompi_lib.humanize_time(MPI.Wtime() -
                                                               t0_cube),
                    rate=os.path.getsize(target_dir + os.sep + median_filename)
                    / (1048576 * (MPI.Wtime() - t0_cube))))
    sys.stdout.flush()

    print("\n Program finished, killing all the slaves...")
    print("Total time: " + str(MPI.Wtime() - t_init) + " s = " +
          graphic_nompi_lib.humanize_time((MPI.Wtime() - t_init)))

    ## if 'ESO OBS TARG NAME' in hdr.keys():
    ## log_file=log_file+"_"+string.replace(hdr['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
    ## else:
    ## log_file=log_file+"_"+string.replace(hdr['OBJECT'],' ','')+"_"+str(__version__)+".log"
    ## graphic_lib_310.write_log((MPI.Wtime()-t_init),log_file)

    graphic_nompi_lib.write_log_hdr((MPI.Wtime() - t_init), log_file, hdr,
                                    comments=None, nprocs=nprocs)

    print("Program finished, killing all the slaves...")
    for n in range(nprocs - 1):
        comm.send(None, dest=n + 1)
        comm.send(None, dest=n + 1)
    sys.exit(0)

#################################################################################
#
# SLAVES
#
# slaves need to:
# receive stack and frame
# calculate median
else:

    # Receive number of first column
    start_col = comm.recv(source=0)
    # Receive stack to median
    stack = comm.recv(source=0)

    while not stack is None:
        if stack is None:
            comm.send(None, dest=0)
            continue
        if d > 5:
            print("")
            print(str(rank) + " stack.shape: " + str(stack.shape))
        # Mask out the NaNs
        stack = bottleneck.nanmedian(stack, axis=0)
        comm.send(stack, dest=0)
        # Receive number of first column
        start_col = comm.recv(source=0)
        # Receive stack to median
        stack = comm.recv(source=0)
