#!/usr/bin/env python3
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

Its purpose is to clean bad pixels from each frame in a cube list.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__ = '3.3'
__subversion__ = '0'

import os
import sys
import string
import numpy as np
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
from mpi4py import MPI
import argparse
from graphic_mpi_lib_330 import dprint
import astropy.io.fits as pyfits

nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

target_dir = "."
target_pattern = "cl_"

parser = argparse.ArgumentParser(
        description='Puts bad pixels to median value, based on darks.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern",  default='*',
                    help='Filename pattern')
parser.add_argument('--dark_pattern', action="store", dest="dark_pattern",
                    required=True, help='Darks filename pattern')
parser.add_argument('--dark_dir', action="store", dest="dark_dir",
                    required=True, help='Directory containing the darks')
parser.add_argument('--coef', action="store", dest="coef", type=float,
                    default=5,
                    help='The sigma threshold for pixels to be rejected')
parser.add_argument('--cut', action="store", dest="cut", type=float,
                    default=9999999,
                    help='cut above which a pixel will be considered hot pix')
parser.add_argument('--log_file', action="store", dest="log_file",
                    default='GRAPHIC', help='Log filename')
parser.add_argument('-s', dest='stat', action='store_const',
                    const=True, default=False,
                    help='Print benchmarking statistics')
parser.add_argument('-interactive', dest='interactive', action='store_const',
                    const=True, default=False,
                    help='Switch to set execution to interactive mode')
parser.add_argument('-bottleneck', dest='use_bottleneck', action='store_const',
                    const=True, default=False, help='Use bottleneck module'
                    + ' instead of numpy for nanmedian.')
parser.add_argument('--flat_filename', dest='flat_filename', action='store',
                    default=None, help='Name of flat field to be used.'
                    + ' If this argument is not set, the data will use'
                    + 'flat fiel for the badpixel map')
parser.add_argument('-sphere', dest='sphere', action='store_const',
                    const=True, default=False,
                    help='Switch to set sphere to sphere mode')


args = parser.parse_args()
d = args.d
pattern = args.pattern
dark_pattern = args.dark_pattern
dark_dir = args.dark_dir
coef = args.coef
cut = args.cut
log_file = args.log_file
use_bottleneck = args.use_bottleneck
sphere = args.sphere
flat_filename = args.flat_filename

if use_bottleneck:
    from bottleneck import median as median
    from bottleneck import nanmedian as nanmedian
else:
    from numpy import nanmedian
    from numpy import median as median


comments = []


def gen_badpix(sky, coef, comments, cut):
    """Create badpixel map by searching for pixels

    Input:
    -sky: an array containing a sky
    -sky_head: FITS header of the sky
    """
    global median

    med = nanmedian(sky)
    sigma = 1.4826*nanmedian(np.abs(sky-med))
    # the 1.4826 converts the Median Absolute Deviation
    # to the standard deviation for a Normal distribution

    print("Sigma: "+str(sigma)+", median: "+str(med))

    # Creates a tuple with the x-y positions of the dead pixels
    deadpix = np.where(sky < med-sigma*coef)

    # Creates a tuple with the x-y positions of the dead pixels
    hotpix = np.where(sky > med+sigma*coef)

    # negativepix=np.where(sky < 0 )

    # Warning! x and y inverted with respect to ds9
    dprint(d > 2, "sky.shape: " + str(sky.shape) + ", sky.size: " + str(sky.size))
    # if np.shape(deadpix)[1]+np.shape(negativepix)[1]+np.shape(hotpix)[1]==0:
    if np.shape(deadpix)[1] + np.shape(hotpix)[1] == 0:
        # This used to abort and leave all procs running in infinite loops.
        # But in principal this could legitimately happen, so ACC removed this section
        # c="No bad pixels found. Aborting"
        # dprint(d>2, 'No bad pixels found. Aborting!')
        # ## print(c)
        # comments.append(c)
        # for n in range(nprocs-1):
        #     comm.send(None,dest =n+1)
        # sys.exit(1)

        # Instead, how about we just return an empty list
        c = "No badpix found!"
        comments.append(c)
        print("WARNING: No bad pixels were found!")
    else:
        c = "Found "+str(np.shape(deadpix)[1]) + " = " + str(100.*np.shape(deadpix)[1]/sky.size)+"% dead, " + "and "+str(np.shape(hotpix)[1]) + " = " + str(100.*np.shape(hotpix)[1]/sky.size) + "% hot pixels."
        comments.append(c)

    badpix = tuple(np.append(np.array(deadpix), np.array(hotpix), axis=1))

    return badpix, comments


def clean_bp(badpix, cub_in):
    global nanmedian
    # (deadpix, hotpix, negativepix, cub_in):
    cub_in[:, badpix[0], badpix[1]] = np.NaN
    for f in range(cub_in.shape[0]):
        if args.interactive:
            sys.stdout.write('\r Frame '+str(f+1)+' of '+str(cub_in.shape[0]))
            sys.stdout.flush()

        for j in range(len(badpix[0])):
            y = badpix[0][j]
            x = badpix[1][j]

            if y == cube.shape[1]-1:  # At the image edge !!!
                if x == cube.shape[2]-1:  # In a corner
                    cub_in[f, y, x] = nanmedian([cub_in[f, y-1, x-1],
                                                cub_in[f, y-1, x],
                                                cub_in[f, y, x-1]])
                elif x == 0:
                    cub_in[f, y, x] = nanmedian([cub_in[f, y-1, x],
                                                cub_in[f, y-1, x+1],
                                                cub_in[f, y, x+1]])
                else:  # Along the edge
                    cub_in[f, y, x] = nanmedian([cub_in[f, y-1, x-1],
                                                cub_in[f, y-1, x],
                                                cub_in[f, y-1, x+1],
                                                cub_in[f, y, x-1],
                                                cub_in[f, y, x+1]])
            elif y == 0:  # At the image edge !!!
                if x == cube.shape[2]-1:  # In a corner
                    cub_in[f, y, x] = nanmedian([cub_in[f, y, x-1],
                                                cub_in[f, y+1, x-1],
                                                cub_in[f, y+1, x]])
                elif x == 0:  # In a corner
                    cub_in[f, y, x] = nanmedian([cub_in[f, y, x+1],
                                                cub_in[f, y+1, x],
                                                cub_in[f, y+1, x+1]])
                else:  # Along the edge
                    cub_in[f, y, x] = nanmedian([cub_in[f, y, x-1],
                                                cub_in[f, y, x+1],
                                                cub_in[f, y+1, x-1],
                                                cub_in[f, y+1, x],
                                                cub_in[f, y+1, x+1]])
            elif x == 0:  # Along the edge
                cub_in[f, y, x] = nanmedian([cub_in[f, y-1, x],
                                            cub_in[f, y-1, x+1],
                                            cub_in[f, y, x+1],
                                            cub_in[f, y+1, x],
                                            cub_in[f, y+1, x+1]])
            elif x == cub_in.shape[2]-1:  # Along the edge
                cub_in[f, y, x] = nanmedian([cub_in[f, y-1, x-1],
                                            cub_in[f, y-1, x],
                                            cub_in[f, y, x-1],
                                            cub_in[f, y+1, x-1],
                                            cub_in[f, y+1, x]])
            else:  # Usual case, not on an edge
                cub_in[f, y, x] = nanmedian([cub_in[f, y-1, x-1],
                                            cub_in[f, y-1, x],
                                            cub_in[f, y-1, x+1],
                                            cub_in[f, y, x-1],
                                            cub_in[f, y, x+1],
                                            cub_in[f, y+1, x-1],
                                            cub_in[f, y+1, x],
                                            cub_in[f, y+1, x+1]])
    return cub_in


t_init=MPI.Wtime()

if sphere:
    # for SPHERE a large part of the image are badpix because outside of the
    # filter. We apply a nan mask to the dark so there are not determined as
    # bad pixels and therefore makes the code much faster. This part of the
    # image is set to nan for the science cubes as well
    # applying nan mask on the image part where there is no filter

    x = np.arange(1024)
    y = np.copy(x)
    X, Y = np.meshgrid(x, y)
    z = np.arange(-512, 512)
    centre_filtre_left = [471, 529]
    centre_filtre_right = [1558-1024, 516]
    w = np.copy(z)
    Z, W = np.meshgrid(z, w)
    R_left = np.sqrt((Z + (512-centre_filtre_left[0]))**2
                     + (W + (512-centre_filtre_left[1]))**2)
    R_right = np.sqrt((Z + (512-centre_filtre_right[0]))**2
                      + (W + (512-centre_filtre_right[1]))**2)
    R = np.sqrt(Z**2 + W**2)
    #left part of the image
    mask_nan_l = np.where(X < 50, np.nan, 1.)
    mask_nan_l = np.where(X > 934, np.nan, mask_nan_l)
    mask_nan_l = np.where(Y < 22, np.nan, mask_nan_l)
    mask_nan_l = np.where(R > 521, np.nan, mask_nan_l)
    #right image
    mask_nan_r = np.where(X < 55, np.nan, 1.)
    mask_nan_r = np.where(X > 935, np.nan, mask_nan_r)
    mask_nan_r = np.where(Y < 12, np.nan, mask_nan_r)
    mask_nan_r = np.where(Y > 1014, np.nan, mask_nan_r)
    mask_nan_r = np.where(R > 531, np.nan, mask_nan_r)
    mask_nan = np.append(mask_nan_l, mask_nan_r, axis=1)

if rank == 0:
    graphic_nompi_lib.print_init()

    t_init = MPI.Wtime()

    print("Searching cubes...")
    dirlist = graphic_nompi_lib.create_dirlist(
            pattern, target_pattern=target_pattern)
    print("Searching reference cubes...")
    darklist = graphic_nompi_lib.create_dirlist(
            dark_dir + os.sep + dark_pattern)

    if dirlist is None or darklist is None:
        print('Missing files, leaving...')
        MPI.Finalize()
        sys.exit(1)

    start, dirlist = graphic_mpi_lib.send_dirlist(dirlist)
    dprint(d > 2, 'Dirlist sent to slaves')

    dark_cube = None
    for file_name in darklist:
        # dark_hdulist = fits.open(file_name)
        # data=dark_hdulist[0].data
        data = pyfits.getdata(file_name, header=False)
        if dark_cube is None:
            dark_cube = data   #[ np.newaxis,...]
            # print(dark_cube.shape)
        elif len(data.shape)==3:  # CUBE
            # print(data.shape)
            # dark_cube=np.concatenate((dark_cube,data[np.newaxis,...]),axis=0)
            dark_cube = np.concatenate((dark_cube, data), axis=0)
            # print(dark_cube.shape)
        elif len(data.shape) == 2: # FRAME
            if len(dark_cube.shape) == 3:
                dark_cube = np.rollaxis(dark_cube, 0, 3)
            ## print(data.shape, dark_cube.shape)
            dark_cube = np.rollaxis(np.dstack((dark_cube, data)), 2)

    if len(np.where(np.isnan(dark_cube))[0]):
        print("Found NaNs: "+str(np.where(np.isnan(dark_cube))))
        dark_cube = np.where(np.isnan(dark_cube), 0, dark_cube)

    dprint(d > 2, "dark_cube.shape " + str(dark_cube.shape))
    dark_cube = dark_cube*1.
    if len(dark_cube.shape) == 3:
        dark = median(dark_cube, axis=0)
    elif len(dark_cube.shape) == 2:
        dark = dark_cube
    del dark_cube

    if sphere:
        dark = dark*mask_nan

    bad_pix, comments = gen_badpix(dark, coef, comments, cut)

    badpix_map = np.zeros((np.shape(dark)[0], np.shape(dark)[1]))
    badpix_map[bad_pix] = 1
    hdulist = pyfits.PrimaryHDU()
    hdr_badpix = hdulist.header
    hdr_badpix['dark'] = comments[0]

    if flat_filename and sphere:
        flatfield = pyfits.getdata(flat_filename, header=False)
        flatfield = flatfield*mask_nan
        badpix_flat, comments = gen_badpix(flatfield, coef, comments, cut)
        badpix_map_flat = np.zeros((
                np.shape(flatfield)[0], np.shape(flatfield)[1]))
        badpix_map_flat[badpix_flat] = 1
        badpix_map = badpix_map + badpix_map_flat
        badpix_map = np.where(badpix_map > 1, 1, badpix_map)
        bad_pix = tuple(np.where(badpix_map == 1))
        # bad_pix=tuple(numpy.append(numpy.array(badpix),numpy.array(badpix_flat), axis=1))
        hdr_badpix['flat'] = comments[1]

    pyfits.writeto("badpixel_map.fits", badpix_map, header=hdr_badpix,
                   clobber=True)
    comm.bcast(bad_pix, root=0)

if not rank == 0:
    dirlist = comm.recv(source=0)
    if dirlist is None:
        print('Received None dirlist. Leaving...')
        bad_pix = comm.bcast(None, root=0)
        sys.exit(1)

    start = int(comm.recv(source=0))
    bad_pix = comm.bcast(None, root=0)
    dprint(d > 2, 'Received dirlist, start, and bad_pix')


t0 = MPI.Wtime()

for i in range(len(dirlist)):
    print(str(rank) + ': ['+str(start+i) + '/'+str(len(dirlist)+start-1) + "] "
          + dirlist[i] + " Remaining time: " + graphic_nompi_lib.humanize_time(
                  (MPI.Wtime()-t0)*(len(dirlist)-i)/(i+1)))

    cube, header = pyfits.getdata(dirlist[i], header=True)
    cube = clean_bp(bad_pix, cube)
    if sphere:
        cube = cube*mask_nan

    header["HIERARCH GC BAD_PIX"] = (__version__+'.'+__subversion__, "")
    graphic_nompi_lib.save_fits(target_pattern+dirlist[i],
                                cube, header=header, backend='pyfits')

if 'ESO OBS TARG NAME' in header.keys():
    log_file = log_file+"_"+header['ESO OBS TARG NAME'].replace(' ', '')
    + "_"+__version__+".log"
else:
    log_file = log_file+"_"+header['OBJECT'].replace(' ', '')
    + "_"+str__version__+".log"

print(str(rank)+": Total time: "
      + graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
graphic_nompi_lib.write_log((MPI.Wtime()-t_init), log_file, comments)
sys.exit(0)
