#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

Its purpose is to create a list of frames that should be kept.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""
import sys
import numpy as np
import glob
import argparse
import astropy.io.fits as pyfits

__version__ = '3.3'
__subversion__ = '0'

target_dir = "."
parser = argparse.ArgumentParser(description='Supress the frame in the cubes \
                                 and rdb files from the selection_frame file')
parser.add_argument(
        '--pattern', action="store", dest="pattern", default="sdi",
        help='Filename pattern of the cubes to apply the frame \
                    selections')
parser.add_argument(
        '--r_int', action="store", dest="r_int", default=30, type=float,
        help='The inner radius used to consider the \
                    speckle statistics')
parser.add_argument(
        '--r_ext', action="store", dest="r_ext", default=80, type=float,
        help='The outer radius used to consider the \
                    speckle statistics')
parser.add_argument(
        '--sigma', action="store", dest="sigma", default=5, type=float,
        help='The number of sigma used for the cut \
                    based on the median flux in each frame')
args = parser.parse_args()
pattern = args.pattern
r_int = args.r_int
r_ext = args.r_ext
sigma = args.sigma


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variabililty of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """

    # should be faster to not use masked arrays.
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def frame_selection(filename, r_int=30, r_ext=80, sigma=5):
    '''
    Identifies bad frames by comparing the statistics of speckles in each
    quadrant between radii r_int and r_ext (both in pix).

    '''
    cube, hdr = pyfits.getdata(filename, header=True)

    # Cut out a region around the centre
    cut = np.min([r_ext, cube.shape[1] // 2, cube.shape[2] // 2])
    cube = cube[:,
                np.shape(cube)[1] // 2 - cut:np.shape(cube)[1] // 2 + cut,
                np.shape(cube)[1] // 2 - cut:np.shape(cube)[1] // 2 + cut]

    med = np.nanmedian(cube, axis=0)
    cube = cube - med + np.median(med)

    # Make a donut to cut out a region around the centre
    x = np.arange(-np.shape(cube)[1] // 2, np.shape(cube)[1] // 2)
    y = np.arange(-np.shape(cube)[1] // 2, np.shape(cube)[1] // 2)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    donut = np.where(R < r_int, np.nan, 1)
    donut = np.where(R > r_ext, np.nan, donut)

    cube = cube * donut

    # Separate the 4 quadrants
    quad1 = cube[:, :np.shape(cube)[1] // 2, :np.shape(cube)[2] // 2]
    quad2 = cube[:, np.shape(cube)[1] // 2:, :np.shape(cube)[2] // 2]
    quad3 = cube[:, :np.shape(cube)[1] // 2, np.shape(cube)[2] // 2:]
    quad4 = cube[:, np.shape(cube)[1] // 2:, np.shape(cube)[2] // 2:]

    med_cube = np.nanmedian(cube, axis=(1, 2))

    # Calculate the total flux in each quadrant
    sum1 = np.nansum(quad1, axis=(1, 2))
    sum2 = np.nansum(quad2, axis=(1, 2))
    sum3 = np.nansum(quad3, axis=(1, 2))
    sum4 = np.nansum(quad4, axis=(1, 2))

    # Loop through the frames and compare the quadrants to see if they are bad
    bad_frames = []
    for i in range(np.shape(cube)[0]):
        # Find which quadrant has the minimum total flux to use as a comparison
        index_quad_min = np.where(
                np.array([sum1[i], sum2[i], sum3[i], sum4[i]]) == np.min(
                        [sum1[i], sum2[i], sum3[i], sum4[i]]))[0][0]
        if index_quad_min == 0:
            quad_min = quad1[i]
        elif index_quad_min == 1:
            quad_min = quad2[i]
        elif index_quad_min == 2:
            quad_min = quad3[i]
        elif index_quad_min == 3:
            quad_min = quad4[i]
        mean_quad_min = np.nanmean(quad_min)
        sum_quad_mean = np.nansum(quad_min)
        std_quad_min = np.nanstd(quad_min)

        #  A frame is also bad if the median flux is sigma above the median
        #  absolute deviation
        if med_cube[i] > (np.median(med_cube) + sigma * mad(med_cube)):
            bad_frames.append(i)
            continue  # if it's bad, don't bother checking the quadrants

        # Loop through the quadrants to see if any are bad
        for quad in [quad1, quad2, quad3, quad4]:

            highflux_amount = np.nansum(
                    np.where(quad[i] > mean_quad_min + 3 * std_quad_min,
                             quad[i], np.nan))

            # A frame is bad if > 10% of the flux is located in pixels that are
            # 3 sigma above the mean
            if highflux_amount > 0.1 * sum_quad_mean:
                bad_frames.append(i)
                # if one is bad, we don't need to check the rest
                break

        # This was the old code.
#        if np.nansum(np.where(quad1[i]>mean_quad_min+3*std_quad_min,quad1[i],np.nan))>0.1*sum_quad_mean:
#            bad_frames=bad_frames+[int(i+1)]
#        elif np.nansum(np.where(quad2[i]>mean_quad_min+3*std_quad_min,quad2[i],np.nan))>0.1*sum_quad_mean:
#            bad_frames=bad_frames+[int(i+1)]
#        elif np.nansum(np.where(quad3[i]>mean_quad_min+3*std_quad_min,quad3[i],np.nan))>0.1*sum_quad_mean:
#            bad_frames=bad_frames+[int(i+1)]
#        elif np.nansum(np.where(quad4[i]>mean_quad_min+3*std_quad_min,quad4[i],np.nan))>0.1*sum_quad_mean:
#            bad_frames=bad_frames+[int(i+1)]
#        elif med_cube[i]>np.median(med_cube)+(sigma+2)*mad(med_cube):
#            bad_frames=bad_frames+[int(i+1)]

    print('Bad frames:' + str(bad_frames))
    return bad_frames


# pattern="left_cl_nomed*SCIENCE"
#length = 0
#
#for allfiles in glob.iglob(pattern+'*'):
#    sys.stdout.write(allfiles)
#    sys.stdout.write('\n')
#    #sys.stdout.write(allfiles.replace('left','right'))
#    #sys.stdout.write('\n')
#    length += 1
#sys.stdout.flush()

bad_frames_cube = []

for allfiles in glob.iglob(pattern + '*'):
    print(allfiles)
    bad_frames_cube.append(
            frame_selection(allfiles, r_int=r_int, r_ext=r_ext, sigma=sigma))

f = open('frame_selection.txt', 'w')
f.write("filename\tframe_to_delete\n")
f.write("--------\t---------------\n")
for i, allfiles in enumerate(glob.iglob(pattern + '*')):
    f.write(allfiles + '\t' + str(bad_frames_cube[i]) + '\n')
f.close()

sys.exit(0)
