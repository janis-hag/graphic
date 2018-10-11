#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline
for High-contrast Imaging of planetary Companions".

Its purpose is to look at the statistics of the frames within a cube and reject
those that fall outside some bounds.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""
import glob
import os
import sys
import time
import numpy as np
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
import argparse

__version__ = '3.3'
__subversion__ = '0'

parser = argparse.ArgumentParser(
        description='Creates cubes with less frames by median-combining frames.')
parser.add_argument('--debug', action="store", dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern", default='*',
                    help='Filename pattern')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",
                    required=True, help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",
                    default='cube-info', help='Info directory')
parser.add_argument('--log_file', action="store", dest="log_file",
                    default='GRAPHIC', help='Log filename')
parser.add_argument('--centering_nsigma', action="store",
                    dest="centering_nsigma", type=float, default=5, help="\
                    Number of sigma used for rejection based on measured \
                    center position")
parser.add_argument('--flux_nsigma', action="store", dest="flux_nsigma",
                    type=float, default=5, help="Number of sigma used for \
                    rejection based on measured flux")
parser.add_argument('--psf_width_nsigma', action="store",
                    dest="psf_width_nsigma", type=float, default=5, help="\
                    Number of sigma used for rejection based on measured psf \
                    width")
parser.add_argument('-nici', dest='nici', action='store_const', const=True,
                    default=False, help='Switch for GEMINI/NICI data')
parser.add_argument('-nofit', dest='fit', action='store_const', const=False,
                    default=True, help='Do not use PSF fitting values.')
parser.add_argument('-dithered', dest='dithered', action='store_const',
                    const=True, default=False, help='Switch for dithered \
                    data, so that centring rejection is performed \
                    cube-by-cube instead of globally.')

args = parser.parse_args()
# Some options:
d = args.d
pattern = args.pattern
info_pattern = args.info_pattern
info_dir = args.info_dir
log_file = args.log_file
centering_nsigma = args.centering_nsigma
flux_nsigma = args.flux_nsigma
psf_width_nsigma = args.psf_width_nsigma
nici = args.nici
fit = args.fit
dithered = args.dithered

header_keys = ['frame_number', 'psf_barycentre_x', 'psf_barycentre_y',
               'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y',
               'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
               'frame_num', 'frame_time', 'paralactic_angle']

target_dir = "."

# Test the graphic frame rejection code:
print(sys.argv[0] + ' started on ' + time.strftime("%c"))
hdr = None

dirlist = graphic_nompi_lib.create_dirlist(pattern)

infolist = glob.glob(info_dir+os.sep+info_pattern+'*.rdb')
infolist.sort() # Sort the list alphabetically


cube_list, dirlist = graphic_nompi_lib.create_megatable(
        dirlist, infolist, keys=header_keys, nici=nici, fit=fit)

ncubes = len(dirlist)

# Loop through the cubes and stack the values so we can compare them all
valid_frames = []
xcens = []
ycens = []
fluxes = []
xwidths = []
ywidths = []
cube_frames = []
frame_ix = 0

for cube_ix in range(ncubes):

    # This containts of all the information for this cube
    cube_info = cube_list['info'][cube_ix]

    # How many frames are there in this cube:
    nframes = cube_info.shape[0]

    # Make an array to keep track of the frames we want to keep
    # Ignore already ignored frames (which should have columns 1-8 set to 1)
    valid_frames.extend(cube_info[:, 1] != -1)

    # Record the information
    # First, get the centre positions
    if fit:
        xcen = cube_info[:, 4]
        ycen = cube_info[:, 5]
    else:
        xcen = cube_info[:, 1]
        ycen = cube_info[:, 2]

    xcens.extend(xcen)
    ycens.extend(ycen)

    # Fluxes
    fluxes.extend(cube_info[:, 6])

    # PSF widths
    xwidths.extend(cube_info[:, 7])
    ywidths.extend(cube_info[:, 8])

    # And record the indices of the start and end of this cube
    cube_frames.append(np.arange(frame_ix, frame_ix+nframes))
    frame_ix += nframes

# Make everything into numpy arrays ?
valid_frames = np.array(valid_frames)
xcens = np.array(xcens)
ycens = np.array(ycens)
fluxes = np.array(fluxes)
xwidths = np.array(xwidths)
ywidths = np.array(ywidths)

# Make some arrays to track the invalid frames
n_invalid = np.zeros((ncubes, 4))

# Count the initially invalid frames
n_invalid[:, 0] = [np.sum(valid_frames[these_frames] == False) for these_frames in cube_frames]
# Now make the cuts
#################
# SELECTION 1 : Centering
#################

# Unfortunately if the data is dithered we have to do it cube-by-cube
if dithered:
    for cube_ix in range(ncubes):

        # Calculate the scatter in the centre positions and
        # ignore anything far away
        xcen_sigma = np.median(np.abs(xcens[cube_frames[cube_ix]] - np.median(
                xcens[cube_frames[cube_ix]])))
        ycen_sigma = np.median(np.abs(ycens[cube_frames[cube_ix]] - np.median(
                ycens[cube_frames[cube_ix]])))

        xcen_diff = np.abs(xcens[cube_frames[cube_ix]] - np.median(
                xcens[cube_frames[cube_ix]]))
        ycen_diff = np.abs(ycens[cube_frames[cube_ix]] - np.median(
                ycens[cube_frames[cube_ix]]))

        valid_frames[cube_frames[cube_ix][xcen_diff >
                     (centering_nsigma*xcen_sigma)]] = False
        valid_frames[cube_frames[cube_ix][ycen_diff >
                     (centering_nsigma*ycen_sigma)]] = False
else:

    # Calculate the scatter in the centre positions and ignore
    # anything far away
    xcen_sigma = np.median(np.abs(xcens-np.median(xcens)))
    ycen_sigma = np.median(np.abs(ycens-np.median(ycens)))

    xcen_diff = np.abs(xcens-np.median(xcens))
    ycen_diff = np.abs(ycens-np.median(ycens))

    valid_frames[xcen_diff > (centering_nsigma*xcen_sigma)] = False
    valid_frames[ycen_diff > (centering_nsigma*ycen_sigma)] = False

# Count the invalid frames due to centring
# first subtract the already known ones
n_invalid[:, 1] -= np.sum(n_invalid, axis=1)
n_invalid[:, 1] += [np.sum(valid_frames[these_frames] == False) for these_frames in cube_frames]


#################
# SELECTION 2 : Flux (only for fitted data)
#################
if fit:

    # Calculate the scatter in the flux and ignore anything far away
    flux_med = np.median(fluxes)
    flux_sigma = np.median(np.abs(fluxes-flux_med))
    flux_diff = np.abs(fluxes-flux_med)
    valid_frames[flux_diff > (flux_nsigma*flux_sigma)] = False

# Count the invalid frames due to flux
n_invalid[:, 2] -= np.sum(n_invalid, axis=1) # first subtract the already known ones
n_invalid[:, 2] += [np.sum(valid_frames[these_frames] == False) for these_frames in cube_frames]

#################
# SELECTION 3 : PSF width (only for fitted data)
#################
# First, get the widths
if fit:

    # Calculate the scatter in the psf widths and ignore anything far away
    xwidth_med = np.median(xwidths)
    ywidth_med = np.median(ywidths)
    xwidth_sigma = np.median(np.abs(xwidths-xwidth_med))
    ywidth_sigma = np.median(np.abs(ywidths-np.median(ywidths)))

    xwidth_diff = np.abs(xwidths-xwidth_med)
    ywidth_diff = np.abs(ywidths-np.median(ywidths))

    valid_frames[xwidth_diff > (psf_width_nsigma*xwidth_sigma)] = False
    valid_frames[ywidth_diff > (psf_width_nsigma*ywidth_sigma)] = False

else:
    print("  Not using fitted values. Ignoring flux and psf width!")

# Count the invalid frames due to psf width
# first subtract the already known ones
n_invalid[:, 3] -= np.sum(n_invalid, axis=1)
n_invalid[:, 3] += [np.sum(valid_frames[these_frames] == False) for these_frames in cube_frames]

# Now loop through the cubes again and record the results
for cube_ix in range(ncubes):

    # Calculate how many frames were removed:
    print("  Cube:", cube_ix,": ", np.sum(n_invalid[cube_ix]),
          " invalid frames")
    print_string = "    Breakdown: (initial, centering, flux, width): "
    graphic_mpi_lib.dprint(True, print_string + str(n_invalid[cube_ix]))

    #################
    # Update the rdb file based on the frame selection
    #################
    # Load it again
    cube_info = cube_list['info'][cube_ix]
    cube_info[valid_frames[cube_frames[cube_ix]] == False, 1:] = -1
    info_filename='all_info_framesel_' + cube_list['cube_filename'][cube_ix].replace('.fits', '.rdb')
    graphic_nompi_lib.write_array2rdb(info_dir + os.sep + info_filename,
                                      cube_info, header_keys)

sys.exit(0)