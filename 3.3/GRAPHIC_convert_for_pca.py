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

import numpy as np
import os
import sys
import shutil
import astropy.io.fits as pyfits
import graphic_nompi_lib_330 as graphic_nompi_lib
import argparse
import time

__version__ = '3.3'
__subversion__ = '0'

target_dir = "."
parser = argparse.ArgumentParser(
        description='Converts images to the correct format for the PCA code')
parser.add_argument('--pattern', action="store", dest="pattern",
                    default="cl_nomed_SPHER*STAR_CENTER",
                    help='cubes to apply the star centering.')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",
                    default='all*', help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",
                    default='cube-info', help='Info directory')
parser.add_argument('-sphere', dest='sphere', action='store_const', const=True,
                    default=False, help='Switch for VLT/SPHERE data')
parser.add_argument('-scexao', dest='scexao', action='store_const', const=True,
                    default=False, help='Switch for Subaru/SCExAO data')
parser.add_argument(
        '--output_dir', action="store", dest="output_dir", default="./",
        help='output directory for the cube and parallactic angle \
                    file.')
parser.add_argument(
        '--output_file', action="store", dest='output_file',
        default="master_cube_PCA.fits",
        help='Filename of the output fits file containing the \
                    stacked cube.')
parser.add_argument('-skip_parang', action='store_const', dest='skip_parang',
                    const=True, default=False,
                    help='Skip the generation of the parallactic angle file.')
parser.add_argument(
        '-collapse_cube', action='store_const', dest='collapse_cube',
        const=True, default=False,
        help='Collapse the image cube into a single frame \
                    (used to make the PSF frame).')

args = parser.parse_args()
pattern = args.pattern
info_pattern = args.info_pattern
info_dir = args.info_dir
output_dir = args.output_dir
output_file = args.output_file
skip_parang = args.skip_parang
collapse_cube = args.collapse_cube

# Initialising timer
t0 = time.time()

print(sys.argv[0] + ' started on ' + time.strftime("%c"))
print("Beginning of convert")

#print('  ' + allfiles)

dirlist = graphic_nompi_lib.create_dirlist(pattern)
if dirlist is None:
    print('Leaving without processing.')
    sys.exit(1)

# Check that the output directory exists, and make it if needed
if not output_dir.endswith(os.sep):
    output_dir += os.sep
dir_exists = os.access(output_dir, os.F_OK)
if not dir_exists:
    os.mkdir(output_dir)

infolist = graphic_nompi_lib.create_dirlist(info_dir + os.sep + info_pattern,
                                            extension='.rdb')

cube_list, dirlist, pd_cubelist = graphic_nompi_lib.create_megatable(
        dirlist, infolist, skipped=None, keys=graphic_nompi_lib.header_keys(),
        nici=False, sphere=args.sphere, scexao=args.scexao, fit=False,
        nonan=True, interactive=False, return_pandas=True)

# Loop through the files and combine them into a single cube
for ix, allfiles in enumerate(dirlist):
    if ix == 0:
        master_cube, hdr = pyfits.getdata(allfiles, header=True)
    else:
        cube_temp = pyfits.getdata(allfiles)
        master_cube = np.append(master_cube, cube_temp, axis=0)

# part below rewritten using infolist and pandas
##############################################################################
#    # Read in the cube info file to get the parallactic angles
#    with open(glob.glob("cube-info/*"+allfiles.replace(".fits", ".rdb"))[0], 'r') as f:
#        lines = f.readlines()
#
#    for line in lines:
#        parallactic_angle = line.strip().split()[11]
#        if ((not "paralactic_angle" in parallactic_angle) and (not "---" in parallactic_angle)):
#            parallactic_angle_vec = np.append(parallactic_angle_vec,
#                                              parallactic_angle)

##########################################################################

if collapse_cube:
    master_cube = np.nanmean(master_cube, axis=0)

# Write the output file with all of the frames in the cube
# If there's only 1 file, just copy it rather than saving it with pyfits
# This will be much quicker
if len(dirlist) == 1 and not collapse_cube:
    shutil.copy(dirlist[0], output_dir + output_file)
else:
    pyfits.writeto(output_dir + output_file, master_cube, header=hdr,
                   overwrite=True)

# Write an output file with the parallactic angles
if not skip_parang:
    pd_cubelist.paralactic_angle.to_csv(output_dir + 'parallactic_angle.txt',
                                        header=False, index=False)
#    with open(output_dir+"parallactic_angle.txt", 'w') as f2:
#        for i, parallactic_angle in enumerate(parallactic_angle_vec):
#            f2.write(parallactic_angle + "\n")

sys.stdout.write("Total time: " +
                 graphic_nompi_lib.humanize_time((time.time() - t0)))
sys.stdout.write("\n")
sys.stdout.write("end of convert\n")
sys.stdout.flush()

sys.exit(0)
