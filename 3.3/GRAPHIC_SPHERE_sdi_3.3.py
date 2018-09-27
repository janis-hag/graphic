#!/usr/bin/env python3
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

Its purpose is to substract a generated PSF from each frame in a cube.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""
import glob
import sys
import shutil
import astropy.io.fits as pyfits
import graphic_nompi_lib_330 as graphic_nompi_lib
import argparse
import time

__version__ = '3.3'
__subversion__ = '0'


target_dir = "."
parser = argparse.ArgumentParser(description='Apply the sdi algorythm on \
                                 SPHERE (for now only for IRDIS) data and \
                                 produce sdi_* cubes')
parser.add_argument('--pattern', action="store", dest="pattern",
                    default="left*SCIENCE_DBI", help='cubes to apply the sdi')
parser.add_argument('-additional', action="store_const", dest="additional",
                    const=True,  default=False, help='if True produce in \
                    addition a cube of the left image rescaled (the one used \
                    to do the subtraction with right image)')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",
                    default='all_info', help='Info filename pattern.')
parser.add_argument('--info_dir', action="store", dest="info_dir",
                    default='cube-info', help='Info directory')
parser.add_argument('--r_int', action="store", dest="r_int",  default=30,
                    type=int, help='Interior radius (in pixels) for the area \
                    used to calculate the flux ratio between the cubes')
parser.add_argument('--r_ext', action="store", dest="r_ext",  default=80,
                    type=int, help='Exterior radius (in pixels) for the area \
                    used to calculate the flux ratio between the cubes')


args = parser.parse_args()
additional = args.additional
pattern = args.pattern
info_pattern = args.info_pattern
info_dir = args.info_dir
r_int = args.r_int
r_ext = args.r_ext


t0 = time.time()
print("beginning of sdi:")

length = 0

sys.stdout.write('Application du sdi sur:')
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

# Loop through the cubes and run SDI
for allfiles in glob.iglob(pattern+'*'):
    sys.stdout.write('\n')
    sys.stdout.write('\r Cube ' + str(count) + '/' + str(length))
    sys.stdout.flush()

    # Load the cubes
    cube_left,hdr_l=pyfits.getdata(allfiles,header=True)
    cube_right,hdr_r=pyfits.getdata(allfiles.replace('left','right'),header=True)

    # Work out the wavelengths and which cube we want to rescale
    # For H23 and K12, the right channel has the absorption
    # For Y23 and J23, the left channel has the absorption
    if hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_H23':
        reference_lambda=1588.8
        scaled_lambda=1667.1
        reference_cube = cube_left
        scaled_cube = cube_right

        # And some header params
        hdr = hdr_l
        rescaled_channel_name = 'right'

    elif hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_K12':
        reference_lambda=2102.5
        scaled_lambda=2255
        reference_cube = cube_left
        scaled_cube = cube_right

        # And some header params
        hdr = hdr_l
        rescaled_channel_name = 'right'

    elif hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_Y23':
        scaled_lambda=1025.8
        reference_lambda=1080.2
        reference_cube = cube_right
        scaled_cube = cube_left

        # And some header params
        hdr = hdr_r
        rescaled_channel_name = 'left'

    elif hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_J23':
        scaled_lambda=1189.5
        reference_lambda=1269.8
        reference_cube = cube_right
        scaled_cube = cube_left

        # And some header params
        hdr = hdr_r
        rescaled_channel_name = 'left'

    if count == 1:
        sys.stdout.write('\n')
        sys.stdout.write('filter: ' + hdr_l["HIERARCH ESO INS COMB IFLT "])
        sys.stdout.flush()

    # Rescale all of the cubes to the reference wavelength
    rescaled_cube = graphic_nompi_lib.rescale_image(scaled_cube,reference_lambda/scaled_lambda,reference_lambda/scaled_lambda)

    # Work out the flux scaling factor needed to best subtract the PSF
    flux_factor = graphic_nompi_lib.scale_flux(reference_cube,rescaled_cube,r_int=r_int,r_ext=r_ext)
    hdr['HIERARCH GC SDI Flux rescaling:']=flux_factor # Save it in the header
    hdr['HIERARCH GC SDI Channel rescaled:']= rescaled_channel_name

    # Now do the subtraction
    sdi_cube = reference_cube - flux_factor*rescaled_cube
    sys.stdout.write('\n')
    sys.stdout.write('  Scaling flux of '+rescaled_channel_name+' cube by '+str(flux_factor))
    sys.stdout.flush()

    # Write it out
    pyfits.writeto(allfiles.replace('left','sdi'),sdi_cube,header=hdr,clobber=True)

    if additional:
        scaled_cube_output_name = allfiles.replace('left',rescaled_channel_name+'_rescale')
        pyfits.writeto(scaled_cube_output_name,rescaled_cube,header=hdr,clobber=True)

    count+=1

# Copy the info files
for allfiles in glob.iglob(info_dir+'/'+info_pattern+'*'):
    shutil.copyfile(allfiles,allfiles.replace('left','sdi'))

sys.stdout.write('\n')
print("Total time: "+graphic_nompi_lib.humanize_time((time.time()-t0)))
print("sdi finished")
sys.exit(0)

