#/usr/bin/env python
"""
Based on the ACORNS (Brandt et al 2013)
"""
__version__='3.3'
__subversion__='0'

import astropy.io.fits as pyfits
import numpy as np
import glob
## import bottleneck
from graphic_nompi_lib_330 import create_dirlist
import sys
import os
import argparse

sys.path.append('~/Astro/AO-reduction/destripe_acorns/')
from destripe import *

parser = argparse.ArgumentParser(description='Cleans HICAO images using the destripe routine')
parser.add_argument('--pattern', action="store", dest="pattern",  required=True, help='Filename pattern')
parser.add_argument('--flat', action="store", dest="flat_file",  required=True, help='Domeflat file including path')
parser.add_argument('-no_badpix', dest='badpix', action='store_const',
				   const=False, default=True,
				   help='Do not clean for bad pixels')

args = parser.parse_args()
pattern=args.pattern
flat_file=args.flat_file
badpix=args.badpix

def verticalmed(flux, flat, r_ex=0):

	"""
	Function verticalmed takes two arguments:
	1.  A 2048 x 2048 array of flux values
	2.  A 2048 x 2048 flat-field array

	Optional arguments:
	3.  Exclusion radius for calculting the median of the horizontal stripes
		  (default zero, recommended values from 0 to 800)
		  See Kandori-san's IDL routine for the equivalent.
	4.  Use separate left and right channels, as for HiCIAO's PDI
		mode?  default False

	verticalmed takes the median of the horizontal stripes to calculate a
	vertical template, as in Kandori-san's routine.  The routine ignores a
	circular region about the array center if r_ex > 0, and also ignores
	all pixels flagged with NaN.
	"""

	###############################################################
	# Construct radius array, copy flux to mask
	###############################################################

	dimy, dimx = flux.shape
	x = np.arange(dimx)
	y = np.arange(dimy)
	x, y = np.meshgrid(x, y)

	if r_ex > 0:
		r_ok = ((x - dimx / 2)**2 + (y - dimy / 2)**2) > r_ex**2

		else:
			r_ok = ((x - dimx / 4)**2 + (y - dimy / 2)**2) > r_ex**2
			r_ok *= ((x - 3 * dimx / 4)**2 + (y - dimy / 2)**2) > r_ex**2

		flux2 = np.ndarray(flux.shape, np.float32)
		flux2[:] = flux
		np.putmask(flux2, np.logical_not(r_ok), np.nan)
	else:
		flux2 = flux

	###############################################################
	# Estimate background level
	###############################################################

	backgnd = np.ndarray(flux2.shape)
	backgnd[:] = flux2 / flat
	backgnd = np.sort(np.reshape(backgnd, -1))
	ngood = np.sum(np.isfinite(backgnd))
	level = np.median(backgnd[:ngood])
	flux2 -= level * flat

	###############################################################
	# Sort the flux values.  NaN values will be at the end for
	# numpy versions >= 1.4.0; otherwise this routine may fail
	###############################################################

	tmp = np.ndarray((32, dimy // 32, dimx), np.float32)
	for i in range(1, 33, 2):
		tmp[i] = flux2[64 * i:64 * i + 64]
		tmp[i - 1] = flux2[64 * i - 64:64 * i]

	tmp = np.sort(tmp, axis=0)
	oldshape = tmp[0].shape
	tmp = np.reshape(tmp, (tmp.shape[0], -1))

	oddstripe = np.zeros(tmp[0].shape, np.float32)

	###############################################################
	# imax = number of valid (!= NaN) references for each pixel.
	# Calculate the median using the correct number of elements,
	# doing it only once for each pixel.
	###############################################################

	imax = np.sum(np.logical_not(np.isnan(tmp)), axis=0)
	for i in range(np.amin(imax), np.amax(imax) + 1):
		indxnew = np.where(imax == i)
		if len(indxnew) > 0:
			oddstripe[indxnew] = np.median(tmp[:i, indxnew], axis=0)

	###############################################################
	# Set the median of the pattern to be subtracted equal to the
	# median of the difference between the science and reference
	# pixels.
	###############################################################

	oddstripe -= np.median(oddstripe)
	oddstripe += 0.5 * (np.median(flux[:4]) + np.median(flux[-4:]))

	oddstripe = np.reshape(oddstripe, oldshape)
	evenstripe = oddstripe[::-1]

	return [oddstripe, evenstripe]


def destripe(flux,flat, r_ex=0):

	oddstripe, evenstripe = verticalmed(flux, flat, r_ex=r_ex)

	for i in range(1, 33, 2):
		flux[64 * i:64 * i + 64] -= oddstripe * sub_coef
		flux[64 * i - 64:64 * i] -= evenstripe * sub_coef

	##############################################################
	# Four rows on each edge are reference pixels--don't
	# flatfield them
	##############################################################

		flux[4:-4, 4:-4] /= flat[4:-4, 4:-4]

		flux[flux < -1000] = 0
		flux[flux > 5e4 * ncoadd] = np.nan
	else:
		flux[4:-4, 4:-4] /= flat[4:-4, 4:-4]

	return flux

dirlist=create_dirlist(pattern)

## flat,f_hdr=pyfits.getdata('../../../DOMEFLAT/1.5/dome_flat_HICA00154023.fits', header=True)
flat,f_hdr=pyfits.getdata(flat_file, header=True)

for i in xrange(len(dirlist)):
	## frame,frame_hdr=pyfits.getdata(dirlist[i], header=True)
	if os.access('ds_'+dirlist[i], os.F_OK | os.R_OK):
		print('Already processed: '+dirlist[i])
		continue
	print('Processing '+str(i)+'/'+str(len(dirlist))+' '+dirlist[i])
	hdr=pyfits.getheader(dirlist[i])
	hdr["HIERARCH GC DESTRIPE"]=('T', "")
	## ds_frame=destripe(dirlist[i],flat,False,False,None,False,clean=badpix, storeall=True, r_ex=0, extraclean=True,
			 ## full_destripe=True, do_horiz=True, PDI=False)
	ds_frame=destripe(dirlist[i],flat, r_ex=0)
	pyfits.writeto('ds_'+dirlist[i], ds_frame, header=hdr)
