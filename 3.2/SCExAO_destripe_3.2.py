#/usr/bin/python
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
	ds_frame=destripe(dirlist[i],flat,False,False,None,False,clean=badpix, storeall=True, r_ex=0, extraclean=True,
			 full_destripe=True, do_horiz=True, PDI=False)
	pyfits.writeto('ds_'+dirlist[i], ds_frame, header=hdr)
