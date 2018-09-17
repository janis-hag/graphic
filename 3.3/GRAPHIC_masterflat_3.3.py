#!/usr/bin/env python3
"""
Simple flat field generator
"""
__version__='3.3'
__subversion__='0'

import astropy.io.fits as pyfits
import numpy as np
## import glob
## import bottleneck
import graphic_nompi_lib_330 as graphic_nompi_lib
## import sys
## import os
import argparse


parser = argparse.ArgumentParser(description='Simple flat field generator.')
parser.add_argument('--pattern', action="store", dest="pattern", required=True, help='Filename pattern')
## parser.add_argument('--flat', action="store", dest="flat_file",  help='Domeflat file including path')


args = parser.parse_args()
pattern=args.pattern
## flat_file=args.flat_file

dirlist=graphic_nompi_lib.create_dirlist(pattern)

cube,hdr=pyfits.getdata(dirlist[0], header=True)

for i in range(1,len(dirlist)):
    cube=np.dstack((cube,pyfits.getdata(dirlist[i])))

flat=np.median(cube, axis=2)

pyfits.writeto('flat_'+dirlist[0], flat, header=hdr)

print('Generated flat_'+dirlist[0])
