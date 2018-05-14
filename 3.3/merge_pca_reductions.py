#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:31:53 2017

@author: emilyrickman
"""
from astropy.io import fits
import numpy as np
import os
import glob

basedirectory = "./GRAPHIC_PCA/"

files = glob.glob(basedirectory + "*/*_derot.fits")
files = np.sort(files)

cube = []
nmodes = []
nminmodes = []
for x,filename in enumerate(files):
    image, header = fits.getdata(filename, header = True)
    cube.append(image)
    if x==0:
        outputhdr=header
        
    nmodes.append(float(header['HIERARCH GC PCA NMODES']))
    nminmodes.append(header['HIERARCH GC PCA MINREFFRAMES'])

cube = np.array(cube)
nmodes = np.array(nmodes)
nminmodes = np.array(nminmodes)

# Need to account for the fact that the sort did not actually sort it by number of modes, but
# by the first number...
sort_ix = np.argsort(nmodes)
cube = cube[sort_ix]
nmodes = nmodes[sort_ix]
nminmodes = nminmodes[sort_ix]

for ix in range(len(nmodes)):

    outputhdr['HIERARCH GC PCA FRAME' + str(ix)+' NMODES'] = nmodes[ix]
    outputhdr['HIERARCH GC PCA FRAME' + str(ix)+' MINREFFRAMES'] = nminmodes[ix]
    
fits.writeto(basedirectory + 'pca_multimodes.fits',cube,outputhdr,clobber=True)
