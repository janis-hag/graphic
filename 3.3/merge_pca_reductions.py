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
files = np.array(files)

# Need to account for the fact that the sort did not actually sort it by number of modes, but
# by the first number...
nmodes = []
for f in files:
    nmodes.append(float(f.replace(basedirectory,'').split('/')[0]))
sort_ix = np.argsort(nmodes)
files = files[sort_ix]

cube = []

for x,filename in enumerate(files):
    image, header = fits.getdata(filename, header = True)
    cube.append(image)
    if x==0:
        outputhdr=header

    outputhdr['HIERARCH GC PCA FRAME' + str(x)+' NMODES'] = header['HIERARCH GC PCA NMODES']
    outputhdr['HIERARCH GC PCA FRAME' + str(x)+' MINREFFRAMES'] = header['HIERARCH GC PCA MINREFFRAMES']
    
fits.writeto(basedirectory + 'pca_multimodes.fits',cube,outputhdr,clobber=True)
