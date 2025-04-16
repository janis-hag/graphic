# -*- coding: utf-8 -*-
"""
Estimate the contrast from a GRAPHIC reduction for SPHERE

@author: peretti
"""

import numpy as np
import pyfits, glob


def psf_combine(path,pattern_l,pattern_r):
    nbr_image_to_kill_at_beginning_of_cube=0

    for i,allfiles in enumerate(glob.iglob(path+pattern_l+'*')):
        psf_cube,hdr_psf_l=pyfits.getdata(allfiles,header=True)
        if np.size(np.shape(psf_cube))>2:
            psf_combine=np.median(psf_cube[nbr_image_to_kill_at_beginning_of_cube:,:,:],axis=0)
        else:
            psf_combine=psf_cube
        if i==0:
            psf_left_final=psf_combine
        else:
            psf_left_final=np.median([psf_combine,psf_left_final],axis=0)

    pyfits.writeto(path+"psf_left.fits",psf_left_final,header=hdr_psf_l,clobber=True)


    for i,allfiles in enumerate(glob.iglob(path+pattern_r+'*')):
        psf_cube,hdr_psf_r=pyfits.getdata(allfiles,header=True)
        if np.size(np.shape(psf_cube))>2:
            psf_combine=np.median(psf_cube[nbr_image_to_kill_at_beginning_of_cube:,:,:],axis=0)
        else:
            psf_combine=psf_cube
        if i==0:
            psf_right_final=psf_combine
        else:
            psf_right_final=np.median([psf_combine,psf_right_final],axis=0)

    pyfits.writeto(path+"psf_right.fits",psf_right_final,header=hdr_psf_r,clobber=True)
