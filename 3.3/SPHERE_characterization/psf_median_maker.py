import pyfits
import numpy as np
import pylab as py

path="/Users/sebastien/Documents/Doctorat/SPHERE/GJ504_results/29-03-2016/"
filename_r="right_cl_nomed_SPHER.2016-03-29T06_07_00.010IRD_FLUX_CALIB_CORO_RAW"
filename_l="left_cl_nomed_SPHER.2016-03-29T06_07_00.010IRD_FLUX_CALIB_CORO_RAW"

cube_r,hdr_r=pyfits.getdata(path+filename_r+".fits",header=True)
cube_l,hdr_l=pyfits.getdata(path+filename_l+".fits",header=True)

median_im_r=np.median(cube_r,axis=0)
median_im_l=np.median(cube_l,axis=0)

pyfits.writeto(path+"psf_right.fits",median_im_r,header=hdr)
pyfits.writeto(path+"psf_left.fits",median_im_l,header=hdr)
