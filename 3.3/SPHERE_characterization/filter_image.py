import pyfits
import numpy as np
import pylab as py
import scipy.signal
from scipy import fftpack
from filtres import *

path="/Users/sebastien/Documents/Doctorat/SPHERE/GJ504_results/29-03-2016/"
filename="smart_annular_pca_derot.fits"
im=pyfits.getdata(path+filename)
nan_mask=np.where(np.isnan(im),np.nan,1)
energy_im=np.nansum(im)
im=np.real(np.nan_to_num(im))
l=np.shape(im)[0]
x=np.arange(-l/2,l/2)
y=x
X,Y=np.meshgrid(x,y)
R=np.sqrt(X**2+Y**2)

low_pass_filter=np.where(R<3,1,0)
fft_low_pass=np.fft.fft2(low_pass_filter)
fft_low_pass=fft_low_pass/np.max(abs(fft_low_pass))
fft_im=np.fft.fft2(im)
convolution=np.real(np.fft.fftshift(np.fft.ifft2(fft_low_pass*fft_im)))*nan_mask

#convolution=convolution*energy_im/np.nansum(convolution)
#convolution2=scipy.signal.convolve2d(im, low_pass_filter, mode='full', boundary='fill', fillvalue=0)

pyfits.writeto(path+"test_convol.fits",convolution,clobber=True)

#pyfits.writeto(path+"test_convol2.fits",convolution2,clobber=True)