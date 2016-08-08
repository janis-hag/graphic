import pyfits
import numpy as np
import pylab as py
import scipy
from scipy import ndimage
from scipy import fftpack

def f_translat(image,x,y):
    """

    """

    im=np.abs(np.nan_to_num(image.copy()))
    fft=fftpack.fft2(im)
    k=fftpack.fftfreq(np.shape(im)[0])

    #translation sur l'axe x et y
    fft_bis=np.transpose(np.transpose(fft*np.exp(-2*np.pi*1j*(x*k)))*np.exp(-2*np.pi*1j*(y*k)))

    im_translat=np.real(np.nan_to_num(fftpack.ifft2(fft_bis)))


    return im_translat
