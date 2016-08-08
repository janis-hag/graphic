# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import fftpack


def contrast_curve_sdi_func(band_filter,psf_l,psf_r,binning_factor,mag_l,mag_r):
    if band_filter=="DB_H23":
        lambda1=1.5888e-6
        lambda2=1.6671e-6
    elif band_filter=="DB_K12":
        lambda1=2.1025e-6
        lambda2=2.255e-6
    elif band_filter=="DB_Y23":
        lambda1=1.0258e-6
        lambda2=1.0802e-6
    elif band_filter=="DB_J23":
        lambda1=1.1895e-6
        lambda2=1.2698e-6
    else :
        import sys
        sys.exit("Error of filter name")

    psf_l_lin=psf_l[np.where(psf_l==np.nanmax(psf_l))[0],:][0]
    psf_r_lin=psf_r[np.where(psf_r==np.nanmax(psf_r))[0],:][0]
    psf_l_lin=np.nan_to_num(psf_l_lin).astype(float)
    psf_r_lin=np.nan_to_num(psf_r_lin).astype(float)
    x=np.arange(np.size(psf_l_lin)*binning_factor*1.)/binning_factor

    fft_l_lin=fftpack.fftshift(fftpack.fft(psf_l_lin))
    fft_l_lin=np.append(np.zeros(np.size(x)/2-np.size(psf_l_lin)/2),fft_l_lin)
    fft_l_lin=np.append(fft_l_lin,np.zeros(np.size(x)/2-np.size(psf_l_lin)/2))
    psf_l_lin=np.real(fftpack.ifft(fftpack.ifftshift(fft_l_lin)))*binning_factor
    max_index=np.where(psf_l_lin==np.nanmax(psf_l_lin))[0][0]
    max_l=np.nanmax(psf_l_lin)
    psf_l_lin=psf_l_lin/max_l

    fft_r_lin=fftpack.fftshift(fftpack.fft(psf_r_lin))
    fft_r_lin=np.append(np.zeros(np.size(x)/2-np.size(psf_r_lin)/2),fft_r_lin)
    fft_r_lin=np.append(fft_r_lin,np.zeros(np.size(x)/2-np.size(psf_r_lin)/2))
    psf_r_lin=np.real(fftpack.ifft(fftpack.ifftshift(fft_r_lin)))*binning_factor
    max_r=np.nanmax(psf_r_lin)
    psf_r_lin=psf_r_lin/max_r
    
    psf_l_lin=psf_l_lin*mag_l
    psf_r_lin=psf_r_lin*mag_r


    psf_l_lin=np.roll(psf_l_lin,-(max_index-np.size(psf_l_lin)/2))
    psf_r_lin=np.roll(psf_r_lin,-(max_index-np.size(psf_l_lin)/2))

    max_vec=[]
    psf_l_vec=np.zeros((np.size(x)/(2*int(binning_factor)),np.size(x)))
    psf_r_vec=np.zeros((np.size(x)/(2*int(binning_factor)),np.size(x)))
    index=0
    for i in range(np.size(x)/(2*int(binning_factor))):
        if np.mod(i,50)==0 and i!=0:
            print (i*1.)/(np.size(x)/(2*int(binning_factor)))*100,"%"    
        psf_l_lin_temp=np.roll(psf_l_lin,i*binning_factor)
        psf_r_lin_temp=np.roll(psf_r_lin,int((i*binning_factor-i*binning_factor*(lambda2/lambda1-1))))
        psf_l_vec[index]=psf_l_lin_temp
        psf_r_vec[index]=psf_r_lin_temp
        max_vec=np.append(max_vec,np.max(psf_l_lin_temp-psf_r_lin_temp))
        index+=1

    return max_vec
