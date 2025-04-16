import numpy as np
import pylab as py
import scipy.interpolate
import pyfits
import scipy,glob
from scipy import ndimage
from scipy import fftpack
from scipy.optimize import leastsq
from translat_func import *
from ND_transmission import *
from kapteyn import kmpfit
from Calib import *


def error(par,data):
    im=data[0]
    psf=data[1]
    size=np.shape(im)[0]
    amplitude=par[0]
    center=[par[1],par[2]]
    psf_translat=f_translat(psf*amplitude,center[1]-size/2.,center[0]-size/2.)
    e=(psf_translat-im)
    return np.asarray(np.real(e)).reshape(-1)

def im_simul(par,psf):
    size=np.shape(psf)[0]
    center=[par[1],par[2]]
    im_simulated=f_translat(psf*par[0],center[1]-size/2.,center[0]-size/2.)
    return im_simulated

def adi_self_subtraction_computation(path,pattern_image,pattern_psf,sep_FP,pix_scale,dmag_given):
    for i,allfiles in enumerate(glob.iglob(pattern_psf+"*")):
        if i==0:
            psf_filename=allfiles
            print("psf filename:",psf_filename)
        else:
            print("Error more than one file found with this pattern. Used the first one:",psf_filename)
    for i,allfiles in enumerate(glob.iglob(pattern_image+"*")):
        if i==0:
            image_filename=allfiles
            print("image filename:",image_filename)
        else:
            print("Error more than one file found with this pattern. Used the first one:",image_filename)

    path_psf=path
    adi_self_subtraction_vec=np.zeros(np.size(sep_FP))

    im_final,hdr=pyfits.getdata(path+image_filename,header=True)
    psf,hdr_psf=pyfits.getdata(path_psf+psf_filename,header=True)

    band_filter=hdr['HIERARCH ESO INS COMB IFLT']
    if band_filter=="DB_H23":
    	filter_name="H23"
    elif band_filter=="DB_K12":
    	filter_name="K12"
    elif band_filter=="DB_Y23":
    	filter_name="Y23"
    elif band_filter=="DB_J23":
    	filter_name="J23"
    else :
    	import sys
    	sys.exit("Error of filter name")

    if np.size(np.shape(psf))>2:
        psf=np.nanmedian(psf[3:,:,:],axis=0)

    psf_norm=psf/np.nanmax(psf)

    index_psf_center=np.where(psf_norm==np.nanmax(psf_norm))
    psf_norm_cut=psf_norm[index_psf_center[0][0]-16:index_psf_center[0][0]+16,index_psf_center[1][0]-16:index_psf_center[1][0]+16]

    xmag=pix_scale
    ymag=pix_scale
    xrotation=0
    yrotation=0
    exmag=0.03/1000
    eymag=0.03/1000
    exrotation=0.09
    eyrotation=0.09
    nbr_loop=100

    for l in range(np.size(sep_FP)):
        PA_comp=0
        rho_comp=sep_FP[l]
        Prim_x=-np.sin(PA_comp)*rho_comp/pix_scale+np.shape(im_final)[1]/2.
        Prim_y=np.cos(PA_comp)*rho_comp/pix_scale+np.shape(im_final)[1]/2.

        ########################
        adi_self_subtraction=0
        size_image_cut=32
        image_cut=im_final[Prim_y-size_image_cut/2:Prim_y+size_image_cut/2,Prim_x-size_image_cut/2:Prim_x+size_image_cut/2]

        for i in range(nbr_loop):
            paramsinitial=[np.max(image_cut[np.shape(image_cut)[0]/2-10:np.shape(image_cut)[0]/2+10,np.shape(image_cut)[1]/2-10:np.shape(image_cut)[1]/2+10]),size_image_cut/2,size_image_cut/2]
            fitobj = kmpfit.Fitter(residuals=error, data=(image_cut-np.nanmedian(image_cut),psf_norm_cut))
            fitobj.fit(params0=paramsinitial)
            par=fitobj.params
            err_leastsq=fitobj.stderr

            position_x=Prim_x-size_image_cut/2+par[2]
            position_y=Prim_y-size_image_cut/2+par[1]

            im_fit=im_simul(par,psf_norm_cut)
            x=np.arange(-np.shape(image_cut)[0]/2,np.shape(image_cut)[0]/2)
            y=x
            X,Y=np.meshgrid(x,y)
            R=np.sqrt(X**2+Y**2)
            mask=np.where(R<4,0,image_cut)
            mask_inv=np.where(R>=4,0,im_fit)


            #####################################
            #calcul rho et PA avec calibration

            dxpix=position_x-np.shape(im_final)[1]/2
            dypix=position_y-np.shape(im_final)[0]/2

            edxpix =err_leastsq[2]
            edypix =err_leastsq[1]

            primary_flux=np.nanmax(psf)
            DIT_flux=hdr_psf['HIERARCH ESO DET SEQ1 DIT']
            DIT_image=hdr['HIERARCH ESO DET SEQ1 DIT']
            Neutral_density_flux=hdr_psf['HIERARCH ESO INS4 FILT2 NAME']
            Neutral_density_image=hdr['HIERARCH ESO INS4 FILT2 NAME']

            Neutral_density_flux=ND_filter_transmission(Neutral_density_flux,filter_name)
            Neutral_density_image=ND_filter_transmission(Neutral_density_image,filter_name)
            Neutral_density_flux=1./Neutral_density_flux
            Neutral_density_image=1./Neutral_density_image

            primary_flux=(primary_flux*Neutral_density_flux*DIT_image)/(Neutral_density_image*DIT_flux)

            f1=par[0]/(1-adi_self_subtraction)
            ef1=err_leastsq[0]
            f2=primary_flux
            ef2=0.01*primary_flux #we chose an error on the flux of the primarry that is of 1% as it may vary from frame to frames and change during the night


            rho,erho,PA,ePA,dmag,edmag=Calib(xmag,ymag,xrotation/180.*np.pi,yrotation/180.*np.pi,exmag,eymag,exrotation/180.*np.pi,eyrotation/180.*np.pi,dxpix,dypix,edxpix,edypix,f1,f2,ef1,ef2)

            PA=PA*180/np.pi
            ePA=ePA*180/np.pi

            if dmag>dmag_given:
                adi_self_subtraction+=0.01
            else:
                break
        print("dmag",dmag)
        print("ADI self subtraction at ",str(rho_comp),"arcsec = ",adi_self_subtraction)
        adi_self_subtraction_vec[l]=adi_self_subtraction

    sep_FP=np.append(np.array([0]),sep_FP)
    adi_self_subtraction_vec=np.append(np.array([0.9]),adi_self_subtraction_vec)


    f=open("adi_self_sub.txt",'w')
    f.write("Separation:\t")
    for i in range(np.size(sep_FP)):
        f.write(str(sep_FP[i]))
        if i!=np.size(sep_FP)-1:
            f.write(" ")
        else:
            f.write(" \n")
    f.write("Self-subtraction:\t")
    for i in range(np.size(adi_self_subtraction_vec)):
        f.write(str(adi_self_subtraction_vec[i]))
        if i!=np.size(adi_self_subtraction_vec)-1:
            f.write(" ")
    f.close()
