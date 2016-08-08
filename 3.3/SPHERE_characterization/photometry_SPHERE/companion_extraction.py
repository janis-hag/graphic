# -*- coding: utf-8 -*-
import pyfits,scipy,glob,os
import numpy as np
import pylab as py
from scipy import ndimage
from scipy import fftpack
from scipy.optimize import leastsq
from translat_func import *
import ND_transmission
reload(ND_transmission)
from ND_transmission import *
from kapteyn import kmpfit
from Calib import *
from os.path import isfile
import os.path





def error(par,data):
    im=data[0]
    psf=data[1]
    size=np.shape(im)[0]
    amplitude=par[0]
    center=[par[1],par[2]]
    psf_translat=f_translat(psf*amplitude,center[1]-size/2.,center[0]-size/2.)
    e=(psf_translat-im)
    return np.asarray(np.real(e)).reshape(-1)

def im_simul_test(par,psf):
    size=np.shape(psf)[0]
    center=[par[1],par[2]]
    im_simulated=f_translat(psf*par[0],center[1]-size/2.,center[0]-size/2.)
    return im_simulated


def companion_extraction(path,path_results,pattern_image,pattern_psf,pix_scale):
	for i,allfiles in enumerate(glob.iglob(pattern_psf+"*")):
		if i==0:
			psf_filename=allfiles
			print "psf filename:",psf_filename
		else:
			print "Error more than one file found with this pattern. Used the first one:",psf_filename
	for i,allfiles in enumerate(glob.iglob(pattern_image+"*")):
		if i==0:
			image_filename=allfiles
			print "image filename:",image_filename
		else:
			print "Error more than one file found with this pattern. Used the first one:",image_filename
		
	path_psf=path
	im_final,hdr=pyfits.getdata(path+image_filename,header=True)
	psf,hdr_psf=pyfits.getdata(path_psf+psf_filename,header=True)

	band_filter=hdr['HIERARCH ESO INS COMB IFLT']
	if band_filter=="DB_H23":
		if "sdi" in image_filename:
			filter_name="H23"
		elif "left" in image_filename:
			filter_name="H2"
		elif "right" in image_filename:
			filter_name="H3"
		else:
			print "Error could not find the good filter name"
		lambda1=1.5888e-6
		lambda2=1.6671e-6
	elif band_filter=="DB_K12":
		if "sdi" in image_filename:
			filter_name="K12"
		elif "left" in image_filename:
			filter_name="K1"
		elif "right" in image_filename:
			filter_name="K2"
		else:
			print "Error could not find the good filter name"
		lambda1=2.1025e-6
		lambda2=2.255e-6
	elif band_filter=="DB_Y23":
		if "sdi" in image_filename:
			filter_name="Y23"
		elif "left" in image_filename:
			filter_name="Y2"
		elif "right" in image_filename:
			filter_name="Y3"
		else:
			print "Error could not find the good filter name"
		lambda1=1.0258e-6
		lambda2=1.0802e-6
	elif band_filter=="DB_J23":
		if "sdi" in image_filename:
			filter_name="J23"
		elif "left" in image_filename:
			filter_name="J2"
		elif "right" in image_filename:
			filter_name="J3"
		else:
			print "Error could not find the good filter name"
		lambda1=1.1895e-6
		lambda2=1.2698e-6
	else :
		import sys
		sys.exit("Error of filter name")
    
	target_name=hdr["OBJECT"]

	if np.size(np.shape(psf))>2:
		psf=np.nanmedian(psf[3:,:,:],axis=0)

	psf_norm=psf/np.nanmax(psf)

	index_psf_center=np.where(psf_norm==np.nanmax(psf_norm))
	psf_norm_cut=psf_norm[index_psf_center[0]-16:index_psf_center[0]+16,index_psf_center[1]-16:index_psf_center[1]+16]


	xmag=pix_scale
	ymag=pix_scale
	xrotation=0
	yrotation=0
	exmag=0.03/1000
	eymag=0.03/1000
	exrotation=0.09
	eyrotation=0.09
	find_adi_subtraction=False
	
	if os.path.isfile(path+"ds9_regions.reg"):
		comp=True
	else:
		comp=False
	
	if comp:
		f=open(path+"ds9_regions.reg",'r')
		lines=f.readlines()
		f.close()
		for line in lines:
			if line.strip().split()[0][:6]=="circle":
				Prim_x=float(line[7:].strip().split(",")[0])
				Prim_y=float(line[7:].strip().split(",")[1])

		f=open("adi_self_sub.txt","r")
		lines=f.readlines()
		f.close()
		sep_vec=(lines[0].strip().split())[1:]
		adi_self_subtraction_vec=(lines[1].strip().split())[1:]
		for i in range(np.size(sep_vec)):
			sep_vec[i]=float(sep_vec[i])
			adi_self_subtraction_vec[i]=float(adi_self_subtraction_vec[i])
		p = scipy.interpolate.interp1d(sep_vec, adi_self_subtraction_vec)

		sep_pl_pix=np.sqrt((Prim_x-np.shape(im_final)[1]/2.)**2+(Prim_y-np.shape(im_final)[0]/2.)**2)
		sep_pl_arcsec=sep_pl_pix*pix_scale
		adi_self_subtraction=p(sep_pl_arcsec)
		print "ADI self-subtraction at planet distance: ", adi_self_subtraction

		sep_vec_test=np.arange(100)/100.*np.max(sep_vec)


		########################

		size_image_cut=32
		image_cut=im_final[Prim_y-size_image_cut/2:Prim_y+size_image_cut/2,Prim_x-size_image_cut/2:Prim_x+size_image_cut/2]

		if find_adi_subtraction:
			nbr_loop=100
			adi_self_subtraction=0.01
		else:
			nbr_loop=1

		for i in range(nbr_loop):
			paramsinitial=[np.max(image_cut[np.shape(image_cut)[0]/2-10:np.shape(image_cut)[0]/2+10,np.shape(image_cut)[1]/2-10:np.shape(image_cut)[1]/2+10]),size_image_cut/2,size_image_cut/2]
			fitobj = kmpfit.Fitter(residuals=error, data=(image_cut-np.nanmedian(image_cut),psf_norm_cut))
			fitobj.fit(params0=paramsinitial)
			par=fitobj.params
			err_leastsq=fitobj.stderr

			position_x=Prim_x-size_image_cut/2+par[2]
			position_y=Prim_y-size_image_cut/2+par[1]

			im_fit=im_simul_test(par,psf_norm_cut)
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
			Neutral_density_flux=1/Neutral_density_flux
			Neutral_density_image=1/Neutral_density_image

			primary_flux=(primary_flux*Neutral_density_flux*DIT_image)/(Neutral_density_image*DIT_flux)

			f1=par[0]/(1-adi_self_subtraction)
			ef1=err_leastsq[0]
			f2=primary_flux
			ef2=0.01*primary_flux #we chose an error on the flux of the primarry that is of 1% as it may vary from frame to frames and change during the night


			rho,erho,PA,ePA,dmag,edmag=Calib(xmag,ymag,xrotation/180.*np.pi,yrotation/180.*np.pi,exmag,eymag,exrotation/180.*np.pi,eyrotation/180.*np.pi,dxpix,dypix,edxpix,edypix,f1,f2,ef1,ef2)

			PA=PA*180/np.pi
			ePA=ePA*180/np.pi

			"""if find_adi_subtraction:
				if dmag>dmag_given:
					adi_self_subtraction+=0.01
				else:
					break"""

		if "sdi" in pattern_image:
			print "Warning: SDI image -> separation corrected of factor "+str(lambda1/lambda2)+" from rezising"
			rho=rho*lambda1/lambda2
			
		print "Neutral_density_flux=",Neutral_density_flux
		print "Neutral_density_image=",Neutral_density_image
		print "adi_self_subtraction",adi_self_subtraction

		print "rho=",rho
		print "erho=",erho
		print "PA=",PA
		print "ePA=",ePA
		print "dmag=",dmag
		print "edmag=",edmag


		if not os.path.isfile(path_results+"companion_extraction.txt"):
			f=open(path_results+"companion_extraction.txt","w")
			f.write("filter")
			f.write("\t")
			f.write("rho")
			f.write("\t")
			f.write("erho")
			f.write("\t")
			f.write("PA")
			f.write("\t")
			f.write("ePA")
			f.write("\t")
			f.write("dmag")
			f.write("\t")
			f.write("edmag")
			f.write("\n")
			f.write("----------------------------------------------")
			f.write("\n")
		else:
			f=open(path_results+"companion_extraction.txt","a")
		f.write(filter_name)
		f.write("\t")
		f.write(str(round(rho,4)))
		f.write("\t")
		f.write(str(round(erho,4)))
		f.write("\t")
		f.write(str(round(PA,3)))
		f.write("\t")
		f.write(str(round(ePA,3)))
		f.write("\t")
		f.write(str(round(dmag,3)))
		f.write("\t")
		f.write(str(round(edmag,3)))
		f.write("\n")
		f.close()

		"""f=open("./contrast_curve_SPHERE/companion_extraction.txt","w")
		f.write("rho\terho\tPA\tePA\tdmag\tedmag\n")
		f.write(str(round(rho,4)))
		f.write("\t")
		f.write(str(round(erho,4)))
		f.write("\t")
		f.write(str(round(PA,3)))
		f.write("\t")
		f.write(str(round(ePA,3)))
		f.write("\t")
		f.write(str(round(dmag,3)))
		f.write("\t")
		f.write(str(round(edmag,3)))
		f.write("\n")
		f.close()"""
	else:
		print "No companion detected so no extraction"
