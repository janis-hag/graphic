#!/usr/bin/python
import numpy as np
import scipy, time, glob, sys, os, shutil
from scipy import fftpack
import pylab as py
import pyfftw
import multiprocessing
import astropy.io.fits as pyfits
from mpi4py import MPI
import graphic_nompi_lib_330 as graphic_nompi_lib
import argparse

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Apply the sdi algorythm on SPHERE (for now only for IRDIS) data and produce sdi_* cubes')
parser.add_argument('--pattern', action="store", dest="pattern",  default="left*SCIENCE_DBI", help='cubes to apply the sdi')
parser.add_argument('-additional', action="store_const", dest="additional", const=True,  default=False, help='if True produce in addition a cube of the left image rescaled (the one used to do the subtraction with right image)')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",
					default='all_info', help='Info filename pattern.')
parser.add_argument('--info_dir', action="store", dest="info_dir",
					default='cube-info', help='Info directory')

args = parser.parse_args()
additional=args.additional
pattern=args.pattern
info_pattern=args.info_pattern
info_dir=args.info_dir

if rank==0:
	def sdi(im1_3d,im2_3d,lambda1,lambda2,additional):
	    """
	    take im as an image and rescale it (make it bigger) by a factor lambda2/lambda1
	    input: im=the image to rescale, lambda1= the wavelength of the first filter, lambda2= the wavelength of the second filter, 
	    mask_apodisation= apodisation matrix for the fourier transform so we don't add high frequency when we come back to the image plan
	    output: im2= image with more pixels (2*nbr_pix) but with the same proportions than im,
	    im3= image rescaled and with the same number of pixels than im
	    """
	    sys.stdout.write('\n scaling the cube... ')
	    sys.stdout.flush()
	    mask_nan=np.where(np.isnan(im1_3d),np.nan,1.)
	    
	    im1_3d=np.nan_to_num(im1_3d).astype(float)
	    
	    # l=256 #2xl is the size of the inner part of the image taken to calculate the flux ratio between the to filters
	    # nbr_pix=int(l*lambda2/lambda1-l)
	    # flux_factor1=np.mean(im2_3d[:,np.shape(im2_3d)[1]/2.-l-nbr_pix:np.shape(im2_3d)[1]/2.+l+nbr_pix,np.shape(im2_3d)[2]/2.-l-nbr_pix:np.shape(im2_3d)[2]/2.+l+nbr_pix])/np.mean(im1_3d[:,np.shape(im1_3d)[1]/2.-l:np.shape(im1_3d)[1]/2.+l,np.shape(im1_3d)[2]/2.-l:np.shape(im1_3d)[2]/2.+l])
	    # im1_3d=im1_3d*flux_factor1
	    
	    
	    shape=np.shape(im1_3d)
	    
	    #"0 padding in the image plan to make the images in a power of 2 shape"
	    if (pow(2, np.ceil(np.log(np.shape(im1_3d)[1])/np.log(2)))-np.shape(im1_3d)[1])/2.!=0:
		nbr_pix=round(((pow(2, np.ceil(np.log(np.shape(im1_3d)[1])/np.log(2)))-np.shape(im1_3d)[1])/2.+np.shape(im1_3d)[1]/2.*(1-(lambda2/lambda1)))/(lambda2/lambda1))
		mat=np.zeros(((np.shape(im1_3d)[0],np.shape(im1_3d)[1],nbr_pix)))
		mat2=np.zeros(((np.shape(im1_3d)[0],np.shape(im1_3d)[1]+2*nbr_pix,nbr_pix)))
		im1_3d=np.append(mat,im1_3d,axis=2)
		im1_3d=np.append(im1_3d,mat,axis=2)
		im1_3d=np.transpose(np.append(np.transpose(im1_3d,axes=(0,2,1)),mat2,axis=2),axes=(0,2,1))
		im1_3d=np.transpose(np.append(mat2,np.transpose(im1_3d,axes=(0,2,1)),axis=2),axes=(0,2,1))
	    shape_bis=np.shape(im1_3d)
	    
	    # "preparing the fourier transform"
	    
	    #with fftw
	    pyfftw.interfaces.cache.enable()
	    pyfftw.interfaces.cache.set_keepalive_time(30)
	    test = pyfftw.n_byte_align_empty((shape_bis[1],shape_bis[2]), 16, 'complex128')
	    test = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(test, planner_effort='FFTW_MEASURE', threads=4))
	    # "fourier transforming the cube"

	    fft_3d=pyfftw.n_byte_align_empty(shape_bis, 16, 'complex128')
	    for i in range(shape_bis[0]):
		fft_3d[i,:,:] = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(im1_3d[i,:,:], planner_effort='FFTW_MEASURE', threads=4))
	    
	    #with scipy.fftpack
	    #fft_3d=fftpack.fftshift(fftpack.fft2(im1_3d))
	    # "time of the process:  ",time.time()-t0
	    
	    #print "apodising the cube "
	    #mask_apodisation=apodisation(np.shape(fft_3d)[1],np.shape(fft_3d)[1]/10.)
	    #fft_3d=fft_3d*mask_apodisation # apodisation des haute frequence en cas d image avec mauvais pixels ou haute frequences
	    
	    # "0 padding in fourier space to rescale images"
	    nbr_pix=int((int((lambda2/lambda1)*np.shape(im1_3d)[1])-np.shape(im1_3d)[1])/2.)
	    mat=np.zeros(((np.shape(fft_3d)[0],np.shape(fft_3d)[1],nbr_pix)))
	    mat2=np.zeros(((np.shape(fft_3d)[0],np.shape(fft_3d)[1]+2*nbr_pix,nbr_pix)))
	    fft_3d=np.append(mat,fft_3d,axis=2)
	    fft_3d=np.append(fft_3d,mat,axis=2)
	    fft_3d=np.transpose(np.append(np.transpose(fft_3d,axes=(0,2,1)),mat2,axis=2),axes=(0,2,1))
	    fft_3d=np.transpose(np.append(mat2,np.transpose(fft_3d,axes=(0,2,1)),axis=2),axes=(0,2,1))
	    
	    # "preparing the inverse fourier transform"
	    
	    #with fftw
	    test = pyfftw.n_byte_align_empty((np.shape(fft_3d)[1],np.shape(fft_3d)[2]), 16, 'complex128')
	    test = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(test, planner_effort='FFTW_MEASURE', threads=4))
	    # "inverse fourier transforming the cube"
	    im1_3d_rescale=pyfftw.n_byte_align_empty(np.shape(fft_3d), 16, 'complex128')
	    for i in range(np.shape(fft_3d)[0]):
		im1_3d_rescale[i,:,:]=pyfftw.interfaces.scipy_fftpack.ifft2(fftpack.ifftshift(fft_3d[i,:,:]), planner_effort='FFTW_MEASURE', threads=4)
	    im1_3d_rescale=np.real(im1_3d_rescale)
	    
	    #with scipy.fftpack
	    #im1_3d_rescale=np.real(fftpack.ifft2(fftpack.ifftshift(fft_3d)))
	    
	    im1_3d_rescale_cut=im1_3d_rescale[:,np.shape(im1_3d_rescale)[1]/2.-shape[1]/2.:np.shape(im1_3d_rescale)[1]/2.+shape[1]/2.,np.shape(im1_3d_rescale)[2]/2.-shape[2]/2.:np.shape(im1_3d_rescale)[2]/2.+shape[2]/2.]
	    im1_3d_rescale_cut=im1_3d_rescale_cut*mask_nan
		
	    #applying a donut shape mask on the central part to compute the flux_factor
	    r_int=30
	    r_ext=80
	    l=np.shape(im1_3d_rescale_cut)[1]
	    x=np.arange(-l/2.,l/2.)
	    y=np.arange(-l/2.,l/2.)
	    X,Y=np.meshgrid(x,y)
	    R1=np.sqrt(X**2+Y**2)
	    donut=np.where(R1>r_int,1,np.nan)
	    donut=np.where(R1>r_ext,np.nan,donut)
	    # flux_factor=np.nanmean(np.copy(im2_3d)*donut)/np.nanmean(np.copy(im1_3d_rescale_cut)*donut)
	    # flux_factor=(np.shape(fft_3d)[1]*np.shape(fft_3d)[2])/(float(shape_bis[1]*shape_bis[2]))
	    # im1_3d_rescale_cut=im1_3d_rescale_cut*flux_factor

	    # Instead, let's scale the flux of the H3 image, since H2 is where the planet flux should be.
	    flux_factor=np.nanmean(np.copy(im1_3d_rescale_cut)*donut)/np.nanmean(np.copy(im2_3d)*donut)
	    im2_3d*=flux_factor

	    # print 'Flux factor:',flux_factor
	    
	    if lambda1==1189.5 or lambda1==1025.8: #absorbtion in Y2 and J2 so companion is positive with this
		im_3d_subtracted=im2_3d-im1_3d_rescale_cut
	    else: #absorption in K2 and H3 so companion is positive with this
		im_3d_subtracted=im1_3d_rescale_cut-im2_3d

	    # "end of scaling"
	    if additional:
		return im_3d_subtracted,im1_3d_rescale_cut,flux_factor
	    else:
		return im_3d_subtracted,flux_factor

	#additional=False
	#key_word="left*SCIENCE_DBI*"

	t0=MPI.Wtime()
	print("beginning of sdi:")

	length=0


	sys.stdout.write('Application du sdi sur:')
	sys.stdout.write('\n')
	#for allfiles in glob.iglob(key_word):
	for allfiles in glob.iglob(pattern+'*'):
	    sys.stdout.write(allfiles)
	    sys.stdout.write('\n')
	    sys.stdout.write(allfiles.replace('left','right'))
	    sys.stdout.write('\n')
	    length+=1
	sys.stdout.flush()

	count=1
	#for allfiles in glob.iglob(key_word):
	for allfiles in glob.iglob(pattern+'*'):
	    sys.stdout.write('\n')
	    sys.stdout.write('\r Cube ' + str(count) + '/' + str(length))
	    sys.stdout.flush()
	    cube_left,hdr_l=pyfits.getdata(allfiles,header=True)
	    cube_right,hdr_r=pyfits.getdata(allfiles.replace('left','right'),header=True)
	    
	    if hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_H23':
	    	lambda1=1588.8
		lambda2=1667.1
	    elif hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_K12':
	    	lambda1=2102.5
		lambda2=2255
	    elif hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_Y23':
	    	lambda1=1025.8
		lambda2=1080.2
	    elif hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_J23':
	    	lambda1=1189.5
		lambda2=1269.8
	    if count==1:
		sys.stdout.write('\n')
	    	sys.stdout.write('filter: ' + hdr_l["HIERARCH ESO INS COMB IFLT "])
	    	sys.stdout.flush()
	    
	    if additional:
	    	cube_subtracted,cube_rescale_cut,flux_factor=sdi(cube_left,cube_right,lambda1,lambda2,additional)
	    	# Save the factor used to scale the flux in the left channel
	    	hdr_l['HIERARCH GC SDI Flux rescaling:']=flux_factor
	    	hdr_l['HIERARCH GC SDI Channel rescaled:']='right'
	    	pyfits.writeto(allfiles.replace('left','sdi'),cube_subtracted,header=hdr_l,clobber=True)
	    	pyfits.writeto(allfiles.replace('left','left_rescale'),cube_rescale_cut,header=hdr_l,clobber=True)
	    else:
	    	cube_subtracted,flux_factor=sdi(cube_left,cube_right,lambda1,lambda2,additional)
	    	# Save the factor used to scale the flux in the left channel
	    	hdr_l['HIERARCH GC SDI Flux rescaling:']=flux_factor
	    	hdr_l['HIERARCH GC SDI Channel rescaled:']='right'
	    	pyfits.writeto(allfiles.replace('left','sdi'),cube_subtracted,header=hdr_l,clobber=True)
	    count+=1
	for allfiles in glob.iglob(info_dir+'/'+info_pattern+'*'):
		shutil.copyfile(allfiles,allfiles.replace('left','sdi'))


	print("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
	print("sdi finished")
	sys.exit(0)
	##sys.exit(0)
else:
	sys.exit(0)
