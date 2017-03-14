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
parser.add_argument('--r_int', action="store", dest="r_int",  default=30, type=int, help='Interior radius (in pixels) for the area used to calculate the flux ratio between the cubes')
parser.add_argument('--r_ext', action="store", dest="r_ext",  default=80, type=int, help='Exterior radius (in pixels) for the area used to calculate the flux ratio between the cubes')


args = parser.parse_args()
additional = args.additional
pattern = args.pattern
info_pattern = args.info_pattern
info_dir = args.info_dir
r_int = args.r_int
r_ext = args.r_ext

def rescale_image(im1_3d,lambda1,lambda2):
    ''' Rescales an image using Fourier transforms
    im1_3d: Input image cube to be scaled
    lambda1: the wavelength of the input cube
    lambda2: the reference wavelength (i.e. the wavelength we want to scale the image to)
    '''

    # Find the NaNs in the image
    mask_nan=np.where(np.isnan(im1_3d),0,1.)
    im1_3d=np.nan_to_num(im1_3d).astype(float)
    
    shape=np.shape(im1_3d)
    
    #"0 padding in the image plan to make the images in a power of 2 shape"
    if (pow(2, np.ceil(np.log(np.shape(im1_3d)[1])/np.log(2)))-np.shape(im1_3d)[1])/2.!=0:
        nbr_pix=np.int(round(((pow(2, np.ceil(np.log(np.shape(im1_3d)[1])/np.log(2)))-np.shape(im1_3d)[1])/2.+np.shape(im1_3d)[1]/2.*(1-(lambda2/lambda1)))/(lambda2/lambda1)))
        mat=np.zeros(((np.shape(im1_3d)[0],np.shape(im1_3d)[1],nbr_pix)))
        mat2=np.zeros(((np.shape(im1_3d)[0],np.shape(im1_3d)[1]+2*nbr_pix,nbr_pix)))
        im1_3d=np.append(mat,im1_3d,axis=2)
        im1_3d=np.append(im1_3d,mat,axis=2)
        im1_3d=np.transpose(np.append(np.transpose(im1_3d,axes=(0,2,1)),mat2,axis=2),axes=(0,2,1))
        im1_3d=np.transpose(np.append(mat2,np.transpose(im1_3d,axes=(0,2,1)),axis=2),axes=(0,2,1))
    
        # and the NaN Mask        
        mask_nan=np.append(mat,mask_nan,axis=2)
        mask_nan=np.append(mask_nan,mat,axis=2)
        mask_nan=np.transpose(np.append(np.transpose(mask_nan,axes=(0,2,1)),mat2,axis=2),axes=(0,2,1))
        mask_nan=np.transpose(np.append(mat2,np.transpose(mask_nan,axes=(0,2,1)),axis=2),axes=(0,2,1))
    shape_bis=np.shape(im1_3d)
    
    
    #FFT the data
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)
    
    # "fourier transforming the cube"
    fft_3d=pyfftw.n_byte_align_empty(shape_bis, 16, 'complex128')
    fft_nan_mask = pyfftw.n_byte_align_empty(shape_bis, 16, 'complex128')
    for i in range(shape_bis[0]):
        fft_3d[i,:,:] = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(im1_3d[i,:,:], planner_effort='FFTW_MEASURE', threads=4))
        fft_nan_mask[i,:,:] = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(mask_nan[i,:,:], planner_effort='FFTW_MEASURE', threads=4))
    
    
    # "0 padding in fourier space to rescale images"
    nbr_pix=int((int((lambda2/lambda1)*np.shape(im1_3d)[1])-np.shape(im1_3d)[1])/2.)
    if nbr_pix > 0:
        # Make some arrays of zeros to append to the data
        mat=np.zeros(((np.shape(fft_3d)[0],np.shape(fft_3d)[1],nbr_pix)))
        mat2=np.zeros(((np.shape(fft_3d)[0],np.shape(fft_3d)[1]+2*nbr_pix,nbr_pix)))
        # Add them at the start and end
        fft_3d=np.append(mat,fft_3d,axis=2) # add zeros to the start of dim 2
        fft_3d=np.append(fft_3d,mat,axis=2) # add zeros to the end of dim 2
        fft_3d=np.transpose(np.append(np.transpose(fft_3d,axes=(0,2,1)),mat2,axis=2),axes=(0,2,1)) # add zeros to the end of dim 1
        fft_3d=np.transpose(np.append(mat2,np.transpose(fft_3d,axes=(0,2,1)),axis=2),axes=(0,2,1)) # add zeros to the start of dim 1
        
        # And the NaN mask
        fft_nan_mask=np.append(mat,fft_nan_mask,axis=2) # add zeros to the start of dim 2
        fft_nan_mask=np.append(fft_nan_mask,mat,axis=2) # add zeros to the end of dim 2
        fft_nan_mask=np.transpose(np.append(np.transpose(fft_nan_mask,axes=(0,2,1)),mat2,axis=2),axes=(0,2,1)) # add zeros to the end of dim 1
        fft_nan_mask=np.transpose(np.append(mat2,np.transpose(fft_nan_mask,axes=(0,2,1)),axis=2),axes=(0,2,1)) # add zeros to the start of dim 1
    else:
        # Or remove pixels in Fourier space to rescale the image
        fft_3d = fft_3d[:,-nbr_pix:nbr_pix,-nbr_pix:nbr_pix]
        fft_nan_mask = fft_nan_mask[:,-nbr_pix:nbr_pix,-nbr_pix:nbr_pix]
    
    # "preparing the inverse fourier transform"
    
    #with fftw
    # "inverse fourier transforming the cube"
    im1_3d_rescale=pyfftw.n_byte_align_empty(np.shape(fft_3d), 16, 'complex128')
    nan_mask_rescale = pyfftw.n_byte_align_empty(np.shape(fft_3d), 16, 'complex128')
    for i in range(np.shape(fft_3d)[0]):
        im1_3d_rescale[i,:,:]=pyfftw.interfaces.scipy_fftpack.ifft2(fftpack.ifftshift(fft_3d[i,:,:]), planner_effort='FFTW_MEASURE', threads=4)
        nan_mask_rescale[i,:,:]=pyfftw.interfaces.scipy_fftpack.ifft2(fftpack.ifftshift(fft_nan_mask[i,:,:]), planner_effort='FFTW_MEASURE', threads=4)
    im1_3d_rescale=np.real(im1_3d_rescale)
    nan_mask_rescale = np.real(nan_mask_rescale)
    
    # Cut the nan_mask and cube to the right size
    if nbr_pix >= 0:
        min_x = np.int( np.shape(im1_3d_rescale)[1]/2.-shape[1]/2. )
        max_x = np.int( np.shape(im1_3d_rescale)[1]/2.+shape[1]/2. )
        min_y = np.int( np.shape(nan_mask_rescale)[2]/2.-shape[2]/2. )
        max_y = np.int( np.shape(nan_mask_rescale)[2]/2.+shape[2]/2. )
        # Cut the nan mask and turn it back into NaNs
        nan_mask_rescale = nan_mask_rescale[:, min_x:max_x, min_y:max_y]
        mask_nan = np.where(nan_mask_rescale < 0.5,np.nan,1.)
        # Cut the image
        im1_3d_rescale_cut=im1_3d_rescale[: min_x:max_x, min_y:max_y]
    else:
        # Make a new cube to embed the scaled image into
        out_cube = np.zeros(shape)
        out_nan = np.zeros(shape)*np.NaN

        min_x = -np.int( np.shape(im1_3d_rescale)[1]/2.-shape[1]/2. )
        max_x = np.int( np.shape(im1_3d_rescale)[1]/2.+shape[1]/2. )
        min_y = -np.int( np.shape(nan_mask_rescale)[2]/2.-shape[2]/2. )
        max_y = np.int( np.shape(nan_mask_rescale)[2]/2.+shape[2]/2. )

        # Cut the nan mask and turn it back into NaNs
        out_nan[:, min_x:max_x, min_y:max_y] = nan_mask_rescale
        nan_mask_rescale = out_nan
        mask_nan = np.where(nan_mask_rescale < 0.5,np.nan,1.)
        # Cut the image
        out_cube[:, min_x:max_x, min_y:max_y] = im1_3d_rescale
        im1_3d_rescale_cut = out_cube

    # Multiply by the NaN mask to add the NaNs back in
    im1_3d_rescale_cut = im1_3d_rescale_cut*mask_nan
    
    return im1_3d_rescale_cut

def scale_flux(reference_image,scaled_image,r_int=30,r_ext=80):
    ''' Calculate the factor needed to scale the flux to best subtract the PSF
    r_int and r_ext are the interior and exterior radii of the donut shaped mask used to calculate the flux ratio.
    '''

    # Make a donut shaped mask
    l=np.shape(reference_image)[1]
    x=np.arange(-l/2.,l/2.)
    y=np.arange(-l/2.,l/2.)
    X,Y=np.meshgrid(x,y)
    R1=np.sqrt(X**2+Y**2)
    donut=np.where(R1>r_int,1,np.nan)
    donut=np.where(R1>r_ext,np.nan,donut)
    
    # Calculate the ratio of mean flux in the donut in each image
   # flux_factors = np.nanmean(reference_image*donut,axis=(1,2))/np.nanmean(scaled_image*donut,axis=(1,2))
    # flux_factor = np.nanmedian((reference_image/scaled_image)*donut,axis=(1,2))
    # flux_factor = np.nanmedian((reference_image/scaled_image)*donut)
    flux_factor = np.nanmean((reference_image*donut)/(scaled_image*donut))
    
    return flux_factor

if rank==0:

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

    # Loop through the cubes and run SDI
    for allfiles in glob.iglob(pattern+'*'):
        sys.stdout.write('\n')
        sys.stdout.write('\r Cube ' + str(count) + '/' + str(length))
        sys.stdout.flush()

        # Load the cubes
        cube_left,hdr_l=pyfits.getdata(allfiles,header=True)
        cube_right,hdr_r=pyfits.getdata(allfiles.replace('left','right'),header=True)
        
        # Work out the wavelengths and which cube we want to rescale
        # For H23 and K12, the right channel has the absorption
        # For Y23 and J23, the left channel has the absorption
        if hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_H23':
            reference_lambda=1588.8
            scaled_lambda=1667.1
            reference_cube = cube_left
            scaled_cube = cube_right

            # And some header params
            hdr = hdr_l
            rescaled_channel_name = 'right'

        elif hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_K12':
            reference_lambda=2102.5
            scaled_lambda=2255
            reference_cube = cube_left
            scaled_cube = cube_right

            # And some header params
            hdr = hdr_l
            rescaled_channel_name = 'right'

        elif hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_Y23':
            scaled_lambda=1025.8
            reference_lambda=1080.2
            reference_cube = cube_right
            scaled_cube = cube_left

            # And some header params
            hdr = hdr_r
            rescaled_channel_name = 'left'

        elif hdr_l["HIERARCH ESO INS COMB IFLT "]=='DB_J23':
            scaled_lambda=1189.5
            reference_lambda=1269.8
            reference_cube = cube_right
            scaled_cube = cube_left

            # And some header params
            hdr = hdr_r
            rescaled_channel_name = 'left'

        if count==1:
            sys.stdout.write('\n')
            sys.stdout.write('filter: ' + hdr_l["HIERARCH ESO INS COMB IFLT "])
            sys.stdout.flush()

        # Rescale all of the cubes to the reference wavelength
        rescaled_cube = rescale_image(scaled_cube,scaled_lambda,reference_lambda)

        # Work out the flux scaling factor needed to best subtract the PSF
        flux_factor = scale_flux(reference_cube,rescaled_cube,r_int=r_int,r_ext=r_ext)
        hdr['HIERARCH GC SDI Flux rescaling:']=flux_factor # Save it in the header
        hdr['HIERARCH GC SDI Channel rescaled:']= rescaled_channel_name

        # Now do the subtraction
        sdi_cube = reference_cube - flux_factor*rescaled_cube
        sys.stdout.write('\n')
        sys.stdout.write('  Scaling flux of '+rescaled_channel_name+' cube by '+str(flux_factor))
        sys.stdout.flush()

        # Write it out
        pyfits.writeto(allfiles.replace('left','sdi'),sdi_cube,header=hdr,clobber=True)

        if additional:
            scaled_cube_output_name = allfiles.replace('left',rescaled_channel_name+'_rescale')
            pyfits.writeto(scaled_cube_output_name,rescaled_cube,header=hdr,clobber=True)
        
        count+=1

    # Copy the info files
    for allfiles in glob.iglob(info_dir+'/'+info_pattern+'*'):
        shutil.copyfile(allfiles,allfiles.replace('left','sdi'))

    sys.stdout.write('\n')
    print("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
    print("sdi finished")
    sys.exit(0)
    ##sys.exit(0)
else:
    sys.exit(0)
