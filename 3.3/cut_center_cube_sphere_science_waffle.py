#!/usr/bin/python
import numpy as np
import scipy, glob, sys, os
import astropy.io.fits as pyfits
from scipy import ndimage,signal,fftpack
import pyfftw
import multiprocessing
from mpi4py import MPI
import graphic_nompi_lib_330 as graphic_nompi_lib
import argparse

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Cut and center (with the file created by the star center routine) the cubes for left and right images in SPHERE data')
parser.add_argument('--pattern', action="store", dest="pattern",  default="cl_nomed_SPHER*SCIENCE_DBI", help='cubes to apply the cut and centering')
parser.add_argument('-science_waffle', dest='science_waffle', action='store_const', const=True, default=False, help='Switch to science frame with waffle (usually for high precision astrometry)')
parser.add_argument('-ifs', dest='ifs', action='store_const',const=True,default=False, help='Switch for IFS data, which only need to be centered')

args = parser.parse_args()
pattern=args.pattern
science_waffle=args.science_waffle
ifs=args.ifs

if rank==0:
    def translat_cube(cube,x,y):
        cube=np.nan_to_num(cube).astype(float)
        k=fftpack.fftfreq(np.shape(cube)[1])
        #with fftw
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)
        plan_fft = pyfftw.n_byte_align_empty((np.shape(cube)[1],np.shape(cube)[2]), 16, 'complex128')
        plan_fft = pyfftw.interfaces.scipy_fftpack.fft2(plan_fft, planner_effort='FFTW_MEASURE', threads=4)
        fft=pyfftw.n_byte_align_empty((np.shape(cube)[1],np.shape(cube)[2]), 16, 'complex128')
        fft_translate=pyfftw.n_byte_align_empty((np.shape(cube)[1],np.shape(cube)[2]), 16, 'complex128')
        cube_translate=pyfftw.n_byte_align_empty(np.shape(cube), 16, 'complex128')
        for i in range(np.shape(cube)[0]):
            fft = pyfftw.interfaces.scipy_fftpack.fft2(cube[i,:,:], planner_effort='FFTW_MEASURE', threads=4)
            if np.size(x)>1:#in case of science_waffle we center each frames of the cube individually
                fft_translate=np.transpose(np.transpose(fft*np.exp(-2*np.pi*1j*(x[i]*k)))*np.exp(-2*np.pi*1j*(y[i]*k)))
            else:
                fft_translate=np.transpose(np.transpose(fft*np.exp(-2*np.pi*1j*(x*k)))*np.exp(-2*np.pi*1j*(y*k)))
            cube_translate[i,:,:]=pyfftw.interfaces.scipy_fftpack.ifft2(fft_translate, planner_effort='FFTW_MEASURE', threads=4)
        cube_translate=np.real(cube_translate)
        return cube_translate

    
    def cut_irdis_cube(cube,hdr):
        '''
        Cuts an IRDIS data cube into left and right images, and masks out the known
        bad pixels at the same time
        '''
        
        #cutting the cube in 2
        im_left=cube[:,:,:1024]
        im_right=cube[:,:,1024:]
        mask_nan_l=np.where(np.isnan(im_left[0]),np.nan,1)
        mask_nan_r=np.where(np.isnan(im_right[0]),np.nan,1)
        
        ################## getting rid of the vertical stripes
        x=np.arange(-np.shape(cube)[1]/2,np.shape(cube)[1]/2)
        y=x
        X,Y=np.meshgrid(x,y)
        R=np.sqrt(X**2+Y**2)
        mask_med=np.where(R<448,np.nan,1)
        med_left_up=np.nanmedian((im_left*mask_med)[:,:512,:],axis=1)
        med_right_up=np.nanmedian((im_right*mask_med)[:,:512,:],axis=1)
        med_left_down=np.nanmedian((im_left*mask_med)[:,512:,:],axis=1)
        med_right_down=np.nanmedian((im_right*mask_med)[:,512:,:],axis=1)
        
        for i in range(np.shape(cube)[0]):
            im_left[i,:512,:]=im_left[i,:512,:]-med_left_up[i,:]
            im_right[i,:512,:]=im_right[i,:512,:]-med_right_up[i,:]
            im_left[i,512:,:]=im_left[i,512:,:]-med_left_down[i,:]
            im_right[i,512:,:]=im_right[i,512:,:]-med_right_down[i,:]

        ##################
        im_left=np.nan_to_num(im_left)
        im_right=np.nan_to_num(im_right)
        
        #cutting above the cut off frequency
        print "\n cutting above the cut off frequency"
        x=np.arange(-512,512)
        X,Y=np.meshgrid(x,x)
        R=np.sqrt(X**2+Y**2)
        d_ap=100

        if hdr["HIERARCH ESO INS COMB IFLT"]=='DB_Y23':
            r_l=486.14
            r_r=461.66
        elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_J23':
            r_l=419.06
            r_r=391.74
        elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_H23':
            r_l=313.047
            r_r=299.15
        elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_K12':
            r_l=237.19
            r_r=221.15
        else:
            print 'Problem, no filter mode not detected!'

        mask_l=np.where(R>=r_l-d_ap/2.,(1.+np.cos(np.pi+(np.pi)*(r_l-d_ap/2.-R+d_ap)/d_ap))/2.,1)
        mask_l=np.where(R>r_l-d_ap/2.+d_ap,0,mask_l)
        mask_r=np.where(R>=r_r-d_ap/2.,(1.+np.cos(np.pi+(np.pi)*(r_r-d_ap/2.-R+d_ap)/d_ap))/2.,1)
        mask_r=np.where(R>r_r-d_ap/2.+d_ap,0,mask_r)
        
        # Get rid of high frequency structure (e.g. bad pixels) using a lowpass filter
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)
        for i in range(np.shape(im_left)[0]):
            fft_l = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(im_left[i,:,:], threads=4))*mask_l
            im_left[i,:,:]=pyfftw.interfaces.scipy_fftpack.ifft2(fftpack.ifftshift(fft_l), threads=4)
            fft_r = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(im_right[i,:,:], threads=4))*mask_r
            im_right[i,:,:]=pyfftw.interfaces.scipy_fftpack.ifft2(fftpack.ifftshift(fft_r), threads=4)

        # Mask out the pixels outside of the field of view
        im_left=im_left*mask_nan_l
        im_right=im_right*mask_nan_r

        # Put the left and right images back together into a 4D cube
        cube=np.array([im_left,im_right])
        # And the NaN mask
        mask_nan=np.array([mask_nan_l,mask_nan_r])
        
        return cube,mask_nan

    def clean_ifs_cube(cube,hdr,square_length=220,lenslet_rotation=-10.7,cutoff_radius=120,
                       cutoff_region=50., lowpass_filter=False):
        ''' Function to clean up the IFS cubes and make a NaN mask that shows which
        regions are outside of the field of view.
        The images are lowpass filtered using a cutoff_radius
        
        square_length is the size of the IFS field of view in pixels
        lenslet_rotation is the rotation of the square field of view with respect to the detector
        
        cutoff_radius is the radius (in pixels) used for the lowpass filter
        cutoff_region is the size of the region used to slowly ramp from 1. to 0. in the filter

        10.7deg is from the SPHERE manual v4.
        220 pix seemed to fit ok
        
        '''
        
        # First make a square that defines the actual field of view
        mask_nan=np.nan+np.zeros((cube.shape[-2],cube.shape[-1])) # Just 1s for now
        cen=[mask_nan.shape[0]/2,mask_nan.shape[1]/2]
        mask_nan[cen[0]-square_length/2:cen[0]+square_length/2,
                   cen[1]-square_length/2:cen[1]+square_length/2]=1
        
        # Now rotate the square by the lenslet rotation
        mask_nan=graphic_nompi_lib.fft_rotate(mask_nan,lenslet_rotation)
        mask_nan[np.isnan(mask_nan) == False]=1.
        
        # But the square is clipped at the top and bottom
        mask_nan[259:,:]=np.nan
        mask_nan[0:33,:]=np.nan
        
        # Make the mask the right size
        mask_nan=np.repeat(mask_nan[np.newaxis,:,:],repeats=cube.shape[0],axis=0)
        
        if lowpass_filter:
            #cutting above the cut off frequency
            print "\n cutting above the cut off frequency"
            x=np.arange(-cube.shape[-2]/2,cube.shape[-1]/2)
            X,Y=np.meshgrid(x,x)
            R=np.sqrt(X**2+Y**2)
        
            mask=np.where(R>=(cutoff_radius-cutoff_region/2.),
                    (1.+np.cos(np.pi+(np.pi)*(cutoff_radius-cutoff_region/2.-R+cutoff_region)/cutoff_region))/2.,1)
            mask=np.where(R>(cutoff_radius-cutoff_region/2.+cutoff_region),0,mask)
            
            # Get rid of high frequency structure (e.g. bad pixels) using a lowpass filter
            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(30)
            for i in range(np.shape(cube)[0]):
                fft_l = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(cube[i,:,:], threads=4))*mask
                cube[i,:,:]=np.real(pyfftw.interfaces.scipy_fftpack.ifft2(fftpack.ifftshift(fft_l), threads=4))
        
        cube=cube*mask_nan
        
        return cube,mask_nan

    ###############

    def cut_and_center_cub(file_name,science_waffle=False,ifs=False):
        """
        cut an IRDIS cube in left and right image, center them with the file produced by star_center_sphere.py

        Input: file name of the cube

        Output: produce 2 files, *_left.fits and *_right.fits with the cubes centered
        """
        # Read the cube
        cube,hdr = pyfits.getdata(file_name, header=True)

        """#applying a mask with median value on the dead pixels zone (1 on each filter)
        cube[:,351:374,386:402]=np.nan
        cube[:,351:374,386:402]=np.nanmedian(cube[:,350:375,385:403])
        cube[:,307:316,1395:1404]=np.nan
        cube[:,307:316,1395:1404]=np.nanmedian(cube[:,306:317,1394:1405])"""
        
        # Send the cube to be cut if it is IRDIS data
        if not ifs:
            cube,mask_nan = cut_irdis_cube(cube,hdr)

            # Get the detector dithering position from the headers.
            dithering_x = hdr["HIERARCH ESO INS1 DITH POSX"]
            dithering_y = hdr["HIERARCH ESO INS1 DITH POSY"]
            
            n_wav = 2 # the number of wavelengths
        else:
            # IFS does not dither
            dithering_x = 0
            dithering_y = 0
            
            # Make sure the cube is 4D and get the number of wavelength channels
            if cube.ndim == 3:
                cube = cube[np.newaxis,:,:,:]
                n_wav = 1
            else:
                n_wav = cube.shape[0]
                
            # Make a NaN mask for the IFS field of view
            cube,mask_nan=clean_ifs_cube(cube,hdr)

        # Now cube is 4D: n_wav x n_frames x nX x nY
        
        #reading the star center file
        with open('star_center.txt','r') as f:
            lines = f.readlines()
        
        if science_waffle:
            # This part won't work for IFS yet...
            if ifs:
                # To fix it, make a loop over wavelength instead of assuming left and right channels
                # However this requires knowing the format of the star_center.txt file for IFS waffle data
                # and this wasn't clearly defined at the time ACC added IFS support
                raise Exception("Someone needs to fix the science_waffle mode for IFS in cut_center_cube")
                
            next_cube=0
            end_of_file=0
            for i,line in enumerate(lines):
                if i+1==np.size(lines): #testing if we are at the end of the star_center file
                    index_next_file=i+1
                    break 
                elif next_cube==1: #testing if we are at the end of a cube
                    if "SPHER" in line.strip().split()[0]:
                        index_next_file=i
                        break
                if line.strip().split()[0]==file_name:
                    index_file=i
                    next_cube=1
            left_center=np.zeros(((index_next_file-index_file-1)/2,2))
            right_center=np.zeros(((index_next_file-index_file-1)/2,2))
            for i in range((index_next_file-index_file-1)/2):
                left_center_temp=np.array([np.float(lines[index_file+2*i+1].strip().split()[1]),np.float(lines[index_file+2*i+1].strip().split()[2])])
                right_center_temp=np.array([np.float(lines[index_file+2*i+2].strip().split()[1]),np.float(lines[index_file+2*i+2].strip().split()[2])])
                left_center[i,:]=left_center_temp
                right_center[i,:]=right_center_temp
            
            # Put the wavelength channels together into 1 array
            center = np.array([left_center,right_center])
        else:
            center = np.zeros((n_wav,2))
            for wav in range(n_wav):
                center[wav,:] = [np.float(lines[wav+2].strip().split()[1]),np.float(lines[wav+2].strip().split()[2])]

        print('centering cube')

        # Set up the output array
        cube_translat = 0*cube
        
        # Loop through wavelengths and centre the cubes
        for wav in range(n_wav):
            
            if science_waffle:
                cube_translat[wav] = translat_cube(cube[wav],np.shape(cube)[-1]/2.-0.5-center[wav,:,0]-dithering_x,
                            np.shape(cube)[-2]/2.-0.5-center[wav,:,1]-dithering_y)
            else:
                cube_translat[wav] = translat_cube(cube[wav],np.shape(cube)[-1]/2.-0.5-center[wav,0]-dithering_x,
                            np.shape(cube)[-2]/2.-0.5-center[wav,1]-dithering_y)

        # find the position of the NaN mask after translation of the image.
        # IF science_waffle we use the first frame as the position will change
        # less than a pixel and so does not change for the mask

        for wav in range(n_wav):
            if science_waffle:
                
                # Calculate how many pixels to shift the NaN mask
                nbr_pix1 = np.int(np.round((center[wav,0,1]-np.shape(cube[wav])[-2]/2.)))
                nbr_pix2 = np.int(np.round((center[wav,0,0]-np.shape(cube[wav])[-1]/2.)))
            else:
                # Calculate how many pixels to shift the NaN mask
                nbr_pix1 = np.int(np.round((center[wav,1]-np.shape(cube[wav])[-2]/2.)))
                nbr_pix2 = np.int(np.round((center[wav,0]-np.shape(cube[wav])[-1]/2.)))
                
            # Shift it
            mask_nan[wav] = np.roll(mask_nan[wav],-nbr_pix1,axis=-2)
            mask_nan[wav] = np.roll(mask_nan[wav],-nbr_pix2,axis=-1)
            
            # Then NaN out the extra rows and columns from outside the array that shifted in
            if nbr_pix1 > 0:
                mask_nan[wav,-nbr_pix1:] = np.nan
            else:
                mask_nan[wav,0:-nbr_pix1] = np.nan
            if nbr_pix2 > 0:
                mask_nan[wav,:,-nbr_pix2:] = np.nan
            else:
                mask_nan[wav,:,0:-nbr_pix2] = np.nan
        
        # Apply the new mask to the images
        for wav in range(n_wav):
            cube_translat[wav] *= mask_nan[wav]

        # Update the NX header keyword
        hdr['HIERARCH ESO DET CHIP1 NX'] = cube_translat.shape[-1]
        
        # Write out the images for each wavelength separately
        if not ifs:
            prefixes = ['left_','right_']
        else:
            if n_wav == 1:
                prefixes = ['cen_'] # if there's only 1 wavelength channel, we don't need to specify what wavelength it is
            else:
                prefixes = ['wav'+str(wav)+'_' for wav in range(n_wav)]

            
        for wav in range(n_wav):

            # Add a header keyword saying which number channel this is (left/right = 0/1). For IFS, this should already be added.
            #  So only do it for IRDIS (where n_wav = 2)
            if 'HIERARCH GC WAVE CHANNEL' not in hdr and (n_wav == 2):
                hdr['HIERARCH GC WAVE CHANNEL'] = wav

                # And save the wavelength as well to make it easier later
                filter_hdr = hdr['HIERARCH ESO INS COMB IFLT']
                ##### H band
                if filter_hdr == 'DB_H23':
                    filter_name = ['H2','H3'][wav]
                elif filter_hdr == 'DB_H32':
                    filter_name = ['H3','H2'][wav]
                elif filter_hdr == 'DB_H34':
                    filter_name = ['H3','H4'][wav]
                ##### K band
                elif filter_hdr == 'DB_K12':
                    filter_name = ['K1','K2'][wav]
                ##### Y band
                elif filter_hdr == 'DB_Y23':
                    filter_name = ['Y2','Y3'][wav]
                ##### J band
                elif filter_hdr == 'DB_J23':
                    filter_name = ['J2','J3'][wav]
                else:
                    raise Exception('ERROR! Unknown filter in cut_center_cube_sphere_science_waffle.py!')

                # The filter wavelenth file should be in a subdirectory from the location of this file
                filter_wavelength_file = os.path.dirname(os.path.realpath(__file__))+os.sep+'SPHERE_characterization/photometry_SPHERE/filter_wavelength.dat'
                filters,filter_wavs = np.loadtxt(filter_wavelength_file,skiprows=2,unpack=True,dtype=str)
                filter_wavs = filter_wavs.astype(np.float64)
                filter_wav = filter_wavs[filters == filter_name][0]

                hdr['HIERARCH GC WAVELENGTH'] = (filter_wav*1e-3,'Wavelength in microns')
                
            out_name = prefixes[wav]+file_name
            pyfits.writeto(out_name,cube_translat[wav],header=hdr,output_verify='warn',clobber=True)



    t0=MPI.Wtime()
    sys.stdout.write("beginning of cut and centering cubes:")
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    length=0
    sys.stdout.write('files to cut and center:')
    sys.stdout.write('\n')
    for allfiles in glob.iglob(pattern+'*'):
        sys.stdout.write(allfiles)
        sys.stdout.write('\n')
        length+=1
    sys.stdout.flush()
    
    count=1
    for allfiles in glob.iglob(pattern+'*'):
        sys.stdout.write('\r Cube ' + str(count) + '/' + str(length)+"\n")
        sys.stdout.flush()
        if science_waffle:
            cut_and_center_cub(allfiles,science_waffle,ifs=ifs)
        else:
            cut_and_center_cub(allfiles,ifs=ifs)
        count+=1


    sys.stdout.write("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
    sys.stdout.write("\n")
    sys.stdout.write("Cut and center finished")
    sys.stdout.write("\n")
    sys.stdout.flush()
    # os._exit(1)
    sys.exit(0)
else:
    sys.exit(0)
