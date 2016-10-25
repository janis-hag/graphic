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

args = parser.parse_args()
pattern=args.pattern
science_waffle=args.science_waffle

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

    def cut_and_center_cub(file_name,science_waffle=False):
        """
        cut an IRDIS cube in left and right image, center them with the file produced by star_center_sphere.py
    
        Input: file name of the cube
    
        Output: produce 2 files, *_left.fits and *_right.fits with the cubes centered
        """
        cube,hdr=pyfits.getdata(file_name, header=True)
    
        """#applying a mask with median value on the dead pixels zone (1 on each filter)
        cube[:,351:374,386:402]=np.nan
        cube[:,351:374,386:402]=np.nanmedian(cube[:,350:375,385:403])
        cube[:,307:316,1395:1404]=np.nan
        cube[:,307:316,1395:1404]=np.nanmedian(cube[:,306:317,1394:1405])"""
    
        dithering_x=hdr["HIERARCH ESO INS1 DITH POSX"]
        dithering_y=hdr["HIERARCH ESO INS1 DITH POSY"]

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
    
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)
        plan_fft = pyfftw.n_byte_align_empty((np.shape(im_left)[1],np.shape(im_left)[2]), 16, 'complex128')
        plan_fft = pyfftw.interfaces.scipy_fftpack.fft2(plan_fft, planner_effort='FFTW_MEASURE', threads=4)
        fft=pyfftw.n_byte_align_empty((np.shape(im_left)[1],np.shape(im_left)[2]), 16, 'complex128')
        fft_translate=pyfftw.n_byte_align_empty((np.shape(im_left)[1],np.shape(im_left)[2]), 16, 'complex128')
        for i in range(np.shape(im_left)[0]):
            fft_l = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(im_left[i,:,:], planner_effort='FFTW_MEASURE', threads=4))*mask_l
            im_left[i,:,:]=pyfftw.interfaces.scipy_fftpack.ifft2(fftpack.ifftshift(fft_l), planner_effort='FFTW_MEASURE', threads=4)
            fft_r = fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(im_right[i,:,:], planner_effort='FFTW_MEASURE', threads=4))*mask_r
            im_right[i,:,:]=pyfftw.interfaces.scipy_fftpack.ifft2(fftpack.ifftshift(fft_r), planner_effort='FFTW_MEASURE', threads=4)
    
        im_left=im_left*mask_nan_l
        im_right=im_right*mask_nan_r

        #reading the star center file
        f=open('star_center.txt','r')
        lines=f.readlines()
        f.close()
        if science_waffle:
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
        else:
            left_center=np.array([np.float(lines[2].strip().split()[1]),np.float(lines[2].strip().split()[2])])
            right_center=np.array([np.float(lines[3].strip().split()[1]),np.float(lines[3].strip().split()[2])])
        left_relative_center=left_center-np.shape(im_left)[1]/2.
        right_relative_center=right_center-np.shape(im_right)[1]/2.
        #mask_nan_l=np.where(np.isnan(im_left[0]),np.nan,1)
        #mask_nan_r=np.where(np.isnan(im_right[0]),np.nan,1)

        #im_translat_left=np.copy(im_left)
        #im_translat_right=np.copy(im_right)
        sys.stdout.write('\n centering left cube')
        sys.stdout.flush()
        if science_waffle:
            im_translat_left=translat_cube(im_left,np.shape(im_left)[2]/2.-0.5-left_center[:,0]-dithering_x,np.shape(im_left)[1]/2.-0.5-left_center[:,1]-dithering_y)
        else:
            im_translat_left=translat_cube(im_left,np.shape(im_left)[2]/2.-0.5-left_center[0]-dithering_x,np.shape(im_left)[1]/2.-0.5-left_center[1]-dithering_y)
        sys.stdout.write('\n centering right cube')
        sys.stdout.write('\n')
        sys.stdout.flush()
        if science_waffle:
            im_translat_right=translat_cube(im_right,np.shape(im_right)[2]/2.-0.5-right_center[:,0]-dithering_x,np.shape(im_right)[1]/2.-0.5-right_center[:,1]-dithering_y)
        else:
            im_translat_right=translat_cube(im_right,np.shape(im_right)[2]/2.-0.5-right_center[0]-dithering_x,np.shape(im_right)[1]/2.-0.5-right_center[1]-dithering_y)

        #finding the position of the mask after translation of the image. IF science_waffle we use the first frame as the position will change of less than a pixel and so dose not change for the mask
        
        #left mask
        if science_waffle:
        	nbr_pix1_l=np.round(abs(left_center[0,1]-np.shape(im_left)[1]/2.))
        	nbr_pix2_l=np.round(abs(left_center[0,0]-np.shape(im_left)[2]/2.))
        	mat1_l=np.zeros((np.shape(im_left)[1]-nbr_pix1_l,nbr_pix2_l))*np.nan
        	mat2_l=np.zeros((np.shape(im_left)[2],nbr_pix1_l))*np.nan
        	if left_relative_center[0,0]>0:
        	    if left_relative_center[0,1]>0:
        	        mask_l=mask_nan_l[nbr_pix1_l:,nbr_pix2_l:]
        	        mask_l=np.append(mask_l,mat1_l,axis=1)
        	        mask_l=np.transpose(np.append(np.transpose(mask_l),mat2_l,axis=1))
        	    elif left_relative_center[0,1]<0:
        	        mask_l=mask_nan_l[:np.shape(im_left)[2]-nbr_pix1_l,nbr_pix2_l:]
        	        mask_l=np.append(mask_l,mat1_l,axis=1)
        	        mask_l=np.transpose(np.append(mat2_l,np.transpose(mask_l),axis=1))
        	elif left_relative_center[0,0]<0:
        	    if left_relative_center[0,1]>0:
        	        mask_l=mask_nan_l[nbr_pix1_l:,:np.shape(im_left)[2]-nbr_pix2_l]
        	        mask_l=np.append(mat1_l,mask_l,axis=1)
        	        mask_l=np.transpose(np.append(np.transpose(mask_l),mat2_l,axis=1))
        	    elif left_relative_center[0,1]<0:
        	        mask_l=mask_nan_l[:np.shape(im_left)[1]-nbr_pix1_l,:np.shape(im_left)[2]-nbr_pix2_l]
        	        mask_l=np.append(mat1_l,mask_l,axis=1)
        	        mask_l=np.transpose(np.append(mat2_l,np.transpose(mask_l),axis=1))
        	
        else:
        	nbr_pix1_l=np.round(abs(left_center[1]-np.shape(im_left)[1]/2.))
		nbr_pix2_l=np.round(abs(left_center[0]-np.shape(im_left)[2]/2.))
		mat1_l=np.zeros((np.shape(im_left)[1]-nbr_pix1_l,nbr_pix2_l))*np.nan
	    	mat2_l=np.zeros((np.shape(im_left)[2],nbr_pix1_l))*np.nan
	    	if left_relative_center[0]>0:
			if left_relative_center[1]>0:
			    mask_l=mask_nan_l[nbr_pix1_l:,nbr_pix2_l:]
			    mask_l=np.append(mask_l,mat1_l,axis=1)
			    mask_l=np.transpose(np.append(np.transpose(mask_l),mat2_l,axis=1))
			elif left_relative_center[1]<0:
			    mask_l=mask_nan_l[:np.shape(im_left)[2]-nbr_pix1_l,nbr_pix2_l:]
			    mask_l=np.append(mask_l,mat1_l,axis=1)
			    mask_l=np.transpose(np.append(mat2_l,np.transpose(mask_l),axis=1))
	    	elif left_relative_center[0]<0:
			if left_relative_center[1]>0:
			    mask_l=mask_nan_l[nbr_pix1_l:,:np.shape(im_left)[2]-nbr_pix2_l]
			    mask_l=np.append(mat1_l,mask_l,axis=1)
			    mask_l=np.transpose(np.append(np.transpose(mask_l),mat2_l,axis=1))
			elif left_relative_center[1]<0:
			    mask_l=mask_nan_l[:np.shape(im_left)[1]-nbr_pix1_l,:np.shape(im_left)[2]-nbr_pix2_l]
			    mask_l=np.append(mat1_l,mask_l,axis=1)
			    mask_l=np.transpose(np.append(mat2_l,np.transpose(mask_l),axis=1))
        
        #right mask
        if science_waffle:
        	nbr_pix1_r=np.round(abs(right_center[0,1]-np.shape(im_right)[1]/2.))
        	nbr_pix2_r=np.round(abs(right_center[0,0]-np.shape(im_right)[2]/2.))
        	mat1_r=np.zeros((np.shape(im_right)[1]-nbr_pix1_r,nbr_pix2_r))*np.nan
        	mat2_r=np.zeros((np.shape(im_right)[2],nbr_pix1_r))*np.nan
        	if right_relative_center[0,0]>0:
        	    if right_relative_center[0,1]>0:
        	        mask_r=mask_nan_r[nbr_pix1_r:,nbr_pix2_r:]
        	        mask_r=np.append(mask_r,mat1_r,axis=1)
        	        mask_r=np.transpose(np.append(np.transpose(mask_r),mat2_r,axis=1))
        	    elif right_relative_center[0,1]<0:
        	        mask_r=mask_nan_r[:np.shape(im_right)[2]-nbr_pix1_r,nbr_pix2_r:]
        	        mask_r=np.append(mask_r,mat1_r,axis=1)
        	        mask_r=np.transpose(np.append(mat2_r,np.transpose(mask_r),axis=1))
        	elif right_relative_center[0,0]<0:
        	    if right_relative_center[0,1]>0:
        	        mask_r=mask_nan_r[nbr_pix1_r:,:np.shape(im_right)[2]-nbr_pix2_r]
        	        mask_r=np.append(mat1_r,mask_r,axis=1)
        	        mask_r=np.transpose(np.append(np.transpose(mask_r),mat2_r,axis=1))
        	    elif right_relative_center[0,1]<0:
        	        mask_r=mask_nan_r[:np.shape(im_right)[1]-nbr_pix1_r,:np.shape(im_right)[2]-nbr_pix2_r]
        	        mask_r=np.append(mat1_r,mask_r,axis=1)
        	        mask_r=np.transpose(np.append(mat2_r,np.transpose(mask_r),axis=1))
        else:
        	nbr_pix1_r=np.round(abs(right_center[1]-np.shape(im_right)[1]/2.))
		nbr_pix2_r=np.round(abs(right_center[0]-np.shape(im_right)[2]/2.))
		mat1_r=np.zeros((np.shape(im_right)[1]-nbr_pix1_r,nbr_pix2_r))*np.nan
	    	mat2_r=np.zeros((np.shape(im_right)[2],nbr_pix1_r))*np.nan
		if right_relative_center[0]>0:
			if right_relative_center[1]>0:
			    mask_r=mask_nan_r[nbr_pix1_r:,nbr_pix2_r:]
			    mask_r=np.append(mask_r,mat1_r,axis=1)
			    mask_r=np.transpose(np.append(np.transpose(mask_r),mat2_r,axis=1))
			elif right_relative_center[1]<0:
			    mask_r=mask_nan_r[:np.shape(im_right)[2]-nbr_pix1_r,nbr_pix2_r:]
			    mask_r=np.append(mask_r,mat1_r,axis=1)
			    mask_r=np.transpose(np.append(mat2_r,np.transpose(mask_r),axis=1))
		elif right_relative_center[0]<0:
			if right_relative_center[1]>0:
			    mask_r=mask_nan_r[nbr_pix1_r:,:np.shape(im_right)[2]-nbr_pix2_r]
			    mask_r=np.append(mat1_r,mask_r,axis=1)
			    mask_r=np.transpose(np.append(np.transpose(mask_r),mat2_r,axis=1))
			elif right_relative_center[1]<0:
			    mask_r=mask_nan_r[:np.shape(im_right)[1]-nbr_pix1_r,:np.shape(im_right)[2]-nbr_pix2_r]
			    mask_r=np.append(mat1_r,mask_r,axis=1)
			    mask_r=np.transpose(np.append(mat2_r,np.transpose(mask_r),axis=1))
        	

        #applying new mask to centered images
        im_translat_left=im_translat_left*mask_l
        im_translat_right=im_translat_right*mask_r

        hdr['HIERARCH ESO DET CHIP1 NX']= 1024

        pyfits.writeto(file_name.replace('cl_','left_cl_'),im_translat_left,header=hdr,output_verify='warn',clobber=True)
        pyfits.writeto(file_name.replace('cl_','right_cl_'),im_translat_right,header=hdr,output_verify='warn',clobber=True)


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
            cut_and_center_cub(allfiles,science_waffle)
        else:
            cut_and_center_cub(allfiles)
        count+=1


    sys.stdout.write("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
    sys.stdout.write("\n")
    sys.stdout.write("Cut and center finished")
    sys.stdout.write("\n")
    sys.stdout.flush()
    os._exit(1)
else:
    sys.exit(0)
