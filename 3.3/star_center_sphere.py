#!/usr/bin/python
import numpy as np
import pyfits, scipy, glob, sys, os
from kapteyn import kmpfit
from scipy import ndimage,signal,fftpack
import pyfftw
import multiprocessing
from mpi4py import MPI
import graphic_nompi_lib_330 as graphic_nompi_lib
import argparse

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Detection of the star center for corono images with the waffle pattern')
parser.add_argument('--pattern', action="store", dest="pattern",  default="cl_nomed_SPHER*STAR_CENTER", help='cubes to apply the star centering')

args = parser.parse_args()
pattern=args.pattern

if rank==0:
	def f_translat(image,x,y):
	    im=np.abs(np.nan_to_num(image.copy()))
	    fft=fftpack.fft2(im)
	    k=fftpack.fftfreq(np.shape(im)[0])
	    #translation sur l'axe x et y
	    fft_bis=np.transpose(np.transpose(fft*np.exp(-2*np.pi*1j*(x*k)))*np.exp(-2*np.pi*1j*(y*k)))
	    im_translat=np.real(np.nan_to_num(fftpack.ifft2(fft_bis)))
	    return im_translat

	def low_pass(image, r, order, cut_off):
	    """
	    Low pass filter of an image by fourier transform. Use fftw
	    """
	    image=np.nan_to_num(image).astype(float)
	    pyfftw.interfaces.cache.enable()
	    pyfftw.interfaces.cache.set_keepalive_time(30)
	    test = pyfftw.n_byte_align_empty((np.shape(image)[0],np.shape(image)[0]), 16, 'complex128')
	    test=fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(test, planner_effort='FFTW_MEASURE', threads=4))
	    
	    fft=pyfftw.n_byte_align_empty((np.shape(image)[0],np.shape(image)[0]), 16, 'complex128')
	    fft=fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(np.nan_to_num(image).astype(float), planner_effort='FFTW_MEASURE', threads=4))
	    l=np.shape(image)[1]
	    x=np.arange(-l/2,l/2)
	    y=np.arange(-l/2,l/2)
	    X,Y=np.meshgrid(x,y)
	    R = np.sqrt(X**2 + Y**2)
	    R_ent=np.round(R).astype(int) #partie entiere
	    
	    B,A=signal.butter(order,cut_off/(l/2.)-r/(l/2.))
	    z=np.zeros(l)
	    z[0]=1.
	    zf=signal.lfilter(B,A,z)
	    fft_zf=fftpack.fftshift(fftpack.fft(zf))
	    fft_zf = np.append(fft_zf[l/2.:l],np.zeros(l/2))

	    F_bas=np.zeros((l,l))+0j
	    F_bas=np.where(R_ent<l/2.,np.abs(fft_zf[R_ent]),F_bas)
	    f_bas=fftpack.ifftshift(fft*F_bas)
	    im_bis=np.real(pyfftw.interfaces.scipy_fftpack.ifft2(f_bas, planner_effort='FFTW_MEASURE', threads=4))
	    return im_bis

	def moffat(size,A,alpha,beta):
	    """
	    Produce a 2D moffat profile with an image of size "size", an amplitude A, and the two parameters alpha and beta
	    """
	    x=np.arange(-size/2.,size/2.)
	    y=x
	    X,Y=np.meshgrid(x,y)
	    r=np.sqrt(X**2+Y**2)
	    mof=A*np.power(1+(r/alpha)**2,-beta)
	    return mof

	def error(par,data):
	    """
	    error function for the fit of a moffat profile on a psf in an image. The parameters of the fit are par and the data
	    are the image data[0], and the median of the entire image (not calculated here in because we use a sub image to make the fit
	    faster)
	    """
	    im=data[0]
	    med=data[1]
	    size=np.shape(im)[0]
	    A1=par[0]
	    alpha1=par[1]
	    beta1=par[2]
	    center1=[par[3],par[4]]
	    moffat1=moffat(size,A1,alpha1,beta1)
	    moffat1=f_translat(moffat1,center1[0]-size/2.,center1[1]-size/2.)
	    im_simulated=(moffat1)+med
	    e=(im_simulated-im)
	    return np.asarray(np.real(e)).reshape(-1)

	#def star_center(key_word="cl_nomed_SPHER*STAR_CENTER*"):
	def star_center(key_word):
	    """
	    Determine the star position behind the coronograph using the waffle positions and computing a levenberg-marquardt fit of a 
	    moffat profile on the waffle. We calculate the star position taking the gravity center of the waffles.
	    Input:
	    im_waffle: image with waffle pattern
	    Output:
	    creates an asci file with the position of the star in the image for the left and right image (the two filters of IRDIS)
	    """
	    #extracting the image
	    count=0
	    for allfiles in glob.iglob(key_word):
		if count==0:
		    im_waffle,hdr=pyfits.getdata(allfiles, header=True)
		else:
		    temp,hdr=pyfits.getdata(allfiles, header=True)
		    im_waffle=np.append(im_waffle,temp,axis=0)
		count+=1
	    if len(np.shape(im_waffle))>2: #taking the last frame to find the center of the star
		sys.stdout.write('More than one frame found, taking the mean')
		sys.stdout.flush()
		#im_waffle=im_waffle[np.shape(im_waffle)[0]-1,:,:] #taking the last frame to find the center of the star
		im_waffle=np.mean(im_waffle,axis=0) #taking the mean
	    
	    #rough approximation of the center by detection of the max after a low pass filter
	    im_waffle_l=im_waffle[:,:1024]
	    im_waffle_r=im_waffle[:,1024:]
	    low_pass1=low_pass(im_waffle_l,50,2,100)
	    center1=np.array(scipy.ndimage.measurements.center_of_mass(np.nan_to_num(low_pass1)))
	    center1=[round(center1[1]),round(center1[0])]

	    low_pass2=low_pass(im_waffle_r,50,2,100)
	    center2=np.array(scipy.ndimage.measurements.center_of_mass(np.nan_to_num(low_pass2)))
	    center2=[round(center2[1]),round(center2[0])]
	    
	    #cuting the image to find the center faster, im1 is the left image, im2 is the right
	    l=128
	    im1=im_waffle_l[center1[1]-l/2.:center1[1]+l/2.,center1[0]-l/2.:center1[0]+l/2.]
	    im2=im_waffle_r[center2[1]-l/2.:center2[1]+l/2.,center2[0]-l/2.:center2[0]+l/2.]
	    
	    #applying a donut shape mask on the central bright region to find the waffles
	    if hdr["HIERARCH ESO INS COMB IFLT"]=='DB_Y23':
	    	r_int=20
	    	r_ext=60
	    elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_J23':
	    	r_int=25
	    	r_ext=80
	    elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_H23':
	    	r_int=35
	    	r_ext=80
	    elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_K12':
	    	r_int=45
	    	r_ext=80
	    else:
	    	print 'Problem, no filter mode not detected!'

	    x=np.arange(-l/2.,l/2.)
	    y=np.arange(-l/2.,l/2.)
	    X,Y=np.meshgrid(x,y)
	    R1=np.sqrt(X**2+Y**2)
	    donut=np.where(R1>r_int,1,0)
	    donut=np.where(R1>r_ext,0,donut)
	    im1_donut=donut*im1
	    im2_donut=donut*im2
	    
	    #rough detection of the waffle positions for the initial guess for the moffat fit with a max detection after a low pass filter. We hide the waffle detected under a mask to detect the next one
	    #left image
	    low_pass_im1=low_pass(im1_donut,10,2,20)
	    max_index_vec1=[]
	    for i in range(4):
		max_index1=np.where(low_pass_im1==np.max(low_pass_im1))
		max_index_vec1=np.append(max_index_vec1,np.where(low_pass_im1==np.max(low_pass_im1)))
		R=np.sqrt((X+(l/2-max_index1[1]))**2+(Y+(l/2-max_index1[0]))**2)
		mask=np.where(R<10,0,1)
		low_pass_im1=low_pass_im1*mask
	    
	    #right image
	    low_pass_im2=low_pass(im2_donut,10,2,20)
	    max_index_vec2=[]
	    for i in range(4):
		max_index2=np.where(low_pass_im2==np.max(low_pass_im2))
		max_index_vec2=np.append(max_index_vec2,np.where(low_pass_im2==np.max(low_pass_im2)))
		R=np.sqrt((X+(l/2-max_index2[1]))**2+(Y+(l/2-max_index2[0]))**2)
		mask=np.where(R<10,0,1)
		low_pass_im2=low_pass_im2*mask
		
	    #moffat fit with a levenberg-markardt arlgorithm on the waffles to have the best positions
	    #left image
	    par_vec1=[]
	    err_leastsq_vec1=[]
	    par_init1=[]
	    med1=np.median(im1)
	    sys.stdout.write('\n')
	    sys.stdout.write('Fit left image')
	    sys.stdout.write('\n')
	    for i in range(4):
		sys.stdout.write('\r Waffle nbr '+str(i+1)+' of '+str(4))
		sys.stdout.flush()
		Prim_x1=max_index_vec1[2*i+1]
		Prim_y1=max_index_vec1[2*i]
		paramsinitial1=[np.max(im1_donut),5,5,Prim_x1,Prim_y1]
		fitobj1 = kmpfit.Fitter(residuals=error, data=(im1_donut,med1))
		fitobj1.fit(params0=paramsinitial1)
		par_vec1=np.append(par_vec1,fitobj1.params)
		par_init1=np.append(par_init1,paramsinitial1)
		err_leastsq_vec1=np.append(err_leastsq_vec1,fitobj1.stderr)
	    
	    
	    #right image
	    par_vec2=[]
	    err_leastsq_vec2=[]
	    par_init2=[]
	    med2=np.median(im2)
	    sys.stdout.write('\n')
	    sys.stdout.write('Fit right image')
	    sys.stdout.write('\n')
	    for i in range(4):
		sys.stdout.write('\r Waffle nbr '+str(i+1)+' of '+str(4))
		sys.stdout.flush()
		Prim_x2=max_index_vec2[2*i+1]
		Prim_y2=max_index_vec2[2*i]
		paramsinitial2=[np.max(im2_donut),5,5,Prim_x2,Prim_y2]
		fitobj2 = kmpfit.Fitter(residuals=error, data=(im2_donut,med2))
		fitobj2.fit(params0=paramsinitial2)
		par_vec2=np.append(par_vec2,fitobj2.params)
		par_init2=np.append(par_init2,paramsinitial2)
		err_leastsq_vec2=np.append(err_leastsq_vec2,fitobj2.stderr)
	    sys.stdout.write('\n')
	    #determination of the star position with a "gravity center" determination of the waffles
	    center_star1=[(par_vec1[3]+par_vec1[8]+par_vec1[13]+par_vec1[18])/4.,(par_vec1[4]+par_vec1[9]+par_vec1[14]+par_vec1[19])/4.]
	    center_star2=[(par_vec2[3]+par_vec2[8]+par_vec2[13]+par_vec2[18])/4.,(par_vec2[4]+par_vec2[9]+par_vec2[14]+par_vec2[19])/4.]
	    star_center_left=np.array(center1)-np.array([l/2.,l/2.])+np.array(center_star1)
	    star_center_right=np.array(center2)-np.array([l/2.,l/2.])+np.array(center_star2)
	    
	    #creating and filling the asci file with the star position on left and right image
	    f=open('star_center.txt','w')
	    f.write('image'+'\t'+'x_axis'+'\t'+'y_axis'+'\n')
	    f.write('----------------------------'+'\n')
	    f.write('left_im'+'\t'+str(round(star_center_left[0],3))+'\t'+str(round(star_center_left[1],3))+'\n')
	    f.write('right_im'+'\t'+str(round(star_center_right[0],3))+'\t'+str(round(star_center_right[1],3))+'\n')
	    f.close()

	t0=MPI.Wtime()
	sys.stdout.write("beginning of star centering")
	sys.stdout.write("\n")
	sys.stdout.flush()
	star_center(pattern+'*')

	sys.stdout.write("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
	sys.stdout.write("\n")
	sys.stdout.write("Star center detected and file <<star_center.txt>> created -> end of star center")
	sys.stdout.flush()
	##MPI_Finalize()
	##sys.exit(1)
	os._exit(1)
else:
	sys.exit(1)
