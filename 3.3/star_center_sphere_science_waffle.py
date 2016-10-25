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
parser.add_argument('-science_waffle', dest='science_waffle', action='store_const', const=True, default=False, help='Switch to science frame with waffle (usually for high precision astrometry)')
parser.add_argument('-ifs', dest='ifs', action='store_const', const=True, default=False, help='Switch for IFS data')


args = parser.parse_args()
pattern=args.pattern
science_waffle=args.science_waffle

if rank==0:
        
    def twoD_Gaussian((x,y), amplitude, xo, yo, sigma_x, sigma_y,theta):
        ''' Returns a 2D gaussian function
        (x,y): the 2D coordinate arrays
        amplitude, xo,yo, sigma_x,sigma_y,theta = Gaussian parameters
        '''
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        
        return amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    def low_pass(image, r, order, cut_off,threads=4):
        """
        Low pass filter of an image by fourier transform. Use fftw
        """
        # Remove NaNs from image
        image=np.nan_to_num(image).astype(float)
        
        # Shift to Fourier plane
        fft=fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(image,threads=threads))
        # Set up the coordinate and distance arrays
        l=np.shape(image)[1]
        x=np.arange(-l/2,l/2)
        y=np.arange(-l/2,l/2)
        X,Y=np.meshgrid(x,y)
        R = np.sqrt(X**2 + Y**2)
        R_ent=np.round(R).astype(int) #partie entiere
        
        # Use a Butterworth filter
        B,A=signal.butter(order,cut_off/(l/2.)-r/(l/2.))
        z=np.zeros(l)
        z[0]=1.
        zf=signal.lfilter(B,A,z)
        fft_zf=fftpack.fftshift(fftpack.fft(zf))
        fft_zf = np.append(fft_zf[int(l/2.):l],np.zeros(l/2))
        
        F_bas=np.zeros((l,l))+0j
        F_bas=np.where(R_ent<l/2.,np.abs(fft_zf[R_ent]),F_bas)
        f_bas=fftpack.ifftshift(fft*F_bas)
        im_bis=np.real(pyfftw.interfaces.scipy_fftpack.ifft2(f_bas,threads=threads))
        
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
            
            moffat1=graphic_nompi_lib.fft_shift(moffat1,center1[1]-size/2.,center1[0]-size/2.)

            im_simulated=(moffat1)+med
            e=(im_simulated-im)
            return np.asarray(np.real(e)).reshape(-1)

    def star_center(key_word,science_waffle=False,ifs=False):
            """
            Determine the star position behind the coronograph using the waffle positions and computing a levenberg-marquardt fit of a 
            moffat profile on the waffle. We calculate the star position taking the gravity center of the waffles.
            Input:
            im_waffle: image with waffle pattern
            Output:
            creates an asci file with the position of the star in the image for the left and right image (the two filters of IRDIS)
            """
            #extracting the image
            if science_waffle:
                cube_waffle,hdr=pyfits.getdata(key_word, header=True)
                size_cube=np.shape(cube_waffle)[0]
            else:
                count=0
                for allfiles in glob.iglob(key_word):
                    if count==0:
                        cube_waffle,hdr=pyfits.getdata(allfiles, header=True)
                    else:
                        temp,hdr=pyfits.getdata(allfiles, header=True)
                        cube_waffle=np.append(cube_waffle,temp,axis=0)
                    count+=1
                if cube_waffle.ndim >2: #if it is a cube we take the median over frames
                    sys.stdout.write('More than one frame found, taking the mean')
                    sys.stdout.flush()
                    cube_waffle=np.median(cube_waffle,axis=0) #taking the median
                size_cube=1
            
            # Work out how many wavelength channels there are
            if ifs:
                if cube_waffle.ndim==2:
                    n_channels=1
                else:
                    n_channels=cube_waffle.shape[0]
            else:
                n_channels=2
                # Split the IRDIS data into its two channels
                if science_waffle:
                    cube_waffle=np.array([cube_waffle[:,:,:1024],cube_waffle[:,:,1024:]])
                else:
                    cube_waffle=np.array([cube_waffle[:,:1024],cube_waffle[:,1024:]])
                
            # Make sure cube_waffle is 4D so we can loop over frames and wavelengths
            if cube_waffle.ndim == 2:
                cube_waffle=cube_waffle[np.newaxis,np.newaxis,:,:]
            elif cube_waffle.ndim == 3:
                cube_waffle=cube_waffle[np.newaxis,:,:]
                
            
            # Loop over frames
            for frame_ix in range(size_cube):
                    
                # Loop over wavelength channels
                for channel_ix in range(n_channels):
                    
                    im_waffle=cube_waffle[frame_ix,channel_ix]
                    
                    
                    #rough approximation of the center by detection of the max after a low pass filter
                    low_pass_im=low_pass(im_waffle,50,2,100)
                    
                    center=np.array(scipy.ndimage.measurements.center_of_mass(np.nan_to_num(low_pass_im)))
                    center=[int(round(center[1])),int(round(center[0]))]
                
                    #cut the image to find the center faster
                    if ifs:
                        l=200 # H band will be outside of 128 pix, so take a larger area
                    else:
                        l=128
                    im=im_waffle[center[1]-l/2:center[1]+l/2,center[0]-l/2:center[0]+l/2]
            
                    #apply a donut shape mask on the central bright region to find the waffles
                    if ifs:
                        # What is the wavelength? We should have stored this during the re-cubing
                        diff_limit=(hdr['HIERARCH GC WAVELENGTH']*1e-6/8.2)*180/np.pi*3600*1000
                        diff_limit_pix = diff_limit / 7.46 # 7.46 mas/pix is the pixel scale.
                        
                        # they're at about 14.5 lambda/D, so take 2 lambda/D each side to be safe
                        r_int=12.5*diff_limit_pix
                        r_ext=16.5*diff_limit_pix
                        
                    elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_Y23':
                        r_int=20
                        r_ext=60
                    elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_J23':
                        r_int=25
                        r_ext=80
                    elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_H23':
                        r_int=35
                        r_ext=60
                    elif hdr["HIERARCH ESO INS COMB IFLT"]=='DB_K12':
                        r_int=45
                        r_ext=80
                    else:
                        print 'Problem, no filter mode not detected!'
        
                    x=np.arange(-l/2.,l/2.)
                    y=np.arange(-l/2.,l/2.)
                    X,Y=np.meshgrid(x,y)
                    R=np.sqrt(X**2+Y**2)
                    donut=np.where(R>r_int,1,0)
                    donut=np.where(R>r_ext,0,donut)
                    im_donut=donut*im
            
                    # rough detection of the waffle positions for the initial guess 
                    # for the moffat fit with a max detection after a low pass filter. 
                    # We hide the waffle detected under a mask to detect the next one
        
                    model=twoD_Gaussian((X,Y), 1, 0, 0, 3, 3,0)
                    mask=np.where(R>4,-1,1)
                    mask=np.where(R>6,0,mask)
        
                    model=model*mask
                    low_pass_im=signal.correlate2d(im_donut,model,"same")
                    
                    max_index_vec1=[]
                    # Loop over the waffle spots
                    # print X.shape,Y.shape,im_donut.shape
                    for i in range(4):
                        max_index=np.where(low_pass_im==np.nanmax(low_pass_im))
                        max_index_vec1=np.append(max_index_vec1,np.where(low_pass_im==np.nanmax(low_pass_im)))
                        R=np.sqrt((X+(l/2-max_index[1]))**2+(Y+(l/2-max_index[0]))**2)
                        mask=np.where(R<30,0,1)
                        low_pass_im=low_pass_im*mask
            
                    #moffat fit with a levenberg-markardt arlgorithm on the waffles to have the best positions
                    par_vec=[]
                    err_leastsq_vec=[]
                    par_init=[]
                    med=np.median(im)
                    sys.stdout.write('\n')
                    sys.stdout.write('Fitting channel '+str(channel_ix))
                    sys.stdout.write('\n')
                    
                    # Loop over waffle spots
                    for i in range(4):
                        sys.stdout.write('\r Waffle nbr '+str(i+1)+' of '+str(4))
                        sys.stdout.flush()
                        Prim_x=max_index_vec1[2*i+1]
                        Prim_y=max_index_vec1[2*i]
                        #cutting the image just around the waffle to make the fit faster
                        im_temp=im_donut[int(Prim_y)-10:int(Prim_y)+10,int(Prim_x)-10:int(Prim_x)+10]
                        Prim_x_temp=Prim_x-int(Prim_x)+10
                        Prim_y_temp=Prim_y-int(Prim_y)+10
                        
                        paramsinitial=[np.max(im_donut),5,5,Prim_x_temp,Prim_y_temp]
                        fitobj = kmpfit.Fitter(residuals=error, data=(im_temp,med))
                        fitobj.fit(params0=paramsinitial)
                        par_vec_temp=np.copy(fitobj.params)
                        par_vec_temp[3]=par_vec_temp[3]+int(Prim_x)-10
                        par_vec_temp[4]=par_vec_temp[4]+int(Prim_y)-10
                        par_vec=np.append(par_vec,par_vec_temp)
                        par_init=np.append(par_init,paramsinitial)
                        err_leastsq_vec=np.append(err_leastsq_vec,fitobj.stderr)
            
            
                    sys.stdout.write('\n')
                    #determination of the star position with a "gravity center" determination of the waffles
                    center_star=[(par_vec[3]+par_vec[8]+par_vec[13]+par_vec[18])/4.,(par_vec[4]+par_vec[9]+par_vec[14]+par_vec[19])/4.]
                    star_center=np.array(center)-np.array([l/2.,l/2.])+np.array(center_star)
                    
                
                    if ifs:
                        channel_name='wavelength_'+str(channel_ix)
                    else:
                        if channel_ix==0:
                            channel_name='left_im'
                        else:
                            channel_name='right_im'
                    #creating and filling the asci file with the star position
                    with open('star_center.txt','a') as f:
                        f.write(channel_name+'\t'+str(round(star_center[0],3))+'\t'+str(round(star_center[1],3))+'\n')


    t0=MPI.Wtime()
    sys.stdout.write("beginning of star centering")
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    f=open('star_center.txt','w')
    f.write('image'+'\t'+'x_axis'+'\t'+'y_axis'+'\n')
    f.write('----------------------------'+'\n')
    f.close()
    
    if science_waffle:
        sys.stdout.write("Science waffle frames: finding the center for each frame in the cubes")
        sys.stdout.write("\n")
        for allfiles in glob.iglob(pattern+"*"):
            sys.stdout.write(allfiles)
            sys.stdout.write("\n")
            f=open('star_center.txt','a')
            f.write(allfiles+' \n')
            f.close()
            star_center(allfiles,science_waffle,ifs=args.ifs)
    else:
        star_center(pattern+'*',ifs=args.ifs)

    sys.stdout.write("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
    sys.stdout.write("\n")
    sys.stdout.write("Star center detected and file <<star_center.txt>> created -> end of star center")
    sys.stdout.write("\n")
    sys.stdout.flush()
    ##MPI_Finalize()
    ##sys.exit(1)
    # os._exit(1)
    sys.exit(0)
else:
    # sys.exit(1)
    sys.exit(0)