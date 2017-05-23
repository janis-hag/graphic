#!/usr/bin/python
import numpy as np
import pyfits, scipy, glob, sys, os
#from kapteyn import kmpfit
from scipy.optimize import minimize
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
parser.add_argument('--lowpass_r',dest='lowpass_r',action='store',type=int,default=50,help='Radius of low pass filter to apply (in pixels) prior to the rough guess of the centre. Default 50.')
parser.add_argument('--manual_rough_centre',dest='manual_rough_centre',action='store',type=int,nargs='+',default=-1,help='Rough position of star behind coronagraph. Use this to overwrite the rough centring before the waffle positions are measured.')
parser.add_argument('--ignore_frame', dest='ignore_frame', type=float, nargs='+',required=False,help='ds9 number of frames for bad frames to be ignored')


args = parser.parse_args()
pattern=args.pattern
science_waffle=args.science_waffle
lowpass_r = args.lowpass_r
manual_rough_centre = args.manual_rough_centre
ignore_frame=args.ignore_frame

if rank==0:
    if type(ignore_frame) != type(None):
        ignore_frame = np.array(ignore_frame)
        bad_frame = True
    else:
        bad_frame = False
        print "no bad frame to ignore"
    
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
            
    def moffat3(size,S0,A,x0,y0,alpha1,alpha2,beta,theta):
        x=np.arange(-size/2.,size/2.)
        y=x
        X,Y=np.meshgrid(x,y)
        theta_rad=np.pi*theta/180.
        alpha1=float(alpha1)
        alpha2=float(alpha2)
        #alpha=fwhm/(2*np.sqrt(2.**(1./beta)-1.))
        a=(np.cos(theta_rad)/alpha1)**2+(np.sin(theta_rad)/alpha2)**2
        b=(np.sin(theta_rad)/alpha1)**2+(np.cos(theta_rad)/alpha2)**2
        c=2*np.sin(theta_rad)*np.cos(theta_rad)*(1./alpha1**2-1/alpha2**2)
        
        mof=S0+A*(np.power(1+(a*(X-x0)**2+b*(Y-y0)**2+c*(X-x0)*(Y-y0)),-float(beta)))

        return mof
    
    def error3(par,im):
        """
        error function for the fit of a moffat profile on a psf in an image. The parameters of the fit are par and the data
        are the image data[0], and the median of the entire image (not calculated here in because we use a sub image to make the fit
        faster)
        """
        
        size=np.shape(im)[0]
        S0=float(par[0])
        A1=float(par[1])
        x01=float(par[2])
        y01=float(par[3])
        fwhm1=float(par[4])
        fwhm2=float(par[5])
        beta1=float(par[6])
        theta1=float(par[7])
        
        moffat1=moffat3(size,S0,A1,x01,y01,fwhm1,fwhm2,beta1,theta1)
        
        im_simulated=moffat1
        e=np.nansum((im_simulated-im)**2)
        
        return e

    def star_center(key_word,science_waffle=False,ifs=False,lowpass_r=50,manual_rough_centre=-1):
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
            filenames=glob.glob(key_word)
            filenames_sorted=np.sort(filenames)
            print filenames_sorted
            for allfiles in filenames_sorted:
                if count==0:
                    cube_waffle,hdr=pyfits.getdata(allfiles, header=True)
                else:
                    temp,hdr=pyfits.getdata(allfiles, header=True)
                    cube_waffle=np.append(cube_waffle,temp,axis=0)
                count+=1
            if bad_frame:
                sys.stdout.write('bad frame to be deleted '+str(ignore_frame)+'\n')
                sys.stdout.flush()
                cube_waffle=np.delete(cube_waffle,ignore_frame-1,axis=0)
                star_center_bad_frame_deleted_filename='STAR_CENTER_cube_bad_frame_del.fits'
                pyfits.writeto(star_center_bad_frame_deleted_filename,cube_waffle,header=hdr,clobber=True)
            if cube_waffle.ndim >2: #if it is a cube we take the median over frames
                sys.stdout.write('More than one frame found, taking the median \n')
                sys.stdout.flush()
                cube_waffle=np.nanmedian(cube_waffle,axis=0) #taking the median
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
                low_pass_im=low_pass(im_waffle,lowpass_r,2,100)
                
                if manual_rough_centre == -1:
                    # center=np.array(scipy.ndimage.measurements.center_of_mass(np.nan_to_num(low_pass_im)))
                    center =np.where(low_pass_im == np.max(low_pass_im))
                    center=[int(round(center[1])),int(round(center[0]))]
                else:
                    print('')
                    print('  Using manual rough centre position: '+str(manual_rough_centre[0])+' '+str(manual_rough_centre[1]))
                    center = manual_rough_centre
            
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
                    max_index=np.array(np.where(low_pass_im==np.nanmax(low_pass_im)))
                    max_index=max_index[...,0] # take the first one (in case there are multiple peaks with the same value)
                    max_index += [1,1] # we need to add 1 to each direction because the correlation above shifts the image by 1 pixel
                    max_index_vec1=np.append(max_index_vec1,max_index)
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

                # Turn the area outside of the donut into NaNs to avoid problems with the background
                im_donut[im_donut==0] = np.NaN
                
                # Loop over waffle spots
                cutout_sz = 10 # pix, size around waffle to fit
                for i in range(4):
                    sys.stdout.write('\r Waffle nbr '+str(i+1)+' of '+str(4))
                    sys.stdout.flush()
                    Prim_x=max_index_vec1[2*i+1]
                    Prim_y=max_index_vec1[2*i]
                    #cutting the image just around the waffle to make the fit faster
                    im_temp=im_donut[int(Prim_y)-cutout_sz:int(Prim_y)+cutout_sz,int(Prim_x)-cutout_sz:int(Prim_x)+cutout_sz]
                    Prim_x_temp=np.shape(im_temp)[0]/2.
                    Prim_y_temp=np.shape(im_temp)[0]/2.
                    
                    #test in which quadrant to initiate the angle
                    if ((Prim_x<center[0]) & (Prim_y<center[1])) or ((Prim_x>center[0]) & (Prim_y>center[1])):
                            theta_init=-40
                    else:
                            theta_init=40
                    
                    paramsinitial=[np.nanmedian(im_temp),np.nanmax(im_temp),0.1,0.1,7,9,7,theta_init]
                    #fitobj = kmpfit.Fitter(residuals=error, data=(im_temp,med))
                    #fitobj.fit(params0=paramsinitial)
                    res = minimize(error3,paramsinitial, args=(im_temp), method='nelder-mead')
                    par_vec_temp=np.copy(res.x)[1:4]
                    par_vec_temp[1]=par_vec_temp[1]+cutout_sz#+int(Prim_x)#-10
                    par_vec_temp[2]=par_vec_temp[2]+cutout_sz#+int(Prim_y)#-10
                    par_vec=np.append(par_vec,par_vec_temp)
                    par_init=np.append(par_init,paramsinitial)
                    #err_leastsq_vec=np.append(err_leastsq_vec,fitobj.stderr)
                    
                    mof_init=moffat3(20,paramsinitial[0],paramsinitial[1],paramsinitial[2],paramsinitial[3],paramsinitial[4],paramsinitial[5],paramsinitial[6],paramsinitial[7])
                    mof=moffat3(20,res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],res.x[6],res.x[7])
        
                sys.stdout.write('\n')
                center_spot1=np.array([center[0]+max_index_vec1[1]-np.shape(low_pass_im)[1]/2.+par_vec[1]-np.shape(im_temp)[0]/2.,center[1]+max_index_vec1[0]-np.shape(low_pass_im)[0]/2.+par_vec[2]-np.shape(im_temp)[0]/2.])
                center_spot2=np.array([center[0]+max_index_vec1[3]-np.shape(low_pass_im)[1]/2.+par_vec[4]-np.shape(im_temp)[0]/2.,center[1]+max_index_vec1[2]-np.shape(low_pass_im)[0]/2.+par_vec[5]-np.shape(im_temp)[0]/2.])
                center_spot3=np.array([center[0]+max_index_vec1[5]-np.shape(low_pass_im)[1]/2.+par_vec[7]-np.shape(im_temp)[0]/2.,center[1]+max_index_vec1[4]-np.shape(low_pass_im)[0]/2.+par_vec[8]-np.shape(im_temp)[0]/2.])
                center_spot4=np.array([center[0]+max_index_vec1[7]-np.shape(low_pass_im)[1]/2.+par_vec[10]-np.shape(im_temp)[0]/2.,center[1]+max_index_vec1[6]-np.shape(low_pass_im)[0]/2.+par_vec[11]-np.shape(im_temp)[0]/2.])
                #determination of the star position with a "gravity center" determination of the waffles
                center_star=[np.mean(np.array([center_spot1[0],center_spot2[0],center_spot3[0],center_spot4[0]])),np.mean(np.array([center_spot1[1],center_spot2[1],center_spot3[1],center_spot4[1]]))]
                star_center=np.array(center_star)
                
                print "spot 1 center (x,y): (",center_spot1[0],",",center_spot1[1],")"
                print "spot 2 center (x,y): (",center_spot2[0],",",center_spot2[1],")"
                print "spot 3 center (x,y): (",center_spot3[0],",",center_spot3[1],")"
                print "spot 4 center (x,y): (",center_spot4[0],",",center_spot4[1],")"
            
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
            star_center(allfiles,science_waffle,ifs=args.ifs,lowpass_r=lowpass_r,manual_rough_centre=manual_rough_centre)
    else:
        star_center(pattern+'*',ifs=args.ifs,lowpass_r=lowpass_r,manual_rough_centre=manual_rough_centre)

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
