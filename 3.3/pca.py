# -*- coding: utf-8 -*-
"""
Some PCA code for ADI
"""

import scipy,time,pickle
import numpy as np
import astropy.io.fits as pyfits
from scipy import ndimage
from multiprocessing import Pool
try:
    import pyfftw.interfaces.scipy_fftpack as fftpack
except:
    print('Failed to load pyfftw')
    from scipy import fftpack


###############

###############

def principal_components(pix_array,n_modes=None):
    ''' Calculates the principal components for a set of flattened pixel arrays
     (i.e. pix_array must be 2D, and is n_frames x n_pixels)
     n_modes is the number of modes to return. Default is n_frames'''
    # Get the principal components
    cov=np.dot(pix_array,pix_array.T)
    evals,evects=scipy.linalg.eigh(cov)
            
    # Get the principal components
    pc= np.dot(evects.T,pix_array)
    pcomps=pc[::-1]
    S = np.sqrt(evals)[::-1]
    for comp_ix in range(pcomps.shape[1]):
        pcomps[:,comp_ix]/=S
    pcomps=pcomps[:n_modes]
    return pcomps

###############

###############
   
def subtract_principal_components(pcomps,image_region):
    ''' Projects an array of pixel values onto the principal components and 
    subtracts the result from the original values.'''
    transformed = np.dot(pcomps, image_region.T)
    reconstructed = np.dot(transformed.T, pcomps)
    residuals = image_region - reconstructed
    return residuals

###############

###############

def pca_multi(all_vars):
    ''' A wrapper for the pca step, needed for multiprocessing. Image region is
    the part of the image that will be pca subtracted, cube_region is the part
    of the cube that the components are calculated from'''
    # Get the principal components
    pcomps=principal_components(all_vars['cube_region'],n_modes=all_vars['n_modes'])      
    # Subtract them from the data
    resids=subtract_principal_components(pcomps,all_vars['image_region'])
    return [resids,pcomps]

###############

###############

def pca_multi_annular(all_vars):
    ''' A wrapper for the pca step, needed for multiprocessing. Image region is
    the part of the image that will be pca subtracted, cube_region is the part
    of the cube that the components are calculated from'''
    ix=all_vars['ix']
    image_out=np.zeros((all_vars['npix']))
    pcomps_out=np.zeros((all_vars['n_modes'],image_out.size))
    for region_ix,region in enumerate(all_vars['regions']):
        
        # Find which frames are ok to use
        parang=all_vars['parangs'][ix]
        rot_dist=np.abs(2*all_vars['region_radii'][region_ix]*
            np.sin((all_vars['parangs']-parang)*np.pi/180/2.))
        good_frames= rot_dist > (all_vars['n_fwhm']*all_vars['fwhm'])  # which frames have rotated enough?
        
        cube_region=all_vars['region_cube'][region_ix][good_frames]
        image_region=all_vars['region_cube'][region_ix][ix]
        
        # Get the principal components
        pcomps=principal_components(cube_region,n_modes=all_vars['n_modes'])      
        # Subtract them from the data
        resids=subtract_principal_components(pcomps,image_region)
        # Save it out
        image_out[region]=resids
        pcomps_out[:len(pcomps),region]=pcomps

    if (ix % 10) ==9:
        time_left=(len(all_vars['parangs'])-ix-1)*(time.time()-all_vars['t_start'])/(ix+1)
        print('  Frame '+str(ix+1)+' done. Time remaining: '+str(np.round(time_left/60.,2))+'mins' )
    return [image_out,pcomps_out]

###############

###############

def fft_rotate(in_frame, alpha, pad=4,x1=0,x2=0,y1=0,y2=0,return_full=False):
    """
    3 FFT shear based rotation, following Larkin et al 1997

    in_frame: the numpy array which has to be rotated
    alpha: the rotation alpha in degrees
    pad: the padding factor
    x1,x2: the borders of the original image in x
    y1,y2: the borders of the original image in y


    Return the rotated array
    """

    #################################################
    # Check alpha validity and correcting if needed
    #################################################
    alpha=1.*alpha-360*np.floor(alpha/360)

    # FFT rotation only work in the -45:+45 range
    if alpha > 45 and alpha < 135:
        in_frame=np.rot90(in_frame, k=1)
        alpha_rad=-np.deg2rad(alpha-90)
    elif alpha > 135 and alpha < 225:
        in_frame=np.rot90(in_frame, k=2)
        alpha_rad=-np.deg2rad(alpha-180)
    elif alpha > 225 and alpha < 315:
        in_frame=np.rot90(in_frame, k=3)
        alpha_rad=-np.deg2rad(alpha-270)
    else:
        alpha_rad=-np.deg2rad(alpha)

    ###################################
    # Preparing the frame for rotation
    ###################################

    # Calculate the position that the input array will be in the padded array to simplify
    #  some lines of code later 
    px1=np.int(((pad-1)/2.)*in_frame.shape[0])
    px2=np.int(((pad+1)/2.)*in_frame.shape[0])
    py1=np.int(((pad-1)/2.)*in_frame.shape[1])
    py2=np.int(((pad+1)/2.)*in_frame.shape[1])

    # Make the padded array    
    pad_frame=np.ones((in_frame.shape[0]*pad,in_frame.shape[1]*pad))*np.NaN
    pad_mask=np.ones((pad_frame.shape), dtype=bool)
    pad_frame[px1:px2,py1:py2]=in_frame
    pad_mask[px1:px2,py1:py2]=np.where(np.isnan(in_frame),True,False)
    
    # Rotate the mask, to know what part is actually the image
    pad_mask=ndimage.interpolation.rotate(pad_mask, np.rad2deg(-alpha_rad),
          reshape=False, order=0, mode='constant', cval=True, prefilter=False)

    # Replace part outside the image which are NaN by 0, and go into Fourier space.
    pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)
    # This part causes problems when there are NaNs in the array... Why do we want 
    #   to halve the edges anyway?
#    pad_frame[px1-1,py1:py2]=in_frame[0,:]/2.
#    pad_frame[px2,py1:py2]=in_frame[-1,:]/2.
#    pad_frame[px1:px2,py1-1]=in_frame[:,0]/2.
#    pad_frame[px1:px2,py2] =in_frame[:,-1]/2.

    ###############################
    # Rotation in Fourier space
    ###############################
    a=np.tan(alpha_rad/2.)
    b=-np.sin(alpha_rad)

    M=-2j*np.pi*np.ones(pad_frame.shape)
    N=fftpack.fftfreq(pad_frame.shape[0])

    X=np.arange(-pad_frame.shape[0]/2.,pad_frame.shape[0]/2.)#/pad_frame.shape[0]

    pad_x=fftpack.ifft((fftpack.fft(pad_frame, axis=0,overwrite_x=True).T*
        np.exp(a*((M*N).T*X).T)).T, axis=0,overwrite_x=True)
    pad_xy=fftpack.ifft(fftpack.fft(pad_x,axis=1,overwrite_x=True)*
        np.exp(b*(M*X).T*N), axis=1,overwrite_x=True)
    pad_xyx=fftpack.ifft((fftpack.fft(pad_xy, axis=0,overwrite_x=True).T*
        np.exp(a*((M*N).T*X).T)).T,axis=0,overwrite_x=True)

    # Go back to real space
    # Put back to NaN pixels outside the image.

    pad_xyx[pad_mask]=np.NaN

    if return_full:
        return np.real(pad_xyx).copy()
    else:
        return np.real(pad_xyx[px1:px2,py1:py2]).copy()

###############

###############

def fft_rotate_multi(all_vars):
    ''' A wrapper for fft_rotate, needed for multiprocessing'''
    return fft_rotate(all_vars['in_frame'],all_vars['alpha'],all_vars['pad'],
                      return_full=all_vars['return_full'])

###############

###############

def define_annuli(npix,n_radii,arc_length,r_min):
    ''' Defines the annuli used for pca. The image is divided into these regions
    and the principal components are calculated for each region individually.
    npix is the number of pixels in the image. The image is assumed to be square!
    rmin is the minimum radius for the annuli. npix/2 is expected to be the maximum radius
    '''

    npix=np.int(npix)
    n_radii=np.int(n_radii)
    arc_length=np.int(arc_length)
    r_min=np.int(r_min)
    
    radii=np.linspace(r_min,np.sqrt(2)*npix/2,num=(n_radii+1))

    # Make maps of the distance from the origin and the angle in azimuth
    xarr=np.arange(0,npix)-npix/2
    xx,yy=np.meshgrid(xarr,xarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)
    azimuth_map=np.arctan2(xx,yy)
    
    pix_dist_map_2d=pix_dist_map.ravel()
    azimuth_map_2d=azimuth_map.ravel()
    
    regions=[]
    radii_output=[]
    # Loop over the radial annuli
    for radius_ix in range(n_radii):
        # The min and max radius for this annulus
        minrad=radii[radius_ix]
        maxrad=radii[radius_ix+1]
        meanrad=(minrad+maxrad)/2.
        
        # Loop over the azimuthal segments
        n_azimuth=np.int(np.round(2*np.pi*meanrad/arc_length))
        for azimuth_ix in range(n_azimuth):
            # The min and max azimuth for this segment
            minaz=2*np.pi/n_azimuth*azimuth_ix-np.pi
            maxaz=2*np.pi/n_azimuth*(azimuth_ix+1)-np.pi

            # Calculate which pixels satisfy both the azimuth and radius constraints
            region_2d=np.where((pix_dist_map_2d>=minrad) & (pix_dist_map_2d <maxrad) 
                & (azimuth_map_2d > minaz) & (azimuth_map_2d <= maxaz))
            
            # Ignore any regions that are partially outside the FoV and don't have many pixels
            if (minrad >npix/2) and (len(region_2d[0]) <(0.75*arc_length*(maxrad-minrad))):
                continue

            regions.append(region_2d[0])
            radii_output.append(meanrad)

    return [regions,radii_output]

###############

###############

def simple_pca(image_file,n_modes,save_name,pc_name=None):
    ''' Performs a simple PCA reduction on the input datacube.
    Performs PCA on the full frame, and uses all frames at once (i.e. not 
    "smart-pca".
    n_modes is the number of modes to remove from the image
    pc_name is the name of the file that the principal components will be saved (as a pickle file)
    '''
    
    # Load the datacube and make it 2d
    cube=pyfits.getdata(image_file)
    initial_shape=cube.shape
    cube=cube.reshape([cube.shape[0],cube.shape[1]*cube.shape[2]])
    
    # median subtract
    cube-=np.median(cube)
    
    # Get the principal components
    pcomps=principal_components(cube,n_modes=n_modes)
    
    # Subtract them from the data
    cube_out=subtract_principal_components(pcomps,cube)
    
    # Make the cube 3d again
    cube_out=cube_out.reshape(initial_shape)
    if save_name:
        pyfits.writeto(save_name,cube_out,clobber=True)
        print '  PCA subtracted cube saved as:',save_name
    
    if pc_name:
        data={'principal_components':pcomps,'regions':None,'npix':initial_shape[1],
              'protection_angle':None}
        with open(pc_name,'w') as myf:
            pickle.dump(data,myf)
    
    return cube_out
        

###############

###############

def annular_pca(image_file,n_modes,save_name,n_annuli=5,arc_length=50,r_min=5,
                pc_name=None,threads=2):
    ''' Performs a simple PCA reduction on the input datacube.
    Performs PCA in annular regions, using all frames at once (i.e. not 
    "smart-pca".
    n_modes is the number of modes to remove from the image
    n_annuli is the number of radial annuli to use
    arc_length is the approximate length (in pixels) used to split the annuli azimuthally
    pc_name is the name of the file that the principal components will be saved (as a pickle file)
    '''
    print("Running annular pca")
    # Load the datacube and make it 2d
    print("  Loading cube")
    cube=pyfits.getdata(image_file)
    initial_shape=cube.shape
    cube=cube.reshape([cube.shape[0],cube.shape[1]*cube.shape[2]])
    cube_out=0*cube
    
    # Calculate the annular regions
    print('  Defining annuli')
    regions,region_radii=define_annuli(initial_shape[1],n_annuli,arc_length,r_min=r_min)
            
    print('  Setting up arrays for multiprocessing')
    # Set up the pool and arrays for multiprocessing
    pool=Pool(processes=threads)
    all_vars=[]
        
    # Loop over the pca regions
    for region_ix,region in enumerate(regions):
        cube_region=1*cube[:,region]
        these_vars={'cube_region':cube_region,'n_modes':n_modes,'image_region':cube_region}
        all_vars.append(these_vars)
    
    print('  Starting PCA')
    # Now let multiprocessing do it all
    pca_output=pool.map(pca_multi,all_vars)
    
    pool.close()
    
    # Reorder it all
    cube_out=0*cube
    pcomps=np.zeros((n_modes,cube.shape[1]))
    for region_ix,region in enumerate(regions):
        cube_out[:,region]=pca_output[region_ix][0]
        pcomps[:,region]=pca_output[region_ix][1]
        
      
    # Make the cube 3d again
    cube_out=cube_out.reshape(initial_shape)
    pcomps=pcomps.reshape((n_modes,cube_out.shape[1],cube_out.shape[2]))
    if save_name:
        pyfits.writeto(save_name,cube_out,clobber=True)
        print('  PCA subtracted cube saved as: '+save_name)
    
    if pc_name:
        data={'principal_components':pcomps,'regions':regions,'npix':initial_shape[1],
              'protection_angle':None}
        with open(pc_name,'w') as myf:
            pickle.dump(data,myf)

    return pcomps
    
###############

###############

def smart_pca(image_file,n_modes,save_name,parang_file,protection_angle=20,
              pc_name=None):
    ''' Performs a smart PCA reduction on the input datacube.
    Performs PCA on the full frame, using only frames that have a parallactic
    angle difference that is above some threshold, to minimize self-subtraction.
    n_modes is the number of modes to remove from the image
    protection_angle is the minimum parallactic angle change used to include a frame in pca. In degrees
    pc_name is the name of the file that the principal components will be saved (as a pickle file)
    '''
    
    # Load the datacube and make it 2d
    cube=pyfits.getdata(image_file)
    initial_shape=cube.shape
    cube=cube.reshape([cube.shape[0],cube.shape[1]*cube.shape[2]])
    try:
        parangs=np.loadtxt(parang_file)
    except:
        parangs=pyfits.getdata(parang_file)
    
    cube_out=0*cube # this should preserve the NaNs
    
    # Set the NaNs to zero and restore them later
    nan_mask=np.isnan(cube)
    cube[nan_mask]=0.
    
    t_start=time.time()
    # Loop through frames in the cube
    print 'smart_pca: starting loop over frames in image'
    for ix,frame in enumerate(cube):
        
        parang=parangs[ix]
        # Find which frames are ok to use
        good_frames=cube[np.abs(parang-parangs) > protection_angle]
        
        # Get the principal components
        pcomps=principal_components(good_frames,n_modes=n_modes)
        
        # Subtract them from the data
        cube_out[ix]=subtract_principal_components(pcomps,cube[ix])
    
        if (ix % 10) ==9:
            time_left=(cube.shape[0]-ix-1)*(time.time()-t_start)/(ix+1)
            print '  Done',ix+1,'of',cube.shape[0],np.round(time_left/60.,2),'mins remaining'
    
    cube_out[nan_mask]=np.nan
    # Make the cube 3d again
    cube_out=cube_out.reshape(initial_shape)
    if save_name:
        pyfits.writeto(save_name,cube_out,clobber=True)
        print '  PCA subtracted cube saved as:',save_name

    if pc_name:
        data={'principal_components':pcomps,'regions':None,'npix':initial_shape[1],
              'protection_angle':protection_angle}
        with open(pc_name,'w') as myf:
            pickle.dump(data,myf)
    return cube_out

###############

###############

def smart_annular_pca(image_file,n_modes,save_name,parang_file,n_fwhm=2,fwhm=4.5,
              pc_name=None,n_annuli=5,arc_length=50,r_min=5,threads=3):
    ''' Performs a smart PCA reduction on the input datacube, with PCA performed
    locally on annuli, using only frames that have a parallactic angle
    difference that is above some threshold, to minimize self-subtraction.
    n_modes is the number of modes to remove from the image
    
    pc_name is the name of the file that the principal components will be saved (as a pickle file)
    '''
    print('Running smart annular PCA')
    
    # Load the datacube and make it 2d
    cube=pyfits.getdata(image_file)
    initial_shape=cube.shape
    cube=cube.reshape([cube.shape[0],cube.shape[1]*cube.shape[2]])
    try:
        parangs=np.loadtxt(parang_file)
    except:
        parangs=pyfits.getdata(parang_file)
    
    cube_out=0*cube # this should preserve the NaNs
    
    # Set the NaNs to zero and restore them later
    nan_mask=np.isnan(cube)
    cube[nan_mask]=0.
    
    # Calculate the annular regions
    print('  Defining annuli')
    regions,region_radii=define_annuli(initial_shape[1],n_annuli,arc_length,r_min=r_min)
    
    # Check that the rotation is enough for the closest annuli
    min_radius=np.min(region_radii)
    n_good_frames=[]
    for parang in parangs:
        rot_dist=np.abs(2*min_radius*np.sin((parangs-parang)*np.pi/180/2.))
        n_good_frames.append(np.sum(rot_dist > (n_fwhm*fwhm)))
    print('  Minimum of: '+str(np.min(n_good_frames))+' frames per annuli')
    if np.min(n_good_frames) < n_modes:
        print(' WARNING: some annuli have less available frames than the requested number of modes to subtract')
        if np.min(n_good_frames) ==0:
            # What rotation do we need to get at least one frame
            min_rot=2*np.arcsin(n_fwhm*fwhm/(2*min_radius))*180./np.pi
            print('Minimum rotation needed for inner annuli: '+str(np.round(min_rot,decimals=1)))
            print('Found: '+str(np.round(np.max(parangs)-np.min(parangs),decimals=1)))
            raise ValueError('Need at least 1 frame for PCA! Try decreasing n_fwhm or increasing r_min')
            
    # Set up the pool and arrays for the loop
    pool=Pool(processes=threads)
    cube_out=0*cube
    t_start=time.time()
    pcomps=np.zeros((n_modes,cube.shape[0],cube.shape[1]))

    print('Setting up arrays for the loop')
    # Set up arrays for the loop
    region_cube=[]
    for region in regions:
        region_cube.append(cube[:,region]) # restructuring this now might save time later
    all_vars=[]
    for ix,frame in enumerate(cube): 
        these_vars={'ix':ix, 'region_cube':region_cube, 'n_modes':n_modes,
            'regions':regions,'region_radii':region_radii,'parangs':parangs,
            'n_fwhm':n_fwhm,'fwhm':fwhm,'npix':cube.shape[1],'t_start':time.time()}
        all_vars.append(these_vars)
     
    
    print('  Starting loop over frames in image')
    # Now let multiprocessing do it all
    pca_output=pool.map(pca_multi_annular,all_vars,chunksize=1)
    # Close the threads
    pool.close()
    all_vars=[] # free up some RAM
    
    # Uncomment these lines to do it without multiprocessing
#    for ix,these_vars in enumerate(all_vars):
#        pca_output=pca_multi_annular(these_vars)
#        cube_out[ix]=pca_output[0]
#        pcomps[:,ix,:]=pca_output[1]

    print('  Done! Took '+str((time.time()-t_start)/60)+' mins')
    print('  Reordering the output')
    # Reorder it all and store in the output arrays
    for ix in range(len(pca_output)):
        output=pca_output[ix]
        cube_out[ix]=output[0]
        pcomps[:,ix,:]=output[1]
        pca_output[ix]=[]
        
    print('  Reordered output') 
    
    cube_out[nan_mask]=np.nan
    # Make the cube 3d again
    cube_out=cube_out.reshape(initial_shape)
    if save_name:
        pyfits.writeto(save_name,cube_out,clobber=True)
        print('  PCA subtracted cube saved as: '+save_name)

    if pc_name:
        data={'principal_components':pcomps,'regions':None,'npix':initial_shape[1],
              }
        with open(pc_name,'w') as myf:
            pickle.dump(data,myf)
    return cube_out

###############

###############

def derotate_and_combine(image_file,parang_file,save_name='derot.fits',
                 median_combine=False):
    '''Derotates and mean-combines a cube. Uses FFT derotation '''
    
    #Load the image
    if isinstance(image_file,str):
        cube=pyfits.getdata(image_file)
    else: # assume it is already a cube
        cube=image_file
    # Load the parangs
    try:
        parangs=np.loadtxt(parang_file)
    except:
        parangs=pyfits.getdata(parang_file)
    
    out_cube=np.zeros(cube.shape*np.array([1,2,2]))+np.nan # assuming pad=2 in fft_rotate
    t_start=time.time()

    print 'Starting image derotation'
    # Loop through the images
    for ix in range(cube.shape[0]):
        in_frame=cube[ix]
        derot_frame=fft_rotate(in_frame,-parangs[ix], pad=2,return_full=True)
        out_cube[ix]=derot_frame
        
        if (ix % 20) ==19:
            time_left=(cube.shape[0]-ix-1)*(time.time()-t_start)/(ix+1)
            print '  Done',ix+1,'of',cube.shape[0],np.round(time_left/60.,2),'mins remaining'
        
    # now sum into a final image
    if median_combine:
        out_frame=np.nanmedian(out_cube,axis=0)
    else:
        out_frame=np.nanmean(out_cube,axis=0)
        
    # cut it down. Count the number of NaNs in each row/column
    n_nans_x=np.sum(np.isnan(out_frame),axis=1)
    n_nans_y=np.sum(np.isnan(out_frame),axis=0)
    # Find the first elements that have less than n_pixels NaNs. 
    # Check both the +ve and -ve directions by reversing the array the second time.
    # The minimum of these is the one with the largest distance from the centre. 
    # Then turn into distance from the centre by doing n_pixels/2-index
    xradius=out_frame.shape[0]/2-np.min([np.argmax(n_nans_x < out_frame.shape[0]),np.argmax(n_nans_x[::-1] < out_frame.shape[0])])
    yradius=out_frame.shape[1]/2-np.min([np.argmax(n_nans_y < out_frame.shape[1]),np.argmax(n_nans_y[::-1] < out_frame.shape[1])])
    
    out_frame=out_frame[out_frame.shape[0]/2-xradius:out_frame.shape[0]/2+xradius,
                        out_frame.shape[1]/2-yradius:out_frame.shape[1]/2+yradius]
    
    if save_name:
        pyfits.writeto(save_name,out_frame,clobber=True)
        print 'Combined image saved as:',save_name
    return out_frame

###############

###############

def derotate_and_combine_multi(image_file,parang_file,save_name='derot.fits',
               threads=2,median_combine=False):
    '''Derotates and mean-combines a cube. Uses FFT derotation.
    This module is optimized for parallel computing on a single node using multiprocessing'''
    
    #Load the image
    if isinstance(image_file,str):
        cube=pyfits.getdata(image_file)
    else: # assume it is already a cube
        cube=image_file
    # Load the parangs
    try:
        parangs=np.loadtxt(parang_file)
    except:
        parangs=pyfits.getdata(parang_file)
    
    print 'Starting image derotation'
    
    # Initialize the pool and the list of variables needed
    pool=Pool(processes=threads)
    all_vars=[]
    for ix in range(cube.shape[0]):
        these_vars={'in_frame':cube[ix],'alpha':-parangs[ix],'pad':2,"return_full":True}
        all_vars.append(these_vars)
    out_cube=pool.map(fft_rotate_multi,all_vars)
    pool.close()

    # now sum into a final image
    if median_combine:
        out_frame=np.nanmedian(out_cube,axis=0)
    else:
        out_frame=np.nanmean(out_cube,axis=0)
        
    # cut it down. Count the number of NaNs in each row/column
    n_nans_x=np.sum(np.isnan(out_frame),axis=1)
    n_nans_y=np.sum(np.isnan(out_frame),axis=0)
    # Find the first elements that have less than n_pixels NaNs. 
    # Check both the +ve and -ve directions by reversing the array the second time.
    # The minimum of these is the one with the largest distance from the centre. 
    # Then turn into distance from the centre by doing n_pixels/2-index
    xradius=out_frame.shape[0]/2-np.min([np.argmax(n_nans_x < out_frame.shape[0]),np.argmax(n_nans_x[::-1] < out_frame.shape[0])])
    yradius=out_frame.shape[1]/2-np.min([np.argmax(n_nans_y < out_frame.shape[1]),np.argmax(n_nans_y[::-1] < out_frame.shape[1])])
    
    out_frame=out_frame[out_frame.shape[0]/2-xradius:out_frame.shape[0]/2+xradius,
                        out_frame.shape[1]/2-yradius:out_frame.shape[1]/2+yradius]
        
    if save_name:
        pyfits.writeto(save_name,out_frame,clobber=True)
        print 'Combined image saved as:',save_name
    return out_cube

###############

###############
