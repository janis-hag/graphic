# -*- coding: utf-8 -*-
"""
Some PCA code for ADI
"""

import scipy,time,pickle
import bottleneck
import numpy as np
import astropy.io.fits as pyfits
#from scipy import ndimage
from multiprocessing import Pool
import graphic_nompi_lib_330 as graphic_nompi_lib
try:
    import pyfftw.interfaces.scipy_fftpack as fftpack
except:
    print('Failed to load pyfftw')
    from scipy import fftpack


###############

###############

def principal_components(pix_array,n_modes=None):
    ''' Calculates the principal components for a set of flattened pixel arrays
     (i.e. pix_array must be 2D, and is n_frames x n_pixels)p
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
   
def subtract_principal_components(pcomps,image_region,mask=False):
    ''' Projects an array of pixel values onto the principal components and 
    subtracts the result from the original values.
    mask: A boolean array that masks out some pixels when calculating the 
          coefficients used for the projection. e.g. useful for excluding the star
          when performing PCA sky subtraction.'''
          
    if (mask is False):
        transformed = np.dot(pcomps, image_region.T) #i.e. the coefficients
        reconstructed = np.dot(transformed.T, pcomps)
        residuals = image_region - reconstructed
    else:
        pcomps_windowed = pcomps[:,mask]
        image_windowed = image_region[:,mask]
        transformed = np.dot(pcomps_windowed, image_windowed.T) #i.e. the coefficients
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
    ''' A wrapper for the pca step, needed for multiprocessing. The parallelization
    is done over regions of the image, so each process needs access to the entire
    datacube for a small region only, and no processes need to share memory.

    cube_region is a 2D cube of all of the pixels in the region from all of the frames
        i.e. it is (n_frames, n_pixels_in_region)
      '''

    # Unpack the data cube for this region of the image
    cube_region = all_vars['cube_region']
    region_ix = all_vars['region_ix']

    # Set up the output array
    region_out=np.zeros(cube_region.shape)

    for frame_ix,frame in enumerate(all_vars['cube_region']):
        
        # Find which frames are ok to use
        good_frames = all_vars['all_good_frames'][frame_ix][region_ix]
                
        frame = cube_region[frame_ix]
        reference_frames = cube_region[good_frames]
        
        # Get the principal components
        pcomps=principal_components(reference_frames,n_modes=all_vars['n_modes'])      
        # Subtract them from the data
        resids=subtract_principal_components(pcomps,frame)
        # Save it out
        region_out[frame_ix]=resids

    if (region_ix % 5) ==4:
        n_regions = len(all_vars['all_good_frames'][0])
        time_left=(n_regions-region_ix-1)*(time.time()-all_vars['t_start'])/(region_ix+1)
        print('  Region '+str(region_ix+1)+' done. Time remaining: '+str(np.round(time_left/60.,2))+'mins' )
    return region_out

###############

###############

def find_reference_frames(parangs, radii, fwhm, n_fwhm, min_frames=None):
    ''' Find which frames can be used as reference frames in PCA. For each frame, 
    this calculates the number of FWHM that the parallactic angle change of every 
    other frame corresponds to.
    
    This is used in smart_annular_pca to minimize self-subtraction.
    
    parangs: parallactic angles for each frame.
    radii: radius of the annulus or region (in pixels). If this is a list, the 
        calculation will loop over radii as well
    fwhm: full-width at half-maximum of PSF (in pixels)
    n_fwhm: minimum number of FWHM of rotation required at the requested radius
    min_frames: Minimum number of frames that will be returned. If not enough 
        frames fit the fwhm criteria, it will return this many frames instead.
    '''
    
    # Check input types are iterables so I can loop over them
    if not hasattr(parangs,'__iter__'):
        parangs = np.array([parangs])
    if not hasattr(radii,'__iter__'):
        radii = [radii]
        
    # Make them numpy arrays
    parangs = np.array(parangs)
    radii = np.array(radii)

    all_good_frames = []
    # Loop through frames
    for parang in parangs:
        
        # Calculate the parang differences
        delta_parang = parangs - parang
        
        good_frames_this_parang = []
        # Loop over radius
        for radius in radii:
            # How far does a structure move in each frame (in pix)?
            rot_dist=np.abs(2*radius*np.sin(delta_parang*np.pi/180/2.))
            
            # Take those that satisfy the criteria
            good_frames = rot_dist > (n_fwhm*fwhm)
            
            # Now check that we have enough frames
            if np.sum(good_frames) < min_frames:
                
                # then take the ones with max parang rotation  
                extra_frames = np.argsort(np.abs(delta_parang))[-min_frames:]
                
                # and append them
                good_frames[extra_frames] = True
            
            # Add it to the list for this frame
            good_frames_this_parang.append(good_frames)
        
        # And add the list for this frame to the list for all frames
        all_good_frames.append(good_frames_this_parang)
    
    return all_good_frames

###############

###############

def fft_rotate_multi(all_vars):
    ''' A wrapper for fft_3shear_rotate_pad, needed for multiprocessing'''
    return graphic_nompi_lib.fft_3shear_rotate_pad(all_vars['in_frame'],all_vars['alpha'],all_vars['pad'],
                      return_full=all_vars['return_full'])

###############

###############

def define_annuli(npix,n_radii,arc_length,r_min,r_max='Default'):
    ''' Defines the annuli used for pca. The image is divided into these regions
    and the principal components are calculated for each region individually.
    npix is the number of pixels in the image. The image is assumed to be square!
    rmin is the minimum radius for the annuli. npix/2 is expected to be the maximum radius
    '''

    npix=np.int(npix)
    n_radii=np.int(n_radii)
    arc_length=np.int(arc_length)
    r_min=np.int(r_min)
    if r_max == 'Default':
        r_max=np.sqrt(2)*npix/2
    
    radii=np.linspace(r_min,r_max,num=(n_radii+1))

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
        if n_azimuth <1: # make sure there is at least one segment...
            n_azimuth =1

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
    cube,hdr = pyfits.getdata(image_file,header=True)
    initial_shape=cube.shape
    cube=cube.reshape([cube.shape[0],cube.shape[1]*cube.shape[2]])
    
    # median subtract
    cube-=bottleneck.nanmedian(cube)
    
    # Set the NaNs to zero and restore them later
    nan_mask=np.isnan(cube)
    cube[nan_mask]=0.
    
    # Get the principal components
    pcomps=principal_components(cube,n_modes=n_modes)
    
    # Subtract them from the data
    cube_out=subtract_principal_components(pcomps,cube)
    
    # Restore the NaNs
    cube_out[nan_mask] = np.nan
    
    # Make the cube 3d again
    cube_out=cube_out.reshape(initial_shape)
    
    if save_name:
        # Make a header to store the reduction parameters
        hdr = make_pca_header('simple_pca',n_modes, hdr=hdr, image_file=image_file)
        
        pyfits.writeto(save_name,cube_out,header=hdr,clobber=True,output_verify='silentfix')
        print('  PCA subtracted cube saved as:',save_name)
    
    if pc_name:
        data={'principal_components':pcomps.astype(np.float32),'regions':None,'npix':initial_shape[1],
              'protection_angle':None}
        with open(pc_name,'w') as myf:
            pickle.dump(data,myf)
    
    return cube_out
        

###############

###############

def annular_pca(image_file,n_modes,save_name,n_annuli=5,arc_length=50,r_min=5,
                pc_name=None,threads=2,r_max='Default'):
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
    cube,hdr = pyfits.getdata(image_file,header=True)
    initial_shape=cube.shape
    cube=cube.reshape([cube.shape[0],cube.shape[1]*cube.shape[2]])
    cube_out = np.zeros(cube.shape)+np.NaN
    
    # Calculate the annular regions
    print('  Defining annuli')
    regions,region_radii=define_annuli(initial_shape[1],n_annuli,arc_length,r_min=r_min,r_max=r_max)
            
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
    cube_out=np.zeros(cube.shape)+np.NaN
    pcomps=np.zeros((n_modes,cube.shape[1]))
    for region_ix,region in enumerate(regions):
        cube_out[:,region]=pca_output[region_ix][0]
        pcomps[:,region]=pca_output[region_ix][1]
        
      
    # Make the cube 3d again
    cube_out=cube_out.reshape(initial_shape)
    pcomps=pcomps.reshape((n_modes,cube_out.shape[1],cube_out.shape[2]))
    if save_name:
        # Make a header to store the reduction parameters
        hdr = make_pca_header('annular_pca',n_modes, n_annuli=n_annuli, 
                  arc_length=arc_length, hdr=hdr,
                  r_min=r_min, r_max=r_max, image_file=image_file)
        
        pyfits.writeto(save_name,cube_out,header=hdr,clobber=True,output_verify='silentfix')
        print('  PCA subtracted cube saved as: '+save_name)
    
    if pc_name:
        data={'principal_components':pcomps.astype(np.float32),'regions':regions,'npix':initial_shape[1],
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
    cube,hdr = pyfits.getdata(image_file,header=True)
    initial_shape=cube.shape
    cube=cube.reshape([cube.shape[0],cube.shape[1]*cube.shape[2]])
    # Load the parallactic angles
    if isinstance(parang_file,str):
        if parang_file.endswith('.txt'):
            parangs=np.loadtxt(parang_file)
        else:
            parangs=pyfits.getdata(parang_file)
    else: # assume it is an array
        parangs = parang_file
    
    cube_out=np.zeros(cube.shape)+np.NaN
    
    # Set the NaNs to zero and restore them later
    nan_mask=np.isnan(cube)
    cube[nan_mask]=0.
    
    t_start=time.time()
    # Loop through frames in the cube
    print('smart_pca: starting loop over frames in image')
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
            print('  Done',ix+1,'of',cube.shape[0],np.round(time_left/60.,2),'mins remaining')
    
    cube_out[nan_mask]=np.nan
    # Make the cube 3d again
    cube_out=cube_out.reshape(initial_shape)
    if save_name:
        # Make a header to store the reduction parameters
        hdr = make_pca_header('smart_pca',n_modes, hdr=hdr, image_file=image_file)
        
        pyfits.writeto(save_name,cube_out,header=hdr,clobber=True,output_verify='silentfix')
        print('  PCA subtracted cube saved as:',save_name)

    if pc_name:
        data={'principal_components':pcomps.astype(np.float32),'regions':None,'npix':initial_shape[1],
              'protection_angle':protection_angle}
        with open(pc_name,'w') as myf:
            pickle.dump(data,myf)
    return cube_out

###############

###############

def smart_annular_pca(image_file,n_modes,save_name,parang_file,n_fwhm=2,fwhm=4.5,
              pc_name=None,n_annuli=5,arc_length=50,r_min=5,threads=3,r_max='Default',
              min_reference_frames=5,silent=False,hdr=None):
    ''' Performs a smart PCA reduction on the input datacube, with PCA performed
    locally on annuli, using only frames that have a parallactic angle
    difference that is above some threshold, to minimize self-subtraction.
    n_modes is the number of modes to remove from the image
    
    pc_name is the name of the file that the principal components will be saved (as a pickle file)
    '''
    if not silent:
        print('Running smart annular PCA')
    
    # Load the datacube and make it 2d
    if isinstance(image_file,str):
        cube,hdr = pyfits.getdata(image_file,header=True)
    else:
        cube = image_file # assume that the cube was the input if it was not a filename
        hdr = pyfits.Header()
        image_file=''
    initial_shape=cube.shape
    # cube = cube.astype(np.float32) # We don't need this much precision, and this saves RAM
    cube=cube.reshape([cube.shape[0],cube.shape[1]*cube.shape[2]])
    
    # Load the parallactic angles
    if isinstance(parang_file,str):
        if parang_file.endswith('.txt'):
            parangs=np.loadtxt(parang_file)
        else:
            parangs=pyfits.getdata(parang_file)
    else: # assume it is an array
        parangs = parang_file
    
    cube_out=np.zeros(cube.shape)+np.NaN
    
    # Set the NaNs to zero and restore them later
    nan_mask=np.isnan(cube)
    cube[nan_mask]=0.
    
    # Calculate the annular regions
    if not silent:
        print('  Defining annuli')
    regions,region_radii=define_annuli(initial_shape[1],n_annuli,arc_length,r_min=r_min,r_max=r_max)
    
    # Calculate which reference frames to use for each frame in the cube
    all_good_frames = find_reference_frames(parangs,region_radii,fwhm,n_fwhm,
                                            min_frames = min_reference_frames)
    # Count how many reference frames each region and frame has
    n_good_frames = []
    for parang_frames in all_good_frames:
        for region_frames in parang_frames:
            n_good_frames.append(np.sum(region_frames))
    
    # Check that the rotation is enough for the closest annuli
    if not silent:
        print('  Minimum of: '+str(np.min(n_good_frames))+' frames per annuli')
        print('  Using '+str(n_modes)+' modes, '+str(n_fwhm)+' FWHM protection angle,')
        print('  '+str(fwhm)+' pix FWHM, '+str(n_annuli)+' annuli, '+str(arc_length)+' pix arc length,')
        print('  minimum and maximum radii of '+str(r_min)+' and '+str(r_max)+' pix')
    
    if (np.min(n_good_frames) < n_modes) and not silent:
        print(' WARNING: some annuli have less available frames than the requested number of modes to subtract')
        if np.min(n_good_frames) ==0:
            # What rotation do we need to get at least one frame
            min_rot=2*np.arcsin(n_fwhm*fwhm/(2*np.min(region_radii)))*180./np.pi
            print('Minimum rotation needed for inner annuli: '+str(np.round(min_rot,decimals=1)))
#            print('Found: '+str(np.round(np.max(parangs)-np.min(parangs),decimals=1))) # This doesnt work so dont bother printing it
            raise ValueError('Need at least 1 frame for PCA! Try decreasing n_fwhm or increasing min_reference_frames')
            
    # Set up the pool and arrays for the loop
    pool=Pool(processes=threads)
    cube_out = np.zeros(cube.shape)+np.NaN
    t_start=time.time()
#    pcomps=np.zeros((n_modes,cube.shape[0],cube.shape[1]))

    if not silent:
        print('Setting up arrays for the loop')
    # Set up arrays for the loop
    global region_cube
    region_cube=[]
    for region in regions:
        region_cube.append(cube[:,region]) # restructuring this now might save time later
    all_vars=[]

    # New RAM saving way: Loop over regions instead of frames
    for region_ix,region in enumerate(regions): 
        cube_region = region_cube[region_ix]
        these_vars={'region_ix':region_ix, 'n_modes':n_modes, 'region':region,
        'all_good_frames':all_good_frames,'t_start':time.time(),
        'cube_region':cube_region}
        all_vars.append(these_vars)  
    
    if not silent:
        print('  Starting loop over regions in image')
    # Now let multiprocessing do it all
    pca_output=pool.map(pca_multi_annular,all_vars,chunksize=1)

    # Close the threads
    pool.close()
    all_vars=[] # free up some RAM
    region_cube = []
    
    # Uncomment these lines to do it without multiprocessing
    # for ix,these_vars in enumerate(all_vars):
    #    pca_output=pca_multi_annular(these_vars)
    #    cube_out[ix]=pca_output[0]

    if not silent:
        print('  Done! Took '+str((time.time()-t_start)/60)+' mins')
        print('  Reordering the output')
    # Reorder it all and store in the output arrays
    # It is n_regions by n_frames
    for ix in range(len(pca_output)):
        output=pca_output[ix]
        region = regions[ix]
        cube_out[:,region]=output

        pca_output[ix]=[]
        
    if not silent:
        print('  Reordered output') 
    
    cube_out[nan_mask]=np.nan
    # Make the cube 3d again
    cube_out=cube_out.reshape(initial_shape)

    # Now cut it to r_max (if applicable)
    if (r_max < initial_shape[1]/2.) and (r_max < initial_shape[2]/2.):
        cube_out = cube_out[:,np.int(cube_out.shape[1]//2-r_max):np.int(cube_out.shape[1]//2+r_max),
                              np.int(cube_out.shape[2]//2-r_max):np.int(cube_out.shape[2]//2+r_max)]
    if save_name:

        # Make a header to store the reduction parameters
        hdr = make_pca_header('smart_annular_pca',n_modes, n_fwhm=n_fwhm,
                  fwhm=fwhm, n_annuli=n_annuli, arc_length=arc_length,
                  hdr=hdr, r_min=r_min, r_max=r_max, image_file=image_file,
                  min_reference_frames=min_reference_frames)
        
        pyfits.writeto(save_name,cube_out,header=hdr,clobber=True,output_verify='silentfix')
        if not silent:
            print('  PCA subtracted cube saved as: '+save_name)

    return cube_out

###############

###############

def make_pca_header(pca_type,n_modes,hdr=None,n_fwhm='',fwhm='',n_annuli='',
                    arc_length='',r_min='',r_max='',image_file='',
                    min_reference_frames = ''):
    ''' Function that makes a basic header for the output PCA files.
    This is in a separate function since it is used by all of the PCA
    routines (smart_pca, smart_annular_pca etc.)
    '''

    # Make a header to store the reduction parameters
    if hdr is None:
        hdr = pyfits.Header()
    
    hdr['HIERARCH GC PCA TYPE'] = 'smart_annular'
    hdr['HIERARCH GC PCA NMODES'] = n_modes
    hdr['HIERARCH GC PCA NFWHM'] = n_fwhm
    hdr['HIERARCH GC PCA FWHM'] = fwhm
    hdr['HIERARCH GC PCA NANNULI'] = n_annuli
    hdr['HIERARCH GC PCA ARCLENGTH'] = arc_length
    hdr['HIERARCH GC PCA RMIN'] = r_min
    hdr['HIERARCH GC PCA RMAX'] = r_max
    hdr['HIERARCH GC PCA INPUTFILE'] = image_file.split('/')[-1]
    hdr['HIERARCH GC PCA MINREFFRAMES'] = min_reference_frames
    
    return hdr
###############

###############

def classical_adi(image_file,save_name,parang_file,median=False,silent=False,hdr=None):
    ''' Runs classical ADI on an image cube. Subtracts the mean or median of the cube
    from every individual frame. No protection angle is used.'''

    if isinstance(image_file,str):
        cube,hdr = pyfits.getdata(image_file,header=True)
    else:
        cube = image_file

    # Load the parallactic angles
    if isinstance(parang_file,str):
        if parang_file.endswith('.txt'):
            parangs=np.loadtxt(parang_file)
        else:
            parangs=pyfits.getdata(parang_file)
    else: # assume it is an array
        parangs = parang_file

    if not hdr:
        hdr = pyfits.Header()

    # Force the cube to be in the right format (to remove endian problems)
    cube = cube.astype(np.float)

    # Get the mean or median frame
    if median:
        psf = bottleneck.nanmedian(cube,axis=0)
    else:
        psf = bottleneck.nanmean(cube,axis=0)

    out_cube = 1*cube
    out_cube -= psf

    if save_name:
        hdr = make_pca_header('cADI',1,
                  hdr=hdr, image_file=image_file)
        pyfits.writeto(save_name,out_cube,header=hdr,clobber=True)
    else:
        return out_cube

###############

###############

def derotate_and_combine(image_file,parang_file,save_name='derot.fits',
                 median_combine=False,return_cube=True,silent=False):
    '''Derotates and mean-combines a cube. Uses FFT derotation '''
    
    #Load the image
    if isinstance(image_file,str):
        cube, hdr =pyfits.getdata(image_file,header=True)
    else: # assume it is already a cube
        cube=image_file
        hdr = pyfits.Header()
    # Load the parallactic angles
    if isinstance(parang_file,str):
        if parang_file.endswith('.txt'):
            parangs=np.loadtxt(parang_file)
        else:
            parangs=pyfits.getdata(parang_file)
    else: # assume it is an array
        parangs = parang_file

    
    out_cube=np.zeros(cube.shape*np.array([1,2,2]))+np.nan # assuming pad=2 in fft_rotate
    t_start=time.time()

    if not silent:
        print('Starting image derotation')
    # Loop through the images
    for ix in range(cube.shape[0]):
        in_frame=cube[ix]
        derot_frame=graphic_nompi_lib.fft_3shear_rotate_pad(in_frame,-parangs[ix], pad=2,return_full=True)
        out_cube[ix]=derot_frame
        
        if ((ix % 20) ==19) and not silent:
            time_left=(cube.shape[0]-ix-1)*(time.time()-t_start)/(ix+1)
            print('  Done',ix+1,'of',cube.shape[0],np.round(time_left/60.,2),'mins remaining')
        
    # now sum into a final image
    if median_combine:
        out_frame=bottleneck.nanmedian(out_cube,axis=0)
    else:
        out_frame=bottleneck.nanmean(out_cube,axis=0)
        
    # cut it down. Count the number of NaNs in each row/column
    n_nans_x=np.sum(np.isnan(out_frame),axis=1)
    n_nans_y=np.sum(np.isnan(out_frame),axis=0)
    # Find the first elements that have less than n_pixels NaNs. 
    # Check both the +ve and -ve directions by reversing the array the second time.
    # The minimum of these is the one with the largest distance from the centre. 
    # Then turn into distance from the centre by doing n_pixels/2-index
    xradius=out_frame.shape[0]//2-np.min([np.argmax(n_nans_x < out_frame.shape[0]),np.argmax(n_nans_x[::-1] < out_frame.shape[0])])
    yradius=out_frame.shape[1]//2-np.min([np.argmax(n_nans_y < out_frame.shape[1]),np.argmax(n_nans_y[::-1] < out_frame.shape[1])])
    
    # actually just take the maximum of these so the output is square
    xradius=np.max([xradius,yradius])
    yradius=xradius
    
    out_cube=out_cube[:,out_frame.shape[0]//2-xradius:out_frame.shape[0]//2+xradius,
                        out_frame.shape[1]//2-yradius:out_frame.shape[1]//2+yradius]
    out_frame=out_frame[out_frame.shape[0]//2-xradius:out_frame.shape[0]//2+xradius,
                        out_frame.shape[1]//2-yradius:out_frame.shape[1]//2+yradius]
    
    if save_name:
        pyfits.writeto(save_name,out_frame,header=hdr,clobber=True,output_verify='silentfix')
        if not silent:
            print('Combined image saved as:',save_name)
    if return_cube:
        return out_cube

###############

###############

def derotate_and_combine_multi(image_file,parang_file,save_name='derot.fits',
               threads=2,median_combine=False,return_cube=True,hdr=None,silent=False):
    '''Derotates and mean-combines a cube. Uses FFT derotation.
    This module is optimized for parallel computing on a single node using multiprocessing'''
    
    #Load the image
    if isinstance(image_file,str):
        cube, hdr = pyfits.getdata(image_file,header=True)
    else: # assume it is already a cube
        cube=image_file

    # Load the parallactic angles
    if isinstance(parang_file,str):
        if parang_file.endswith('.txt'):
            parangs=np.loadtxt(parang_file)
        else:
            parangs=pyfits.getdata(parang_file)
    else: # assume it is an array
        parangs = parang_file
    
    if not silent:
        print('Starting image derotation')
    
    # Initialize the pool and the list of variables needed
    pool=Pool(processes=threads)
    all_vars=[]
    for ix in range(cube.shape[0]):
        these_vars={'in_frame':cube[ix],'alpha':-parangs[ix],'pad':2,"return_full":True}
        all_vars.append(these_vars)

    # Use imap to save RAM
    imap_pool = pool.imap(fft_rotate_multi,all_vars)
    # In the following line, the *2 is because pad = 2, and 
    # the +2 is because we fft_3shear_rotate adds 2 pixels to the array before padding
    out_cube = np.zeros((cube.shape[0],(cube.shape[1]+2)*2,(cube.shape[2]+2)*2)) 
    for ix in range(cube.shape[0]):
        out_cube[ix] = imap_pool.next()


    pool.close()
     
    # Old way
    # out_cube=pool.map(fft_rotate_multi,all_vars)
    # out_cube = np.array(out_cube,copy=False)

    # now sum into a final image
    if median_combine:
        out_frame=bottleneck.nanmedian(out_cube,axis=0)
    else:
        out_frame=bottleneck.nanmean(out_cube,axis=0)
        
    # Cut down the output frame to remove the NaNs at the edges
#    xradius=None
#    out_frame = crop_nans(out_frame,xradius=xradius)
#    
    # cut it down. Count the number of NaNs in each row/column
#    
    n_nans_x=np.sum(np.isnan(out_frame),axis=1)
    n_nans_y=np.sum(np.isnan(out_frame),axis=0)
    # Find the first elements that have less than n_pixels NaNs. 
    # Check both the +ve and -ve directions by reversing the array the second time.
    # The minimum of these is the one with the largest distance from the centre. 
    # Then turn into distance from the centre by doing n_pixels/2-index
    xradius=out_frame.shape[0]//2-np.min([np.argmax(n_nans_x < out_frame.shape[0]),np.argmax(n_nans_x[::-1] < out_frame.shape[0])])
    yradius=out_frame.shape[1]//2-np.min([np.argmax(n_nans_y < out_frame.shape[1]),np.argmax(n_nans_y[::-1] < out_frame.shape[1])])
    
    # actually just take the maximum of these so the output is square
    xradius=np.max([xradius,yradius])
    yradius=xradius


    out_cube=out_cube[:,out_frame.shape[0]//2-xradius:out_frame.shape[0]//2+xradius,
                        out_frame.shape[1]//2-xradius:out_frame.shape[1]//2+xradius]    
    out_frame=out_frame[out_frame.shape[0]//2-xradius:out_frame.shape[0]//2+xradius,
                        out_frame.shape[1]//2-yradius:out_frame.shape[1]//2+yradius]
        
    if save_name:
        pyfits.writeto(save_name,out_frame,header=hdr,clobber=True,output_verify='silentfix')
        if not silent:
            print('Combined image saved as:',save_name)
    
    if return_cube:
        return out_cube

###############

###############

def crop_nans(cube,xradius=None):
    ''' Crop an image to remove NaNs that mark the edge of the field of view
    Always returns a square image
    '''
    
    # Count the number of NaNs in each row/column
    n_nans_x=np.sum(np.isnan(cube),axis=-2)
    n_nans_y=np.sum(np.isnan(cube),axis=-1)
    # Find the first elements that have less than n_pixels NaNs. 
    # Check both the +ve and -ve directions by reversing the array the second time.
    # The minimum of these is the one with the largest distance from the centre. 
    # Then turn into distance from the centre by doing n_pixels/2-index
    xradius=cube.shape[-2]//2-np.min([np.argmax(n_nans_x < cube.shape[-2]),np.argmax(n_nans_x[::-1] < cube.shape[-2])])
    yradius=cube.shape[-1]//2-np.min([np.argmax(n_nans_y < cube.shape[-1]),np.argmax(n_nans_y[::-1] < cube.shape[-1])])
    
    # actually just take the maximum of these so the output is square
    xradius=np.max([xradius,yradius])
    yradius=xradius
    
    # Now crop it
    if cube.ndim == 3:
        cube = cube[:,cube.shape[-2]//2-xradius:cube.shape[-2]//2+xradius,
                        cube.shape[-1]//2-yradius:cube.shape[-1]//2+yradius]
    elif cube.ndim == 2:
        cube = cube[cube.shape[-2]//2-xradius:cube.shape[-2]//2+xradius,
                cube.shape[-1]//2-yradius:cube.shape[-1]//2+yradius]
                
    return cube

###############