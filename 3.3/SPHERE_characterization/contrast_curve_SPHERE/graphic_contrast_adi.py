# -*- coding: utf-8 -*-
"""
Estimate the contrast from a GRAPHIC reduction for SPHERE

@author: peretti
"""

import numpy as np
import pyfits as pf
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal
import time
import scipy.ndimage
from scipy import interpolate
from fancy_plot import *


def graphic_contrast_adi(wdir,path_results,pattern_image,pattern_psf,fwhm,n_r,r_min,plate_scale,smoothing_length,contrast_in_mags,convol_with_circ_ap,smooth_image):
    for i,allfiles in enumerate(glob.iglob(pattern_psf+"*")):
        if i==0:
            flux_filename=allfiles
            print "psf filename:",flux_filename
        else:
            print "Error more than one file found with this pattern. Used the first one:",psf_filename
    for i,allfiles in enumerate(glob.iglob(pattern_image+"*")):
        if i==0:
            image_filename=allfiles
            print "image filename:",image_filename
        else:
            print "Error more than one file found with this pattern. Used the first one:",image_filename
    # reading image and psf (flux) files
    
    image_file=wdir+image_filename
    flux_file=wdir+flux_filename
    
    f=open("adi_self_sub.txt","r")
    lines=f.readlines()
    f.close()
    correction_seps=(lines[0].strip().split())[1:]
    correction_factor=(lines[1].strip().split())[1:]
    for i in range(np.size(correction_seps)):
        correction_seps[i]=float(correction_seps[i])
        correction_factor[i]=float(correction_factor[i])
    
    # Load the image and the flux image
    image,hdr=pf.getdata(image_file,header=True)
    flux_image,hdr_flux=pf.getdata(flux_file,header=True)
    target_name=hdr['OBJECT']
    DIT_image=hdr['HIERARCH ESO DET SEQ1 DIT']
    DIT_flux=hdr_flux['HIERARCH ESO DET SEQ1 DIT']
    
    
    band_filter_image=hdr['HIERARCH ESO INS COMB IFLT']
    band_filter_flux=hdr_flux['HIERARCH ESO INS COMB IFLT']
    
    if "left" in image_filename:
        filter1=band_filter_image[3:5] #ex: 'DB_H23' -> H2
    elif "right" in image_filename:
        filter1=band_filter_image[3]+band_filter_image[5] #ex: 'DB_H23' -> H3
    elif "sdi" in image_filename:
        filter1=band_filter_image[3:] #ex: 'DB_H23' -> H23
    else:
        print "Error: Couldn't find the filter for name. Please verify the target filename (left, right or sdi)"
    
    name=target_name+"_"+filter1+"_"+"adi" #name for the files produced at the end
    print name
    
    correction_factor=1./(1-np.array(correction_factor))
    r_max=np.shape(image)[0] # pixels
    
    # Numbers that are needed (and shouldnt change)
    n_sigma=5 # Number of sigma for the limits

    # SWITCHES
    #copy_absil=False
    
    import os.path
    if os.path.isfile(path_results+"companion_extraction.txt"):
        comp=True
        remove_planet=True
        f=open(path_results+"companion_extraction.txt",'r')
        lines=f.readlines()
        f.close()
        for line in lines:
            if line.strip().split()[0]==filter1:
                rho_comp=float(line.strip().split()[1])/plate_scale
                PA_comp=float(line.strip().split()[3])*np.pi/180.
                planet_pos=np.array([-np.sin(PA_comp)*rho_comp,np.cos(PA_comp)*rho_comp])+np.array([np.shape(image)[1]/2.,np.shape(image)[0]/2]) #for subtraction
        planet_rad=5 # in pixels for subtraction
    else:
        comp=False
        remove_planet=False

    # Cut the image (if needed)
    npix=image.shape[1]
    image=image[npix/2-r_max/2:npix/2+r_max/2,npix/2-r_max/2:npix/2+r_max/2]

    # Pixel distance map
    npix=image.shape[1]
    xarr=np.arange(0,npix)-npix/2
    xx,yy=np.meshgrid(xarr,xarr)
    pix_dist_map=np.sqrt(xx**2+yy**2)

    # Get rid of large-scale structure by doing a median filter (slow loop for now)
    if smooth_image:
        smoothed_image=scipy.ndimage.filters.median_filter(image,size=smoothing_length)
        final_image=image-smoothed_image
        image=final_image


    # Convolve the image with a circular aperture of rad=FWHM
    circ_ap=np.zeros((npix,npix))
    circ_ap[pix_dist_map<(fwhm/2)]=1
    convol_sz=np.int(np.ceil(fwhm)+3)
    circ_ap=circ_ap[npix/2-convol_sz/2:npix/2+convol_sz/2,npix/2-convol_sz/2:npix/2+convol_sz/2]
    if convol_with_circ_ap:
        image=np.nan_to_num(image)
        image=signal.fftconvolve(image,circ_ap,mode='same')

    flux_image=np.nan_to_num(flux_image)
    # Get the flux of the primary (make sure we do the same things to the flux image)
    if smooth_image:
        smoothed_flux_image=scipy.ndimage.filters.median_filter(flux_image,size=smoothing_length)
        flux_image=flux_image-smoothed_flux_image
    if convol_with_circ_ap:
        flux_image=signal.fftconvolve(flux_image,circ_ap,mode='same')
    primary_flux=np.nanmax(flux_image)


    import Neutral_density
    reload(Neutral_density)
    from Neutral_density import *

    lambda_image=band_filter(band_filter_image,image_filename)
    lambda_flux=band_filter(band_filter_flux,flux_filename)

    Neutral_density_image=1./Neutral_density(hdr,lambda_image)
    Neutral_density_flux=1./Neutral_density(hdr_flux,lambda_flux)
    print "Neutral density for image:", Neutral_density_image
    print "Neutral density for psf:", Neutral_density_flux

    primary_flux=(primary_flux*Neutral_density_flux*DIT_image)/(Neutral_density_image*DIT_flux)
    #primary_flux=1

    # remove the planet (if needed)
    if remove_planet:
        image[planet_pos[1]-planet_rad:planet_pos[1]+planet_rad,planet_pos[0]-planet_rad:planet_pos[0]+planet_rad]=np.nan

    # Get the arrays ready
    detec_limits=np.zeros(n_r)
    detec_limits_mag=np.zeros(n_r)
    detec_r=np.linspace(r_min,npix/2-fwhm/2.,n_r) # the mean radii of each annulus
    detec_r_arcsec=detec_r*plate_scale

    interpolation_correction_factor=interpolate.interp1d(correction_seps,correction_factor,bounds_error=False)
    correction_factor_vec_new=interpolation_correction_factor(detec_r_arcsec)
    correction_factor_vec_new=np.where(np.isnan(correction_factor_vec_new),np.nanmin(correction_factor_vec_new),correction_factor_vec_new)

    # Loop through annuli
    for r in np.arange(n_r):
    
        # What is this radius in pixels?
        r_in_pix=detec_r[r]-fwhm/2.
        r_out_pix=detec_r[r]+fwhm/2.
        #print 'Radius:',r_in_pix,r_out_pix
    
        # What pixels does that correspond to?
        pix=(pix_dist_map>r_in_pix) & (pix_dist_map < r_out_pix)
        vals=image[pix]

        # Correct for self-subtraction
        vals_corrected=vals*correction_factor_vec_new[r]

        # What is the scatter of these pixels?
        std=np.nanstd(vals_corrected)

        # What is the minimum detectable counts?
        detec_counts=n_sigma*std
    
        # Turn into contrast
        if contrast_in_mags:
            detec_contrast_mag=-2.5*np.log10(detec_counts/primary_flux)
        detec_contrast=detec_counts/primary_flux
    
        detec_limits_mag[r]=detec_contrast_mag
        detec_limits[r]=detec_contrast
    

    print 'Using an ADI self subtraction correction factor of:',np.round(correction_factor,3)
    print 'for separations of [arcsec]:',correction_seps

    ########################################
    # plot the detection limits
    ########################################

    #companion detected
    if comp:
        f=open(path_results+"companion_extraction.txt",'r')
        lines=f.readlines()
        f.close()
        for line in lines:
            if line.strip().split()[0]!='filter' and line.strip().split()[0][0]!='-':
                rho=float(line.strip().split()[1])
                erho=float(line.strip().split()[2])
                dmag=float(line.strip().split()[5])
                edmag=float(line.strip().split()[6])


    import os
    path_results=wdir+"Contrast_curve_resulting_files/"
    try:
        os.stat(path_results)
    except:
        os.mkdir(path_results) 


    ###############################
    # use a nice font
    font = {'family' : 'Bitstream Vera Sans',
            'weight' : 'normal',
            'size'   : 17}

    matplotlib.rc('font', **font)
    plt.close('all')
    
    ylims=[1e-8,1e-2] #plot y axis limits
    ylims=[np.nanmin(detec_limits)*0.1,np.nanmax(detec_limits)*10]
    ylims_mags=[np.nanmin(detec_limits_mag)-1,np.nanmax(detec_limits_mag)+1] #plot y axis in mag limits
    xlims=[0.,8] #plot x axis limit
    #contrast_in_mags:
    if contrast_in_mags:
        fig=plt.figure()
        ax = fig.add_subplot(111)
        #ax.fill_between(detec_r_arcsec, np.zeros(np.size(detec_r_arcsec))+18, detec_limits_mag, facecolor='blue', alpha=1.0)
        #p1 = plt.Rectangle((0, 0), 0, 0, fc="blue",alpha=0.1)
        #ax.legend([p1], ["detected"],loc=10).get_frame().set_linewidth(0.0)
        #ax.legend(loc='upper left')
        if comp:
            ax.errorbar(rho,abs(dmag),erho,edmag,'.r')
        fancy_plot(detec_r_arcsec,detec_limits_mag,"Contrast curve",'Angular separation [arcsec]',r''+str(n_sigma)+'$\sigma$ contrast [mag]',leg=False, xlim=xlims, ylim=ylims_mags, xscale='std', yscale='std', width=1, line_color='b', style="standard",grid=True,filename_to_save=path_results+name+"_mag.png",text=False,fig=False)
    

    #contrast in flux
    fancy_plot(detec_r_arcsec,detec_limits,"Contrast curve",'Angular separation [arcsec]',r''+str(n_sigma)+'$\sigma$ contrast',leg=False, xlim=xlims, ylim=ylims, xscale='std', yscale='log', width=1, line_color='b', style="standard",grid=True,filename_to_save=path_results+name+"_flux.png")


    plt.figure()
    plt.plot(correction_seps,correction_factor,'-x')
    plt.xlabel("separation [arcsec]")
    plt.ylabel("self subtraction factor")
    plt.title("Self subtraction factor")
    plt.savefig(path_results+"ADI_self_sub.png",dpi=300)

    #plt.show()

    ###############################
    #ecriture des fichiers 
    ###############################

    pf.writeto(path_results+'image_mask_planet.fits',image,clobber=True)

    f=open(path_results+"contrast_curve.rdb","w")
    f.write("detec_r_arcsec")
    f.write("\t")
    f.write("detec_limits_mag")
    f.write("\t")
    f.write("Informations")
    f.write("\n")
    for i in range(np.size(detec_r_arcsec)):
        f.write(str(detec_r_arcsec[i]))
        f.write("\t")
        f.write(str(detec_limits_mag[i]))
        if i==0:
            f.write("\t")
            f.write(hdr["HIERARCH ESO INS COMB IFLT "])
        f.write("\n")
    f.close()

