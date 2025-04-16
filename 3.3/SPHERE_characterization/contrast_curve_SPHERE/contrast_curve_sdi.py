# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyfits,glob
import read_cond_model
reload(read_cond_model)
from read_cond_model import *
from fancy_plot import *
import urllib
import urllib2
import os

def contrast_curve_sdi(wdir,pattern_image,age,plate_scale):
    for i,allfiles in enumerate(glob.iglob(pattern_image+"*")):
        if i==0:
            image_filename=allfiles
            print("image filename:",image_filename)
        else:
            print("Error more than one file found with this pattern. Used the first one:",image_filename)
    im,hdr=pyfits.getdata(image_filename,header=True)
    target=hdr['OBJECT']
    # directory of the images
    path_psf=wdir
    psf_left_filename='psf_left.fits'
    psf_right_filename='psf_right.fits'
    path_results='Contrast_curve_resulting_files/'


    # SETTINGS
    band_filter_image=hdr['HIERARCH ESO INS COMB IFLT']
    if "Y23" in band_filter_image:
    	filter1="Y2"
    	filter2="Y3"
    elif "J12" in band_filter_image:
    	filter1="J1"
    	filter2="J2"
    elif "H23" in band_filter_image:
    	filter1="H2"
    	filter2="H3"
    elif "K12" in band_filter_image:
    	filter1="K1"
    	filter2="K2"
    else:
		sys.exit("Error could not find the wright filter in image header \"HIERARCH ESO INS COMB IFLT\"")

    binning_factor=100

    if os.path.isfile(path_results+"companion_extraction.txt"):
        comp=True #if a companion has been detected to put it on the graph
    else:
		comp=False

    #Query Simbad for mag and parallax
    data = urllib.urlencode({'script': 'format object form1 "%PLX(V) | %FLUXLIST(H;F)"\nhd4113'})
    reply = urllib2.urlopen("http://simbad.u-strasbg.fr/simbad/sim-script", data).read()
    parallax=float(reply.strip().split('\n')[-1].strip().split()[0]) #in mas
    Mag_star=float(reply.strip().split('\n')[-1].strip().split()[2])


    masses,Teff,abs_mag_left=read_model(age,filter1,parallax)
    masses2,Teff2,abs_mag_right=read_model(age,filter2,parallax)

    distance=1./(parallax/1000.) #in parsec
    apparent_mag_left=abs_mag_left+5.*np.log10(distance)-5.
    apparent_mag_right=abs_mag_right+5.*np.log10(distance)-5.
    delta_mag_left=apparent_mag_left-Mag_star #delta magnitude of left chanel comming from models
    delta_mag_right=apparent_mag_left-Mag_star #delta magnitude of right chanel comming from models

    ########################################################################################
    #computing the curve of self subtraction vs radius for sdi subtraction
    ########################################################################################

    psf_l=pyfits.getdata(path_psf+psf_left_filename)
    psf_r=pyfits.getdata(path_psf+psf_right_filename)

    f=open(wdir+path_results+'contrast_curve.rdb',"r")
    lines=f.readlines()
    f.close()
    band_filter=lines[1].split()[2]

    from contrast_curve_sdi_func import *
    contrast_curve_vec=np.zeros((np.size(masses),512))
    flux_left=np.power(10,-apparent_mag_left/2.5)
    flux_right=np.power(10,-apparent_mag_right/2.5)

    for i in range(np.size(masses)):
        print("\n masse",i+1,"/",np.size(masses),"\n")
        contrast_curve_vec[i]=contrast_curve_sdi_func(band_filter,psf_l,psf_r,binning_factor,flux_left[i],flux_right[i])
    contrast_curve_vec_mag=-2.5*np.log10(contrast_curve_vec)-Mag_star

    contrast_curve_vec_mass=np.zeros((np.shape(contrast_curve_vec_mag)))
    for i in range(np.shape(contrast_curve_vec_mag)[0]):
        for j in range(np.shape(contrast_curve_vec_mag)[1]):
            mag_to_transform=contrast_curve_vec_mag[i,j]
            if np.shape(np.where(delta_mag_left<mag_to_transform))[1]==0:
                contrast_curve_vec_mass[i,j]=np.nan
            elif np.where(delta_mag_left<mag_to_transform)[0][0]>0:
                mag_inf=delta_mag_left[np.where(delta_mag_left<mag_to_transform)[0][0]]
                mag_sup=delta_mag_left[np.where(delta_mag_left<mag_to_transform)[0][0]-1]
                mag_interval=[mag_sup,mag_inf]
                index_inf=np.where(delta_mag_left==mag_inf)
                index_sup=np.where(delta_mag_left==mag_sup)
                mass_interval=[masses[index_sup][0]*1048,masses[index_inf][0]*1048] #in jupiter mass
                a=(mag_interval[0]-mag_interval[1])/(mass_interval[0]-mass_interval[1])
                b=mag_interval[0]
                contrast_curve_vec_mass[i,j]=(mag_to_transform-b)/a+mass_interval[0]
            else:
                contrast_curve_vec_mass[i,j]=np.nan

    arcsec_sep=np.arange(np.shape(contrast_curve_vec_mass)[1])*plate_scale

    f=open(wdir+path_results+"contrast_curve.rdb","r")
    lines=f.readlines()
    f.close()

    detec_r_arcsec=[]
    detec_limits_mag=[]
    index=0
    for line in lines:
        if index>0:
            detec_r_arcsec=np.append(detec_r_arcsec,float(line.split()[0]))
            detec_limits_mag=np.append(detec_limits_mag,float(line.split()[1]))
        index+=1

    from scipy import interpolate
    new_detec_r_arcsec=np.linspace(detec_r_arcsec[0],detec_r_arcsec[np.size(detec_r_arcsec)-1],np.shape(contrast_curve_vec_mass)[1])
    interpolation_detec_limits_mag=interpolate.interp1d(detec_r_arcsec,detec_limits_mag)
    new_detec_limits_mag=interpolation_detec_limits_mag(new_detec_r_arcsec)

    interpolation_mass_vec=[]
    for i in range(np.shape(contrast_curve_vec_mag)[1]):
        interpolation_mass_temp=interpolate.interp1d(contrast_curve_vec_mag[:,i],masses*1048,bounds_error=False)
        interpolation_mass_vec=np.append(interpolation_mass_vec,interpolation_mass_temp)

    from scipy import stats
    contrast_curve_mass=[]
    for i in range(np.shape(contrast_curve_vec_mag)[1]):
        if new_detec_limits_mag[i]>np.max(contrast_curve_vec_mag[:,i]): #if we are above the minimum mass curve we prolongate the curve linearly
            mag_vec_temp=np.arange(100)/100.+np.max(contrast_curve_vec_mag[:,i])-1 #we take the points for the last magnitude
            slope, intercept, r_value, p_value, std_err = stats.linregress(mag_vec_temp,interpolation_mass_vec[i](mag_vec_temp))
            contrast_curve_mass=np.append(contrast_curve_mass,slope*new_detec_limits_mag[i]+intercept)
        else :
            contrast_curve_mass=np.append(contrast_curve_mass,interpolation_mass_vec[i](new_detec_limits_mag[i]))

    #companion detected
    if comp:
        f=open(wdir+path_results+"companion_extraction.txt",'r') #produced in the companion_extraction.py routine (must have been used before)
        lines=f.readlines()
        f.close()
        for line in lines:
            if line.strip().split()[0]==filter1+filter2[1]:
                rho=float(line.strip().split()[1])
                erho=float(line.strip().split()[2])
                dmag=float(line.strip().split()[5])
                edmag=float(line.strip().split()[6])
        absolut_mag_comp=dmag+Mag_star-5.*np.log10(distance)+5.
        p_comp=interpolate.interp1d(abs_mag_left,masses*1048)
        comp_in_Massj=p_comp(absolut_mag_comp)
        ecomp_in_Massj=abs(slope*edmag)


    #############################
    #graphs
    #############################
    font = {'family' : 'Bitstream Vera Sans',
            'weight' : 'normal',
            'size'   : 17}
    matplotlib.rc('font', **font)

    plt.close("all")
    plt.figure()
    plt.plot(masses*1048,abs_mag_left,'b',label=filter1)
    plt.plot(masses2*1048,abs_mag_right,'r',label=filter2)
    plt.xlabel('Mass [Mj]')
    plt.ylabel("Mag")
    plt.legend()
    plt.savefig(wdir+path_results+target+"_"+filter1+filter2+"_masse_vs_mag_model.png")

    plt.figure()
    plt.plot(masses*1048,apparent_mag_left,'b',label=filter1)
    plt.plot(masses2*1048,apparent_mag_right,'r',label=filter2)
    plt.xlabel('Mass [Mj]')
    plt.ylabel("mag")
    plt.legend()
    plt.savefig(wdir+path_results+target+"_"+filter1+filter2+"_masse_vs_apparent_mag_model.png")


    params = {'legend.fontsize': 10}
    plt.rcParams.update(params)

    fancy_plot(arcsec_sep,contrast_curve_vec_mass,"Contrast curve sdi+adi",'Angular separation [arcsec]','Mass [Mj]',leg=(masses*1048).tolist(), xlim=[0,0], ylim=[0,0], xscale='std', yscale='std', width=1, line_color='b', style="standard",grid=True,filename_to_save="No_saving")

    fig=plt.figure()
    #for i in range(np.size(masses)):
    fancy_plot(arcsec_sep,contrast_curve_vec_mag,"Contrast curve sdi+adi",'Angular separation [arcsec]',r''+str(5)+'$\sigma$ contrast [mag]',leg=(masses*1048).tolist(), xlim=[0,0], ylim=[0,0], xscale='std', yscale='std', width=1, line_color='r', style="standard",grid=True,filename_to_save="No_saving",text=False,fig=False)
    fancy_plot(new_detec_r_arcsec,new_detec_limits_mag,"Contrast curve sdi+adi",'Angular separation [arcsec]',r''+str(5)+'$\sigma$ contrast [mag]',leg=False, xlim=[0,0], ylim=[0,0], xscale='std', yscale='std', width=1, line_color='b', style="standard",grid=True,filename_to_save="No_saving",text=False,fig=False)
    plt.savefig(wdir+path_results+"contrast_"+target+"_"+filter1+filter2+"_adi_sdi_Mag.png",transparent=False,dpi=300)


    fancy_plot(new_detec_r_arcsec,contrast_curve_mass,"ADI+SDI Detection limit for an age of "+age+"Gyr",'Angular separation [arcsec]',r'Comp. Mass('+str(5)+'$\sigma$) [Mj]',leg=False, xlim=[0,0], ylim=[0,0], xscale='std', yscale='std', width=1, line_color='b', style="standard",grid=True,filename_to_save="No_saving",text=True)
    if comp:
        plt.errorbar(rho,comp_in_Massj,ecomp_in_Massj,erho,'r')
    plt.savefig(wdir+path_results+target+"_"+filter1+filter2+"_adi_sdi_masses.png",dpi=300)

    index_zoom_mass=np.where(contrast_curve_mass<0.5*np.nanmax(contrast_curve_mass))
    if comp:
        index_zoom_sep=np.where(new_detec_r_arcsec[index_zoom_mass]<rho+2)
    else:
		index_zoom_sep=np.where(new_detec_r_arcsec[index_zoom_mass]<3)
    #fig=plt.figure()
    #ax = fig.add_subplot(111)
    #ax.fill_between(np.append(0,new_detec_r_arcsec[index_zoom]), np.zeros(np.size(new_detec_r_arcsec[index_zoom])+1), np.append(13.7,contrast_curve_mass[index_zoom]), facecolor='blue', alpha=1.0)
    fancy_plot(new_detec_r_arcsec[index_zoom_mass[0][0]+index_zoom_sep[0]],contrast_curve_mass[index_zoom_mass[0][0]+index_zoom_sep[0]],"ADI+SDI Detection limit for an age of "+age+"Gyr",'Angular separation [arcsec]',r'Comp. Mass('+str(5)+'$\sigma$) [Mj]',leg=False, xlim=[0,0], ylim=[0,0], xscale='std', yscale='std', width=1, line_color='b', style="standard",grid=True,filename_to_save="No_saving",text=True)
    if comp:
        plt.errorbar(rho,comp_in_Massj,ecomp_in_Massj,erho,'r')
    plt.savefig(wdir+path_results+target+"_"+filter1+filter2+"_adi_sdi_zoom_masses.png",dpi=300)
    #plt.show()


    f=open(wdir+path_results+"mass_mag.txt","w")
    f.write("age = "+age+" [Gyr]\n")
    f.write("Masse\tTeff\t"+filter1+"\t"+filter2+"\n")
    f.write("-------------------------------\n")
    for i in range(np.size(masses)):
        f.write(str(masses[i]))
        f.write("\t")
        f.write(str(Teff[i]))
        f.write("\t")
        f.write(str(abs_mag_left[i]))
        f.write("\t")
        f.write(str(abs_mag_right[i]))
        f.write("\n")
    f.close()
