# -*- coding: utf-8 -*-
"""
Estimate the contrast from a GRAPHIC reduction for SPHERE

@author: peretti
"""

import numpy as np
import os


path=os.path.dirname(os.path.realpath(__file__))+"/contrast_curve_SPHERE/"
filename="SPHERE_ND_filter_table.dat"
ND_path_filename=path+filename


def band_filter(band_filter,filename):
    if band_filter=="DB_H23":
        lambda1=1589
        lambda2=1667
    elif band_filter=="DB_K12":
        lambda1=2103
        lambda2=2255
    elif band_filter=="DB_Y23":
        lambda1=1026
        lambda2=1080
    elif band_filter=="DB_J23":
        lambda1=1190
        lambda2=1270
    if "left" in filename:
        return lambda1
    elif "right" in filename:
        return lambda2
    elif "sdi" in filename:
        return lambda2
    else:
        print "Error wavelength not detected because information mission in filename (left, right, or sdi)"
        return 0


def Neutral_density(hdr,wavelength,ND_path_filename=ND_path_filename):
    f=open(ND_path_filename,'r')
    lines=f.readlines()
    f.close()

    Neutral_density=hdr['HIERARCH ESO INS4 FILT2 NAME']

    wavelength_vec=np.zeros(np.size(lines)-7)
    ND_transmission_vec=np.zeros(np.size(lines)-7)
    for i,line in enumerate(lines):
        if i>7:
            wavelength_vec[i-8]=float(line.strip().split()[0])
            if  Neutral_density=='OPEN':
                ND_transmission_vec[i-8]=float(line.strip().split()[1])
            elif Neutral_density=='ND_1.0':
                ND_transmission_vec[i-8]=float(line.strip().split()[2])
            elif Neutral_density=='ND_2.0':
                ND_transmission_vec[i-8]=float(line.strip().split()[3])
            elif Neutral_density=='ND_3.5':
                ND_transmission_vec[i-8]=float(line.strip().split()[4])
            else:
                print "Error Neutral density not understood"
                ND_transmission_vec[i-8]=float(line.strip().split()[4])

    ND_tranmission=ND_transmission_vec[np.where(wavelength_vec==wavelength)][0]
    return ND_tranmission
