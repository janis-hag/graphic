# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def read_model(age,M,parallax):
    """
    age = age of the star in Gyr to be given in string with correct writing (for example age=0.120 and not age=0.12)
    M = band filter
    parallax= parallax of the star [mas] in order to compute the visual mag
    """
    path="/home/spectro/peretti/GRAPHIC/version_seb/SPHERE_characterization/contrast_curve_SPHERE/BHAC15/"
    filename="BHAC15_COND03_iso_t10_10.SPHERE.txt"
    f=open(path+filename)
    data=f.readlines()
    f.close()

    if 'D' not in M:
        M='D_'+M



    time='t (Gyr) =   '+age+'\n'
    i=0
    index=0
    for line in data:
        if repr(data[i])==repr(time):
            print(line.replace("\n",""))
            index=i
        if index!=0 and data[i][0]=="\n" and i>index+3:
            index2=i
            break
        i+=1

    table=data[index:index2]

    header=table[2].split()
    i=0
    for col in header:
        #if repr(header[i])==repr(M):
        if M in header[i]:
            filter1=col
            print("filter: ",filter1,"\n")
            index_col=i
        i+=1
    nbr_lines=index2-index
    nbr_col=i

    table2=np.zeros((nbr_lines-3,nbr_col))
    i=0
    for line in table:
        if i>2:
            table2[i-3]=line.split()
        i+=1

    masses=table2[:,0]
    abs_mag=table2[:,index_col]
    Teff=table2[:,1]

    masses=masses
    Teff=Teff
    abs_mag=abs_mag
    print("Masses in model used for sdi self subtraction [Msol]",masses)
    print("Mag corresponding:",abs_mag)

    return masses,Teff,abs_mag
