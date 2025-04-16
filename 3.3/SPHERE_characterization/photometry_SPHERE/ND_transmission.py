import pyfits
import numpy as np
#sys.path.insert(0, '')

def closest(list1, Number):
    aux = np.array([])
    for valor in list1:
        aux=np.append(aux,abs(Number-valor))
    return np.argmin(aux)

def filter_wavelength(filter_given):
    f=open("/home/spectro/peretti/GRAPHIC/version_seb/SPHERE_characterization/photometry_SPHERE/filter_wavelength.dat")
    lines=f.readlines()
    f.close()
    if ("Y2" in filter_given) or ("J2" in filter_given) or ("H2" in filter_given) or ("K1" in filter_given): #in case of sdi
        filter_given=filter_given[:2]
    for line in lines:
        filter1=line.strip().split()[0]
        if filter1==filter_given:
            wavelength=float(line.strip().split()[1])
    return wavelength

def ND_filter_transmission(Neutral_density,filter_given):
    f=open("/home/spectro/peretti/GRAPHIC/version_seb/SPHERE_characterization/photometry_SPHERE/SPHERE_ND_filter_table.dat",'r')
    lines=f.readlines()
    f.close()

    wavelength=filter_wavelength(filter_given)

    iter1=0
    data=np.zeros((np.size(lines)-8,5))
    for line in lines:
        if iter1>7:
            data_line=map(float,line.strip().split())
            data[iter1-8]=data_line
        iter1+=1
    index=closest(data[:,0],wavelength)

    if Neutral_density=='OPEN':
        transmission=data[index,1]
    elif Neutral_density=='ND_1.0':
        transmission=data[index,2]
    elif Neutral_density=='ND_2.0':
        transmission=data[index,3]
    elif Neutral_density=='ND_3.5':
        transmission=data[index,4]
    return transmission
