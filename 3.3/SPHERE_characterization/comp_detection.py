import pyfits,glob
import numpy as np
import pylab as py
from scipy import fftpack
from scipy import signal
from scipy.ndimage import interpolation
from kapteyn import kmpfit
#from filtres import *

def low_pass_fft(image,r,d):
    nan_mask=np.where(np.isnan(image),np.nan,1)
    fft=fftpack.fftshift(np.fft.fft2(np.nan_to_num(image)))
    l=np.shape(image)[0]
    x=np.arange(-l/2,l/2)
    y=np.arange(-l/2,l/2)
    X,Y=np.meshgrid(x,y)
    R = np.sqrt(X**2 + Y**2)
    mask=np.ones((l,l))
    mask=np.where(R<r+d/2,1-(1-np.sin(np.pi*(R-r)/d))/2.,mask)
    mask=np.where(R<r-d/2,0,mask)
    """py.figure(10)
    py.imshow(mask)
    py.show()"""

    fft_filter=fft*mask
    #fft_filter=np.where(R<r-d/2,0,fft)

    im_filter=nan_mask*np.real(np.fft.ifft2(fftpack.ifftshift(fft_filter)))
    return im_filter

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    #arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.nanmedian(arr)
    return np.nanmedian(np.abs(arr - med))

def twoD_Gaussian((x,y), amplitude, xo, yo, sigma_x, sigma_y,theta):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

def model((x,y),amplitude,x0,y0,width_x,width_y,theta):#,width_donut,depth_donut,theta):
    Z=np.zeros(((np.shape(x)[0],np.shape(x)[1],2)))
    Z[:,:,0]=x-x0
    Z[:,:,1]=y-y0
    rot=np.zeros((2,2))
    theta_rad=np.pi*theta/180.
    rot[0,0]=np.cos(theta_rad)
    rot[0,1]=np.sin(theta_rad)
    rot[1,0]=-np.sin(theta_rad)
    rot[1,1]=np.cos(theta_rad)
    Z_rot=np.dot(Z,rot)
    x_rot=Z_rot[:,:,0]
    y_rot=Z_rot[:,:,1]
    R=np.sqrt(((1.*x-x0)/width_x)**2+((1.*y-y0)/width_y)**2)
    R_rot=np.sqrt(((1.*x_rot)/width_x)**2+((1.*y_rot)/width_y)**2)
    psf=twoD_Gaussian((x,y), amplitude, x0, y0, width_x, width_y,theta_rad)
    #psf=amplitude*np.where(np.sin(np.pi*R*0.7)/(np.pi*R)<0,5*np.sin(np.pi*R*0.7)/(np.pi*R),np.where(R==0,1,np.sin(np.pi*R*0.7)/(np.pi*R)))
    mask1=np.where(R_rot>3,(1-0.5*(R_rot-3)),1)
    mask1=np.where(mask1<0,0,mask1)
    mask2=np.where(R_rot<1.4,1,0)
    #py.imshow(mask2)
    #py.show()
    psf=psf*mask2+amplitude/2.*np.sin(np.pi*R_rot*0.7)*mask1*(1-mask2)

    return psf

def error(par,data):
    """
    error function for the fit of a moffat profile on a psf in an image. The parameters of the fit are par and the data
    are the image data[0], and the median of the entire image (not calculated here in because we use a sub image to make the fit
    faster)
    """
    im=data[0]
    X2=data[1]
    Y2=data[2]
    med=data[3]
    im=np.where(np.isnan(im),med,im)
    size=np.shape(im)[0]
    Amp=par[0]
    x0=par[1]
    y0=par[2]
    width_x=par[3]
    width_y=par[4]
    theta=par[5]
    #width_donut=par[5]
    #depth_donut=par[6]
    #offset=par[6]
    psf=model((X2,Y2),Amp,x0,y0,width_x,width_y,theta)#,width_donut,depth_donut,theta)
    im_simulated=psf+med
    e=(im_simulated-im)
    return np.asarray(np.real(e)).reshape(-1)


def comp_detection(path,pattern):
    size_box=50
    l2=32
    nbr_sigma_max=2. #defines the number of sigma above which a point source must be over the noise to be seen as a potential point source
    nbr_std=0.25 #factor of improvement of the std so a potential point source is really one

    nbr_sup_mask=0.4 #defines the proportion of the psf that will be masked with nan

    for i,allfiles in enumerate(glob.iglob(pattern+"*")):
        if "FP" not in allfiles:
            image_filename=allfiles
            print "image filename:",image_filename

    im,hdr=pyfits.getdata(path+image_filename,header=True)

    im_low_filter=low_pass_fft(im,150,200)

    #pyfits.writeto(path+"low_pass_"+image_filename,im_low_filter,header=hdr,output_verify="fix+warn",clobber=True)

    l1=np.shape(im_low_filter)[0]
    x1=np.arange(-l1/2,l1/2)
    y1=x1
    X1,Y1=np.meshgrid(x1,y1)
    R1=np.sqrt(X1**2+Y1**2)
    mask_center=np.where(R1<15,np.nan,1)
    im_low_filter=im_low_filter*mask_center

    l=size_box

    x=np.arange(-l/2,l/2)
    y=x
    X,Y=np.meshgrid(x,y)
    x2=np.arange(-l2/2,l2/2)
    y2=x2
    X2,Y2=np.meshgrid(x2,y2)
    R2=np.sqrt(X2**2+Y2**2)

    mod=twoD_Gaussian((X2,Y2), 1, 0, 0, 3, 3,0)
    mask=np.where(R2>2,-1,1)
    mask=np.where(R2>4,0,mask)
    mod=mod*mask

    im_cut=im_low_filter#[724-400:724+400,724-400:724+400]
    #im_cut=im
    im_cut_init=np.copy(im_cut)
    im_cut=signal.correlate2d(im_cut_init,mod,"same")




    nbr_box=2*(np.shape(im_cut)[0]/size_box)-1
    max_index_vec=[]
    comp=0
    shifty=0
    shiftx=0
    for i in range(nbr_box):
        for j in range(nbr_box):
            test=im_cut[i*size_box/2.:i*size_box/2.+size_box,j*size_box/2.:j*size_box/2.+size_box]
            if np.size(np.where(np.isnan(test)==False))>1:
                max_index=np.array(np.where(test==np.nanmax(test)))
                #while np.nanmedian(test[max_index[0]-2:max_index[0]+3,max_index[1]-2:max_index[1]+3])>np.nanmedian(test)+nbr_sigma_max*mad(test):
                while np.nanmedian(im_cut[max_index[0][0]+i*size_box/2-2:max_index[0][0]+i*size_box/2+3,max_index[1][0]+j*size_box/2-2:max_index[1][0]+j*size_box/2+3])>np.nanmedian(test)+nbr_sigma_max*mad(test):
                    max_index_vec.append(np.array([np.array(max_index)[0][0],np.array(max_index)[1][0]])+np.array([i*size_box/2.,j*size_box/2.]))
                    if np.size(np.where(np.sqrt(((max_index_vec-max_index_vec[-1])[:,0])**2+((max_index_vec-max_index_vec[-1])[:,1])**2)<np.sqrt(3**2+3**2)))>1:

                        im_cut[max_index_vec[-1][0]-2:max_index_vec[-1][0]+3,max_index_vec[-1][1]-2:max_index_vec[-1][1]+3]=np.nan #in case it is the same point source as before that was not fitted perfectly
                        max_index_vec=max_index_vec[:np.shape(max_index_vec)[0]-1]
                        #max_index=np.array(np.where(test==np.nanmax(test))) #new companion
                    else:
                        if max_index_vec[-1][0]<l2/2.:
                            if max_index_vec[-1][1]<l2/2.: #corner
                                im_temp_center=im_cut[0:l2,0:l2]
                                center=[l2/2.,l2/2.]
                            elif abs(max_index_vec[-1][1]-np.shape(im_cut)[0])<l2/2.: #other corner
                                im_temp_center=im_cut[0:l2,np.shape(im_cut)[1]-l2:np.shape(im_cut)[1]]
                                center[l2/2.,np.shape(im_cut)[1]-l2/2.]
                            else: #along the side
                                im_temp_center=im_cut[0:l2,max_index_vec[-1][1]-l2/2.:max_index_vec[-1][1]+l2/2.]
                                center=[l2/2.,max_index_vec[-1][1]]
                        elif abs(max_index_vec[-1][0]-np.shape(im_cut)[0])<l2/2.:
                            if max_index_vec[-1][1]<l2/2.:#corner
                                im_temp_center=im_cut[np.shape(im_cut)[0]-l2:np.shape(im_cut)[0],0:l2]
                                center=[np.shape(im_cut)[0]-l2/2.,l2/2.]
                            elif abs(max_index_vec[-1][1]-np.shape(im_cut)[1])<l2/2.: # other corner
                                im_temp_center=im_cut[np.shape(im_cut)[0]-l2:np.shape(im_cut)[0],np.shape(im_cut)[1]-l2:np.shape(im_cut)[1]]
                                center=[np.shape(im_cut)[0]-l2/2.,np.shape(im_cut)[1]-l2/2.]
                            else: #along the side
                                im_temp_center=im_cut[np.shape(im_cut)[0]-l2:np.shape(im_cut)[0],max_index_vec[-1][1]-l2/2.:max_index_vec[-1][1]+l2/2.]
                                center=[np.shape(im_cut)[0]-l2/2.,max_index_vec[-1][1]]
                        elif max_index_vec[-1][1]<l2/2.: #along a side
                            im_temp_center=im_cut[max_index_vec[-1][0]-l2/2.:max_index_vec[-1][0]+l2/2.,0:l2]
                            center=center=[max_index_vec[-1][0],l2/2.]
                        elif abs(max_index_vec[-1][1]-np.shape(im_cut)[1])<l2/2.: #along the last side
                            im_temp_center=im_cut[max_index_vec[-1][0]-l2/2.:max_index_vec[-1][0]+l2/2.,np.shape(im_cut)[1]-l2:np.shape(im_cut)[1]]
                            center=[max_index_vec[-1][0],np.shape(im_cut)[1]-l2/2.]
                        else:
                            im_temp_center=im_cut[max_index_vec[-1][0]-l2/2.:max_index_vec[-1][0]+l2/2.,max_index_vec[-1][1]-l2/2.:max_index_vec[-1][1]+l2/2.]
                            center=[max_index_vec[-1][0],max_index_vec[-1][1]]

                        paramsinitial1=[np.nanmax(im_temp_center),np.where(im_temp_center==np.nanmax(im_temp_center))[1][0]-l2/2.,np.where(im_temp_center==np.nanmax(im_temp_center))[0][0]-l2/2.,2.5,2,10]
                        fitobj1 = kmpfit.Fitter(residuals=error, data=(im_temp_center,X2,Y2,np.nanmedian(im_temp_center)))
                        fitobj1.fit(params0=paramsinitial1)
                        #model_temp_init=model((X2,Y2),paramsinitial1[0],paramsinitial1[1],paramsinitial1[2],paramsinitial1[3],paramsinitial1[4],paramsinitial1[5])
                        model_im_temp=model((X2,Y2),fitobj1.params[0],fitobj1.params[1],fitobj1.params[2],fitobj1.params[3],fitobj1.params[4],fitobj1.params[5])
                        model_im_cut=model((X1,Y1),fitobj1.params[0],center[1]-np.shape(im_cut)[0]/2.+fitobj1.params[1],center[0]-np.shape(im_cut)[1]/2.+fitobj1.params[2],fitobj1.params[3],fitobj1.params[4],fitobj1.params[5])
                        #std_im_minus_model=np.nanstd(im_temp_center-model_im_temp)
                        im_super_cut=im_cut[max_index_vec[-1][0]-8:max_index_vec[-1][0]+8,max_index_vec[-1][1]-8:max_index_vec[-1][1]+8]
                        std_im_minus_model=np.nanstd(im_super_cut-model_im_cut[max_index_vec[-1][0]-8:max_index_vec[-1][0]+8,max_index_vec[-1][1]-8:max_index_vec[-1][1]+8])
                        if std_im_minus_model < np.nanstd(im_super_cut)-nbr_std*np.nanstd(im_super_cut):
                            mask=np.where(np.abs(model_im_cut)>nbr_sup_mask*np.nanmax(np.abs(model_im_cut)),np.nan,1)
                            im_cut[np.where(np.isnan(mask))]=np.nan
                            max_index_vec[-1]=np.array([center[0]+fitobj1.params[2],center[1]+fitobj1.params[1]]) #mise a jour du centre
                            comp+=1
                            print "nbr of companion detected:",comp
                        else:
                            im_cut[max_index_vec[-1][0]-2:max_index_vec[-1][0]+3,max_index_vec[-1][1]-2:max_index_vec[-1][1]+3]=np.nan
                            max_index_vec=max_index_vec[:np.shape(max_index_vec)[0]-1]
                        max_index=np.array(np.where(test==np.nanmax(test))) #new companion
                        if np.size(np.where(np.isnan(test)==False))<1:
                            break



    #pyfits.writeto(path+"compsub_"+image_filename,im_cut,header=hdr,output_verify="fix+warn",clobber=True)
    if comp!=0:
        f=open(path+"ds9_regions.reg","w")
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("# Filename: "+image_filename+"\n")
        f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write('physical\n')
        for k in range(np.shape(max_index_vec)[0]):
            f.write('circle('+str(max_index_vec[k][1]+1)+','+str(max_index_vec[k][0]+1)+',5)\n')
        f.close()
    else:
        print "No companion detected!"
