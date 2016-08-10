# 2D gaussian data fitter
# http://www.scipy.org/Cookbook/FittingData

from numpy import *
from scipy import optimize,signal
import numpy as np
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
import scipy.interpolate as interpolate

def gaussian(height, center_x, center_y, width_x, width_y, bg):
	"""Returns a gaussian function with the given parameters"""
	width_x = float(width_x)
	width_y = float(width_y)
	## print("width y="+str(width_x)+", y="+str(width_y))
	return lambda x,y: bg+height*np.exp(
				-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
	"""Returns (height, x, y, width_x, width_y)
	the gaussian parameters of a 2D distribution by calculating its
	moments """

	total = np.sum(data)
	bg = np.median(data)
	X, Y = np.indices(data.shape)
	x = np.sum(X*data)/total
	y = np.sum(Y*data)/total
	col = data[:, np.round(y)]
	width_x = np.sqrt(np.abs(np.sum((np.arange(col.size)-y)**2*col)/np.sum(col)))
	row = data[np.round(x), :]
	width_y = np.sqrt(np.abs(np.sum((np.arange(row.size)-x)**2*row)/np.sum(row)))
	height = data.max()
	## print("width_y: "+str(width_y)+", data.shape :"+str(data.shape)+", row: "+str(row)+", row.size: "+str(row.size)+", row.sum(): "+str(row.sum())+", abs((arange(row.size)-x)**2*row: "+str(abs((arange(row.size)-x)**2*row)))

	return height, x, y, width_x, width_y, bg

def fitgaussian(data):
	"""Returns (height, x, y, width_x, width_y)
	the gaussian parameters of a 2D distribution found by a fit"""
	params = moments(data)
	## print("params: "+str(params))
	errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
								 data)
	p, success, infodict, mesg, ier = optimize.leastsq(errorfunction, params, full_output=1)
	## print(p)
	if p[3]<0. or p[4]<0.:
		print("Error, negative width: w_x="+str(p[3])+" w_y="+str(p[4]))
		p.fill(-1)
		## p[3]=np.abs(p[3])
		## p[4]=np.abs(p[4])
	if not (p[3]/p[4] < 1./0.75 and p[3]/p[4] > 0.75):
		print(params, p)
		## print(p, success, infodict, mesg, ier)
	if ier > 4 or ier==0:
		print(mesg)
		p.fill(-1)
	return p


def moffat(a, b, x, y):
	"""Returns a Moffat function with the given parameters"""
	## x = float(width_x)
	## y = float(width_y)
	## return lambda x,y: height*exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
	return lambda x,y :((b-1)/(np.pi*(a**2)))*((1+((x**2+y**2)/a**2))**(-b))
	## I=((b-1)*(np.pi*(a**2))**(-1))*((1+((x**2+y**2)/a**2))**(-b))
	## return I

def moffat_peak(a, b, x, y):
	"""Returns a Moffat function with the given parameters"""
	I=((b-1)*(np.pi*(a**2))**(-1))*((1+((x**2+y**2)/a**2))**(-b))
	return I

## def moffat_error(p):
	## E=ravel(moffat(*p)(*indices(data.shape)) - data)
	## return E
def mofpar(data):
	g_param=fitgaussian(data)
	b=2.5
	FWHM=(g_param[3]+g_param[4])/2.
	a=FWHM/(2*np.sqrt(2**(1/b)-1))
	x=g_param[1]
	y=g_param[2]
	return a, b, x, y

def fitmoffat(data, params=None):
	"""Returns (height, x, y, width_x, width_y)
	the gaussian parameters of a 2D distribution found by a fit"""
	## params = [a, b, moments(data)[1,2]]
	if params==None:
		params=mofpar(data)
		## g_param=fitgaussian(data)
		## b=2.5
		## FWHM=(g_param[3]+g_param[4])/2.
		## a=FWHM/(2*np.sqrt(2**(1/b)-1))
		## x=g_param[1]
		## y=g_param[2]
		## params = [a, b, x, y]
	moffat_error = lambda p: ravel(moffat(*p)(*np.indices(data.shape))-data)
	p, success, infodict, mesg, ier = optimize.leastsq(moffat_error, params, full_output=1)
	#print(p, success, infodict, mesg, ier)
	if ier > 4 or ier==0:
		print(mesg)
		p.fill(-1)
	return p

def e_moffat(I,x0,y0,bg,b,sX,sY):
	return lambda x,y: bg+(I*((1+(x/sX)**2+(y/sY)**2)**(-b)))

def e_mofpar(data):
	import numpy as np
	g_param=fitgaussian(data)
	b=2.5
	bg=0
	I=g_param[0]
	x=g_param[1]
	y=g_param[2]
	sX=g_param[3]/(2*np.sqrt(2*np.log(2)))
	sY=g_param[4]/(2*np.sqrt(2*np.log(2)))
	FWHM=(g_param[3]+g_param[4])/2.
	a=FWHM/(2*np.sqrt(2**(1/b)-1))
	return I, x, y, bg, b, sX, sY

def e_fitmoffat(data, params=None):
	"""Returns (x, y, background, peak, beta, width_x, width_y)
	the gaussian parameters of a 2D distribution found by a fit"""
	import numpy as np
	## params = [a, b, moments(data)[1,2]]
	if params==None:
		params=e_mofpar(data)
		## g_param=fitgaussian(data)
		## b=2.5
		## FWHM=(g_param[3]+g_param[4])/2.
		## a=FWHM/(2*np.sqrt(2**(1/b)-1))
		## x=g_param[1]
		## y=g_param[2]
		## params = [a, b, x, y]
	# print(params)
	moffat_error = lambda p: ravel(e_moffat(*p)(*indices(data.shape))-data)
	p, success, infodict, mesg, ier = optimize.leastsq(moffat_error, params, full_output=1)
	#print(p, success, infodict, mesg, ier)
	if ier > 4 or ier==0:
		print(mesg, infodict)
		## p.fill(-1)
	return p

def i_moffat(I, x0, y0, a, b, bg):
	"""Returns a Moffat function with the given parameters"""
	import numpy as np
	## x = float(width_x)
	## y = float(width_y)
	## return lambda x,y: height*exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
	return lambda x,y: bg+(I*((b-1)/(np.pi*(a**2)))*((1+(((x0-x)**2+(y0-y)**2)/a**2))**(-b)))
	## I=((b-1)*(np.pi*(a**2))**(-1))*((1+((x**2+y**2)/a**2))**(-b))
	## return I

## def i_moffat_peak(I, a, b, x, y):
	## """Returns a Moffat function with the given parameters"""
	## import numpy as np
	## peak=I*((b-1)*(np.pi*(a**2))**(-1))*((1+(((x0-x)**2+(y0-y)**2)/a**2))**(-b))
	## return peak

## def moffat_error(p):
	## E=ravel(moffat(*p)(*indices(data.shape)) - data)
	## return E

def i_mofpar(data, gaussfitting):
	if gaussfitting:
		g_param=fitgaussian(data)
	else:
		g_param=moments(data)
	b=2.5
	bg=0
	FWHM=(g_param[3]+g_param[4])/2.
	a=FWHM/(2*np.sqrt(2**(1/b)-1))
	I=g_param[0]
	x=g_param[1]
	y=g_param[2]
	return I, x, y, a, b, bg

def i_fitmoffat(data, params=None, gaussfitting=False, full=False):
	"""Returns (height, x, y, a, b, bg)
	the gaussian parameters of a 2D distribution found by a fit"""
	import numpy as np
	## params = [a, b, moments(data)[1,2]]
	if params==None:
		params=i_mofpar(data, gaussfitting)
		# print(params)
		## g_param=fitgaussian(data)
		## b=2.5
		## FWHM=(g_param[3]+g_param[4])/2.
		## a=FWHM/(2*np.sqrt(2**(1/b)-1))
		## x=g_param[1]
		## y=g_param[2]
		## params = [a, b, x, y]
	moffat_error = lambda p: np.ravel(i_moffat(*p)(*np.indices(data.shape))-data)
	p, success, infodict, mesg, ier = optimize.leastsq(moffat_error, params, full_output=1)
	## print(p, success, infodict, mesg, ier)
	if ier > 4 or ier==0:
		print("\n ERROR "+str(ier)+"!\n"+mesg)
		#p.fill(-1)
	if full:
		return p, success, infodict, mesg, ier
	else:
		return p

def agpm_gaussfit(image,width=3.5,agpm_width=1.4,agpm_rejection_fraction=0.97):
    '''Performs a fit to an image slice, using a model with two gaussians. One Gaussian represents
    the star and the other represents the agpm coronagraph, and has a negative amplitude.
    The initi

    data = the input image slice
    width = Initial guess for the psf standard deviation (i.e. sigma for the Gaussian) in pixels
    agpm_width = Initial guess for the with of the Gaussian used to model the AGPM, in pixels
    agpm_rejection_fraction = the fraction of the peak rejected by the agpm. It should be 0.97, 
            but experimentally, 0.8 works better for the fit.

    '''

    sz=image.shape[1]
    
    # Set up the models
    # The first for the star
    star=models.Gaussian2D(amplitude=image.max(),x_mean=sz/2.,y_mean=sz/2.,
                x_stddev=width,y_stddev=width)
    
    # The second for the AGPM
    agpm=models.Gaussian2D(amplitude=-agpm_rejection_fraction*image.max(),x_mean=sz/2.,y_mean=sz/2.,
                x_stddev=agpm_width,y_stddev=agpm_width)
    # agpm.fixed['x_stddev']=True
    # agpm.fixed['y_stddev']=True
    # agpm.bounds['amplitude']=(-2*image.max(),0)
    # agpm.bounds['x_stddev']=[0.5,2.0]
    # agpm.bounds['y_stddev']=[0.5,2.0]
        
    total_model=star+agpm
    
    # Set up the fitting
    y,x=np.indices(image.shape)
                
    # Try fitting with LM
    fitter = LevMarLSQFitter()
    fit=fitter(total_model,x,y,image,acc=1e-4,maxiter=1000)
    
    return fit

def psf_gaussfit(image,width=3.5,saturated=True):
    '''Performs a fit to an image slice, using a 2D Gaussian model and the astropy modeling tools.
    Can work for saturated or non-saturated data. Tests showed that the Moffat profile didn't work
    well, but it could be a drop-in replacement for Gaussian2D here.

    data = the input image slice
    width = Initial guess for the psf standard deviation (i.e. sigma for the Gaussian) in pixels
    saturated = Set to True for saturated data, where it will fit a custom model with 
        a hard flux cutoff (a parameter it will also try to fit, initialised at the image
        maximum). If False, it will just use the standard Gaussian2D model (much faster)
    '''

    sz=image.shape[1]

    def satpsf_model(x,y,amplitude=1.,x_mean=0.,y_mean=0.,x_stddev=1.,y_stddev=1.,
                     theta=0.,satpoint=1000.):
        gauss=models.Gaussian2D.evaluate(x,y,amplitude,x_mean,y_mean,x_stddev,y_stddev,theta)
        gauss=np.where(gauss<satpoint,gauss,satpoint)
        return gauss
        
    SatPSF=models.custom_model(satpsf_model)
    if saturated:
        satpoint=image.max()
        star=SatPSF(amplitude=image.max(),x_mean=sz/2.,y_mean=sz/2.,x_stddev=width,
                y_stddev=width,satpoint=satpoint)
    else:
        satpoint=np.Inf
        star=models.Gaussian2D(amplitude=image.max(),x_mean=sz/2.,y_mean=sz/2.,
            x_stddev=width,y_stddev=width)
    
    # Set up the fitting
    y,x=np.indices(image.shape)
                
    # Try fitting with LM
    fitter = LevMarLSQFitter()
    fit=fitter(star,x,y,image,acc=1e-4,maxiter=1000,estimate_jacobian=True)
    
    return fit

def rough_centre(image,smooth_width=3.):
    '''Calculates a rough image centre by smoothing the image (by convolving with a Gaussian)
    and then taking the location of the peak as the centre.
    data = the input image slice
    smooth_width = the standard deviation of the Gaussian used, in pixels'''

    # Get the x and y coordinates, then make the Gaussian kernel
    x,y=np.indices((smooth_width*4,smooth_width*4+1),dtype=np.float64)-smooth_width*2
    ker=np.exp(-(x**2 /(2*smooth_width**2) + (y**2 /(2*smooth_width**2))))
    # Use fftconvolve for the convolution
    smooth_image=signal.fftconvolve(image,ker,mode='same')
    # Take the maximum as the centre
    centre=np.where(smooth_image==smooth_image.max())
    centre=np.array([centre[0][0],centre[1][0]])

    return centre

def correlate_centre(image1,image2,search_size=10,smooth_width=500):
    ''' Calculate the relative shift between two images by the max of their 
    cross-correlation.
    It also smooths the images by convolution with a Gaussian, to reduce the 
    effects of bad pixels.'''
    image1=(image1-np.median(image1))/np.median(np.abs(image1-np.median(image1)))
    image2=(image2-np.median(image2))/np.median(np.abs(image2-np.median(image2)))
    image2=image2[::-1,::-1]
    
    # I'll do the correlation myself so I can smooth it at the same time
    # Smoothing is equivalent to multiplication by a Gaussian in the Fourier plane

    # Set up the smoothing function
    xc=np.arange(0,image1.shape[-1],dtype=np.float)-image1.shape[-1]/2
    yc=np.arange(0,image1.shape[-2],dtype=np.float)-image1.shape[-2]/2
    xcoords,ycoords=np.meshgrid(xc,yc)    
    smooth_func=np.exp(-( (xcoords**2 + ycoords**2)/(2*smooth_width**2) ) )
    
    # Do the correlation and smoothing at the same time
    ft1=fft.fft2(image1)
    ft2=fft.fft2(image2)
    corr=fft.fftshift(fft.ifft2(ft1*ft2*fft.fftshift(smooth_func)))
    corr=np.abs(corr)

    # cut out the centre
    corr=corr[corr.shape[0]/2-search_size:corr.shape[0]/2+search_size,
              corr.shape[1]/2-search_size:corr.shape[1]/2+search_size]
    
    xc=np.arange(0,corr.shape[-1],dtype=np.float)-corr.shape[-1]/2+1
    yc=np.arange(0,corr.shape[-2],dtype=np.float)-corr.shape[-2]/2+1
    
    # Find the peak by interpolation
    interp_func=interpolate.interp2d(xc,yc,corr,kind='cubic')
    # Unfortunately interp2d doesn't play nicely with fmin by default.
    def fit_func(x):
        return -interp_func(x[0],x[1])
    centre=optimize.fmin(fit_func,np.array([0,0]),disp=False)
    return centre

