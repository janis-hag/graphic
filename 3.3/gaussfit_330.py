# 2D gaussian data fitter
# http://www.scipy.org/Cookbook/FittingData

from numpy import *
from scipy import optimize
import numpy as np

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
	print(params)
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
		print(params)
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
