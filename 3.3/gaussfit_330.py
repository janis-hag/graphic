# 2D gaussian data fitter
# http://www.scipy.org/Cookbook/FittingData

from numpy import *
from scipy import optimize, signal
import numpy as np
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
import scipy.interpolate as interpolate


def gaussian(height, center_x, center_y, width_x, width_y, bg):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    ## print("width y="+str(width_x)+", y="+str(width_y))
    return lambda x, y: bg + height * np.exp(-(((center_x - x) / width_x)**2 + (
            (center_y - y) / width_y)**2) / 2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """

    total = np.sum(data)
    bg = np.median(data)
    X, Y = np.indices(data.shape)
    x = np.sum(X * data) / total
    y = np.sum(Y * data) / total
    col = data[:, np.round(y)]
    width_x = np.sqrt(
            np.abs(np.sum((np.arange(col.size) - y)**2 * col) / np.sum(col)))
    row = data[np.round(x), :]
    width_y = np.sqrt(
            np.abs(np.sum((np.arange(row.size) - x)**2 * row) / np.sum(row)))
    height = data.max()
    ## print("width_y: "+str(width_y)+", data.shape :"+str(data.shape)+", row: "+str(row)+", row.size: "+str(row.size)+", row.sum(): "+str(row.sum())+", abs((arange(row.size)-x)**2*row: "+str(abs((arange(row.size)-x)**2*row)))

    return height, x, y, width_x, width_y, bg


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    ## print("params: "+str(params))
    errorfunction = lambda p: np.ravel(
            gaussian(*p)(*np.indices(data.shape)) - data)
    p, success, infodict, mesg, ier = optimize.leastsq(errorfunction, params,
                                                       full_output=1)
    ## print(p)
    if p[3] < 0. or p[4] < 0.:
        print("Error, negative width: w_x=" + str(p[3]) + " w_y=" + str(p[4]))
        p.fill(-1)
        ## p[3]=np.abs(p[3])
        ## p[4]=np.abs(p[4])
    if not (p[3] / p[4] < 1. / 0.75 and p[3] / p[4] > 0.75):
        print(params, p)
        ## print(p, success, infodict, mesg, ier)
    if ier > 4 or ier == 0:
        print(mesg)
        p.fill(-1)
    return p


def moffat(a, b, x, y):
    """Returns a Moffat function with the given parameters"""
    ## x = float(width_x)
    ## y = float(width_y)
    ## return lambda x,y: height*exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    return lambda x, y: ((b - 1) / (np.pi * (a**2))) * ((1 + (
            (x**2 + y**2) / a**2))**(-b))
    ## I=((b-1)*(np.pi*(a**2))**(-1))*((1+((x**2+y**2)/a**2))**(-b))
    ## return I


def moffat_peak(a, b, x, y):
    """Returns a Moffat function with the given parameters"""
    I = ((b - 1) * (np.pi * (a**2))**(-1)) * ((1 +
                                               ((x**2 + y**2) / a**2))**(-b))
    return I


## def moffat_error(p):
## E=ravel(moffat(*p)(*indices(data.shape)) - data)
## return E
def mofpar(data):
    g_param = fitgaussian(data)
    b = 2.5
    FWHM = (g_param[3] + g_param[4]) / 2.
    a = FWHM / (2 * np.sqrt(2**(1 / b) - 1))
    x = g_param[1]
    y = g_param[2]
    return a, b, x, y


def fitmoffat(data, params=None):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    ## params = [a, b, moments(data)[1,2]]
    if params == None:
        params = mofpar(data)
        ## g_param=fitgaussian(data)
        ## b=2.5
        ## FWHM=(g_param[3]+g_param[4])/2.
        ## a=FWHM/(2*np.sqrt(2**(1/b)-1))
        ## x=g_param[1]
        ## y=g_param[2]
        ## params = [a, b, x, y]
    moffat_error = lambda p: ravel(moffat(*p)(*np.indices(data.shape)) - data)
    p, success, infodict, mesg, ier = optimize.leastsq(moffat_error, params,
                                                       full_output=1)
    #print(p, success, infodict, mesg, ier)
    if ier > 4 or ier == 0:
        print(mesg)
        p.fill(-1)
    return p


def e_moffat(I, x0, y0, bg, b, sX, sY):
    return lambda x, y: bg + (I * ((1 + (x / sX)**2 + (y / sY)**2)**(-b)))


def e_mofpar(data):
    import numpy as np
    g_param = fitgaussian(data)
    b = 2.5
    bg = 0
    I = g_param[0]
    x = g_param[1]
    y = g_param[2]
    sX = g_param[3] / (2 * np.sqrt(2 * np.log(2)))
    sY = g_param[4] / (2 * np.sqrt(2 * np.log(2)))
    FWHM = (g_param[3] + g_param[4]) / 2.
    a = FWHM / (2 * np.sqrt(2**(1 / b) - 1))
    return I, x, y, bg, b, sX, sY


def e_fitmoffat(data, params=None):
    """Returns (x, y, background, peak, beta, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    import numpy as np
    ## params = [a, b, moments(data)[1,2]]
    if params == None:
        params = e_mofpar(data)
        ## g_param=fitgaussian(data)
        ## b=2.5
        ## FWHM=(g_param[3]+g_param[4])/2.
        ## a=FWHM/(2*np.sqrt(2**(1/b)-1))
        ## x=g_param[1]
        ## y=g_param[2]
        ## params = [a, b, x, y]
    # print(params)
    moffat_error = lambda p: ravel(e_moffat(*p)(*indices(data.shape)) - data)
    p, success, infodict, mesg, ier = optimize.leastsq(moffat_error, params,
                                                       full_output=1)
    #print(p, success, infodict, mesg, ier)
    if ier > 4 or ier == 0:
        print(mesg, infodict)
        ## p.fill(-1)
    return p


def i_moffat(I, x0, y0, a, b, bg):
    """Returns a Moffat function with the given parameters"""
    import numpy as np
    ## x = float(width_x)
    ## y = float(width_y)
    ## return lambda x,y: height*exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    return lambda x, y: bg + (I * ((b - 1) / (np.pi * (a**2))) * ((1 + ((
            (x0 - x)**2 + (y0 - y)**2) / a**2))**(-b)))
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
        g_param = fitgaussian(data)
    else:
        g_param = moments(data)
    b = 2.5
    bg = 0
    FWHM = (g_param[3] + g_param[4]) / 2.
    a = FWHM / (2 * np.sqrt(2**(1 / b) - 1))
    I = g_param[0]
    x = g_param[1]
    y = g_param[2]
    return I, x, y, a, b, bg


def i_fitmoffat(data, params=None, gaussfitting=False, full=False):
    """Returns (height, x, y, a, b, bg)
    the gaussian parameters of a 2D distribution found by a fit"""
    import numpy as np
    ## params = [a, b, moments(data)[1,2]]
    if params == None:
        params = i_mofpar(data, gaussfitting)
        # print(params)
        ## g_param=fitgaussian(data)
        ## b=2.5
        ## FWHM=(g_param[3]+g_param[4])/2.
        ## a=FWHM/(2*np.sqrt(2**(1/b)-1))
        ## x=g_param[1]
        ## y=g_param[2]
        ## params = [a, b, x, y]
    moffat_error = lambda p: np.ravel(
            i_moffat(*p)(*np.indices(data.shape)) - data)
    p, success, infodict, mesg, ier = optimize.leastsq(moffat_error, params,
                                                       full_output=1)
    ## print(p, success, infodict, mesg, ier)
    if ier > 4 or ier == 0:
        print("\n ERROR " + str(ier) + "!\n" + mesg)
        #p.fill(-1)
    if full:
        return p, success, infodict, mesg, ier
    else:
        return p


def agpm_gaussfit(image, width=3.5, agpm_width=1.4,
                  agpm_rejection_fraction=0.97, agpm_position='Default'):
    '''Performs a fit to an image slice, using a model with two gaussians. One Gaussian represents
    the star and the other represents the agpm coronagraph, and has a negative amplitude.
    The initi

    data = the input image slice
    width = Initial guess for the psf standard deviation (i.e. sigma for the Gaussian) in pixels
    agpm_width = Initial guess for the with of the Gaussian used to model the AGPM, in pixels
    agpm_rejection_fraction = the fraction of the peak rejected by the agpm. It should be 0.97,
            but experimentally, 0.8 works better for the fit.

    '''

    sz = image.shape[1]

    # Set up the models
    # The first for the star
    star = models.Gaussian2D(amplitude=image.max(), x_mean=sz / 2.,
                             y_mean=sz / 2., x_stddev=width, y_stddev=width)

    # The second for the AGPM
    if type(agpm_position) == type('Default'):
        agpm_position = [sz / 2., sz / 2.]

    agpm = models.Gaussian2D(amplitude=-agpm_rejection_fraction * image.max(),
                             x_mean=agpm_position[0], y_mean=agpm_position[1],
                             x_stddev=agpm_width, y_stddev=agpm_width)
    # agpm.fixed['x_stddev']=True
    # agpm.fixed['y_stddev']=True
    # agpm.bounds['amplitude']=(-2*image.max(),0)
    # agpm.bounds['x_stddev']=[0.5,2.0]
    # agpm.bounds['y_stddev']=[0.5,2.0]

    total_model = star + agpm

    # Set up the fitting
    y, x = np.indices(image.shape)

    # Try fitting with LM
    fitter = LevMarLSQFitter()
    fit = fitter(total_model, x, y, image, acc=1e-4, maxiter=1000)

    return fit


def fixed_agpm_gaussfit(image, width=3.5, agpm_width=1.4,
                        agpm_rejection_fraction=0.97, agpm_position=[0, 0]):
    '''Performs a fit to an image slice, using a model with two gaussians. One Gaussian represents
    the star and the other represents the agpm coronagraph, and has a negative amplitude.
    The initi

    data = the input image slice
    width = Initial guess for the psf standard deviation (i.e. sigma for the Gaussian) in pixels
    agpm_width = Initial guess for the with of the Gaussian used to model the AGPM, in pixels
    agpm_rejection_fraction = the fraction of the peak rejected by the agpm. It should be 0.97,
            but experimentally, 0.8 works better for the fit.

    '''

    sz = image.shape[1]

    # Set up the models
    # The first for the star
    star = models.Gaussian2D(amplitude=1.3 * image.max(),
                             x_mean=agpm_position[0], y_mean=agpm_position[1],
                             x_stddev=width, y_stddev=width)

    # The second for the AGPM
    agpm = models.Gaussian2D(amplitude=-agpm_rejection_fraction * image.max(),
                             x_mean=agpm_position[0], y_mean=agpm_position[1],
                             x_stddev=agpm_width, y_stddev=agpm_width)

    # Force the AGPM to be symmetric
    def tiedfunc(agpm):
        y_stddev_1 = agpm.x_stddev_1
        return y_stddev_1

    agpm.y_stddev.tied = tiedfunc

    # agpm.fixed['x_stddev']=True
    #    agpm.fixed['y_stddev']=True
    agpm.fixed['x_mean'] = True
    agpm.fixed['y_mean'] = True
    agpm.fixed[
            'theta'] = True  # No reason to fit theta to a symmetrical gaussian
    agpm.bounds['x_stddev'] = [0.5, 2.0]
    agpm.bounds['y_stddev'] = [0.5, 2.0]
    agpm.bounds['amplitude'] = [None, 0]
    star.bounds['amplitude'] = [0, None]

    # agpm.bounds['x_mean']=[0.5,image.shape[0]-0.5]
    # agpm.bounds['y_mean']=[0.5,image.shape[0]-0.5]

    total_model = star + agpm

    # Set up the fitting
    y, x = np.indices(image.shape)

    # Try fitting with LM
    fitter = LevMarLSQFitter()
    fit = fitter(total_model, x, y, image, acc=1e-4, maxiter=5000)
    # print 'hack in fixed_agpm_gaussfit'
    # fit = total_model #

    # # Calculate chi2
    # chi2_init = np.sum((total_model(x,y)-image)**2)
    # chi2_final = np.sum((fit(x,y)-image)**2)
    # print 'Chi2',chi2_init/2e5,chi2_final/2e5

    return fit


def psf_gaussfit(image, width=3.5, saturated=True):
    '''Performs a fit to an image slice, using a 2D Gaussian model and the astropy modeling tools.
    Can work for saturated or non-saturated data. Tests showed that the Moffat profile didn't work
    well, but it could be a drop-in replacement for Gaussian2D here.

    data = the input image slice
    width = Initial guess for the psf standard deviation (i.e. sigma for the Gaussian) in pixels
    saturated = Set to True for saturated data, where it will fit a custom model with
        a hard flux cutoff (a parameter it will also try to fit, initialised at the image
        maximum). If False, it will just use the standard Gaussian2D model (much faster)
    '''

    sz = image.shape[1]

    def satpsf_model(x, y, amplitude=1., x_mean=0., y_mean=0., x_stddev=1.,
                     y_stddev=1., theta=0., satpoint=1000.):
        gauss = models.Gaussian2D.evaluate(x, y, amplitude, x_mean, y_mean,
                                           x_stddev, y_stddev, theta)
        gauss = np.where(gauss < satpoint, gauss, satpoint)
        return gauss

    if saturated:
        SatPSF = models.custom_model(satpsf_model)
        satpoint = image.max()
        star = SatPSF(amplitude=image.max(), x_mean=sz / 2., y_mean=sz / 2.,
                      x_stddev=width, y_stddev=width, satpoint=satpoint)
    else:
        satpoint = np.Inf
        star = models.Gaussian2D(amplitude=image.max(), x_mean=sz / 2.,
                                 y_mean=sz / 2., x_stddev=width, y_stddev=width)

    # Add a constant to represent the background level
    const = models.Const2D(amplitude=0)
    total_model = star + const

    # Set up the fitting
    y, x = np.indices(image.shape)

    # Try fitting with LM
    fitter = LevMarLSQFitter()
    fit = fitter(total_model, x, y, image, acc=1e-4, maxiter=1000,
                 estimate_jacobian=True)

    # To keep it compatible with the code before we added the fit to the constant background
    fit.x_mean = fit.x_mean_0
    fit.y_mean = fit.y_mean_0

    return fit


def rough_centre(image, smooth_width=3.):
    '''Calculates a rough image centre by smoothing the image (by convolving with a Gaussian)
    and then taking the location of the peak as the centre.
    data = the input image slice
    smooth_width = the standard deviation of the Gaussian used, in pixels'''

    # Get the x and y coordinates, then make the Gaussian kernel
    ker_sz = np.int(np.round(smooth_width * 4))
    x, y = np.indices((ker_sz, ker_sz), dtype=np.float64) - smooth_width * 2
    ker = np.exp(-(x**2 / (2 * smooth_width**2) + (y**2 /
                                                   (2 * smooth_width**2))))

    # Remove any NaNs
    image = np.nan_to_num(image)

    # Use fftconvolve for the convolution
    smooth_image = signal.fftconvolve(image, ker, mode='same')
    # Take the maximum as the centre
    centre = np.where(smooth_image == smooth_image.max())
    centre = np.array([centre[0][0], centre[1][0]])

    return centre


def correlate_centre(image1, image2, search_size=10, smooth_width=500):
    ''' Calculate the relative shift between two images by the max of their
    cross-correlation.
    It also smooths the images by convolution with a Gaussian, to reduce the
    effects of bad pixels.'''
    image1 = (image1 - np.median(image1)) / np.median(
            np.abs(image1 - np.median(image1)))
    image2 = (image2 - np.median(image2)) / np.median(
            np.abs(image2 - np.median(image2)))
    image2 = image2[::-1, ::-1]

    # I'll do the correlation myself so I can smooth it at the same time
    # Smoothing is equivalent to multiplication by a Gaussian in the Fourier plane

    # Set up the smoothing function
    xc = np.arange(0, image1.shape[-1], dtype=np.float) - image1.shape[-1] // 2
    yc = np.arange(0, image1.shape[-2], dtype=np.float) - image1.shape[-2] // 2
    xcoords, ycoords = np.meshgrid(xc, yc)
    smooth_func = np.exp(-((xcoords**2 + ycoords**2) / (2 * smooth_width**2)))

    # Do the correlation and smoothing at the same time
    ft1 = fft.fft2(image1)
    ft2 = fft.fft2(image2)
    corr = fft.fftshift(fft.ifft2(ft1 * ft2 * fft.fftshift(smooth_func)))
    corr = np.abs(corr)

    # cut out the centre
    corr = corr[corr.shape[0] // 2 - search_size:corr.shape[0] // 2 +
                search_size, corr.shape[1] // 2 -
                search_size:corr.shape[1] // 2 + search_size]

    xc = np.arange(0, corr.shape[-1], dtype=np.float) - corr.shape[-1] // 2 + 1
    yc = np.arange(0, corr.shape[-2], dtype=np.float) - corr.shape[-2] // 2 + 1

    # Find the peak by interpolation
    interp_func = interpolate.interp2d(xc, yc, corr, kind='cubic')

    # Unfortunately interp2d doesn't play nicely with fmin by default.
    def fit_func(x):
        return -interp_func(x[0], x[1])

    centre = optimize.fmin(fit_func, np.array([0, 0]), disp=False)
    return centre


###################
## The following functions are specific to the NACO AGPM, and are used to calculate the centre position of the coronagraph
###################
def agpm_model(params, image_shape=(600, 600)):
    ''' A circular model of the AGPM "big circle" (i.e. the transmissive region of the AGPM substrate).
    params = [xcentre, ycentre, radius]
    Returns an array that has the value True for all pixels inside the circle and False for all pixels outside
    '''
    xx, yy = np.indices(image_shape)
    model = np.sqrt((xx - params[0])**2 + (yy - params[1])**2)
    model = model < params[2]

    return model


def agpm_centre_min_func(params, image_shape=(600, 600), npix_x=0, npix_y=0,
                         use_distance=False):
    '''Function to minimize to calculate the centre of the big circle in the NACO AGPM
    data.
    This uses the parameters to construct a model of the AGPM "big circle". It then counts
    how many pixels are within the circle for each row and column. It returns a goodness-of-fit
    value that represents how different this distribution is compared to the data (i.e. npix_x
    and npix_y)

    Each pixel is weighted by 1-sin(theta)**2 where theta is the angle between the centre of
    the circle and the edge of the circle at that row/column. This is because most of the data
    points are near the centre of the circle, which has very little sensitivity to the movement
    of the agpm. So we want to weight the edge points higher.

    params = [xcentre, ycentre, radius]
    npix_x = number of pixels inside the circle for each column of the data
    npix_y = number of pixels inside the circle for each row of the data

    '''
    model = agpm_model(params, image_shape=image_shape)

    x_model = np.sum(model, axis=1)
    y_model = np.sum(model, axis=0)

    # Throw away points outside of the circle
    # Ideally we would like to use npix_x > 0, but it is noisier
    good_pix_x = x_model > 0
    good_pix_y = y_model > 0

    # Weight the points in the chi2
    # We want higher weights at the edges
    x = np.arange(image_shape[0])
    theta_x = np.arccos((x - params[0]) / params[2])
    weights_x = 1 - np.sin(theta_x)**2  # kind of arbitrary

    y = np.arange(image_shape[1])
    theta_y = np.arccos((y - params[1]) / params[2])
    weights_y = 1 - np.sin(theta_y)**2  # kind of arbitrary

    # Normal chi2
    resids_x = npix_x - x_model
    resids_y = npix_y - y_model
    chi2_x = np.nansum(((resids_x * weights_x)[good_pix_x])**2)
    chi2_y = np.nansum(((resids_y * weights_y)[good_pix_y])**2)

    # Here we could use distance instead of chi2
    # If neither circle is cropped, then for each point on the data, the closest
    #  point on the model is on the line between the centre of the model and the
    #  data point. So the distance between them is (distance from datapoint to
    # the centre of the model circle) - (radius of model circle)
    if use_distance:
        # Distance from each x point to the axis
        dists_x = params[2] - np.sqrt((params[0] - x)**2 + (0.5 * npix_x)**2)
        dists_y = params[2] - np.sqrt((params[0] - y)**2 + (0.5 * npix_y)**2)

        chi2_x = np.nansum(dists_x[good_pix_x])
        chi2_y = np.nansum(dists_y[good_pix_y])

    return chi2_x + chi2_y


def agpm_centre_min_func_1d(params, image_shape=600, npix=0, plot=False):
    '''Function to minimize to calculate the centre of the big circle in the NACO AGPM
    data.
    This uses the parameters to construct a model of the AGPM "big circle". It then counts
    how many pixels are within the circle for each row and column. It returns a goodness-of-fit
    value that represents how different this distribution is compared to the data (i.e. npix_x
    and npix_y)

    Each pixel is weighted by 1-sin(theta)**2 where theta is the angle between the centre of
    the circle and the edge of the circle at that row/column. This is because most of the data
    points are near the centre of the circle, which has very little sensitivity to the movement
    of the agpm. So we want to weight the edge points higher.

    params = [centre, radius]
    npix= number of pixels inside the circle for each column of the data

    '''

    # Weight the points in the chi2
    # We want higher weights at the edges
    x = np.arange(image_shape)
    theta = np.arccos((x - params[0]) / params[1])
    weights = 1 - np.sin(theta)**2  # kind of arbitrary

    # Throw away points outside of the circle
    # These are when the distance from the centre is greater than the AGPM radius
    good_pix = np.abs(x - params[0]) < (2 * params[1])

    # Normal chi2
    # resids = npix - x_model
    # chi2 = np.nansum(((resids*weights)[good_pix])**2)

    # Here we could use distance instead of chi2
    # If neither circle is cropped, then for each point on the data, the closest
    #  point on the model is on the line between the centre of the model and the
    #  data point. So the distance between them is (distance from datapoint to
    # the centre of the model circle) - (radius of model circle)
    # Distance from each x point to the axis
    dists = params[1] - np.sqrt((params[0] - x)**2 + (0.5 * npix)**2)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(0)
        plt.clf()
        plt.plot(dists)
        plt.plot(good_pix * dists)

    chi2 = np.nansum(np.abs(weights * dists[good_pix]))
    if np.sum(good_pix) < (params[1] / 5):
        return np.inf
    return chi2


def pix_inside_big_circle(image):
    ''' Calculate the number of pixels inside the big circle of the NACO AGPM
    as a function of x and y position. Used to calculate the centre of the circle,
    and then the position of the AGPM.
    Pixels inside the circle are detected based on being above a background level.
    The background level is chosen as the value n_sigma from both a region inside
    the circle and a region outside the circle.
    This assumes the top-left corner is outside the circle and the central 200x200
    pixels is inside the circle'''

    central_region = image[image.shape[0] // 2 - 100:image.shape[0] // 2 + 100,
                           image.shape[1] // 2 - 100:image.shape[1] // 2 + 100]
    inside_bckgrd = np.nanmedian(central_region)
    inside_scatter = np.nanmedian(np.abs(central_region - inside_bckgrd))

    corner_region = image[0:50, 0:50]
    outside_bckgrd = np.nanmedian(corner_region)
    outside_scatter = np.nanmedian(np.abs(corner_region - outside_bckgrd))

    # Pick the background as the point n_sigma from both the background and
    # the agpm region
    # This comes from solving:
    ## level = in_bckgrd - n * in_scatter = out_bckgrd + n * out_scatter
    level = (outside_bckgrd * inside_scatter + inside_bckgrd *
             outside_scatter) / (inside_scatter + outside_scatter)

    # above_bckgrd = image > (inside_bckgrd - n_sigma*inside_scatter)
    above_bckgrd = image > level
    npix_x = np.nansum(above_bckgrd, axis=1).astype(np.float)
    npix_y = np.nansum(above_bckgrd, axis=0).astype(np.float)

    #Add NaNs back in
    nans_x = np.sum(np.isnan(image), axis=1) > (image.shape[0] / 2)
    nans_y = np.sum(np.isnan(image), axis=0) > (image.shape[1] / 2)

    npix_x[nans_x] = np.nan
    npix_y[nans_y] = np.nan

    return [npix_x, npix_y]


def fit_to_big_circle(image, use_distance=False, fit_1d=True):
    ''' Perform a fit to calculate the centre of the big circle in an AGPM image'''
    # Find the pixels inside the big circle
    npix_x, npix_y = pix_inside_big_circle(image)

    method = 'Nelder-Mead'

    if fit_1d:
        # 1D method
        initial_guess_x = [image.shape[0] // 2, 289.9]
        result_x = optimize.minimize(agpm_centre_min_func_1d, initial_guess_x,
                                     args=(image.shape[0],
                                           npix_x), tol=1e-4, method=method)
        initial_guess_y = [image.shape[1] // 2, 289.9]
        result_y = optimize.minimize(agpm_centre_min_func_1d, initial_guess_y,
                                     args=(image.shape[1],
                                           npix_y), tol=1e-4, method=method)
        xcen = result_x.x[0]
        ycen = result_y.x[0]
        agpm_rad = (result_x.x[1] + result_y.x[1]) / 2
    else:
        # 2D method:
        initial_guess = [image.shape[0] // 2, image.shape[1] // 2, 290.5]
        # Try fitting
        result = optimize.minimize(
                agpm_centre_min_func, initial_guess,
                args=(image.shape, npix_x, npix_y,
                      use_distance), tol=1e-4, method=method)
        [xcen, ycen, agpm_rad] = result.x

    return [[xcen, ycen], agpm_rad]


###################
