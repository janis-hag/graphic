# Author: Elisabeth Matthews, 2016
#
# This is an excerpt of my main (old) sphereastrometry.py set of functions,
# modified to be useed with GRAPHIC. For now, this just includes one function,
# to convert pixel position angle and separation to on-sky position angle and
# separation.
# Calibrated values for SPHERE from the user manual (version P97.1
# - 2016.01.12 but values still correct in v p.107/108 - 2021-06-29). The citation
# for these values is Maire et al. 2016.

# I define my position angles as ANTI-CLOCKWISE of NORTH

# This code could definitely be better written.

# Note that the PA and SEP values should be provided **WITH THE ERROR ON STAR
# POSITION ALREADY INCORPORATED**

###########################################################

import numpy as np
from astropy.io import fits


def pa_IRDIS_seppa(pa, sep, pscale):
    """
    Take detector position of companion in SEP (pixels) and PA (degrees) and calculate
    on-sky position of companion in SEP (mas) and PA (degrees), or RA (mas) and DEC (mas).

    **NOTE THAT the error on stellar position needs to already be incorporated into the
    error on separation and on position angle!!

    Includes all funny sphere offsets etc.

    syntax:
        pa_IRDIS([280.,0.1], [40.,0.8])
    inputs:
        pa - position angle of target (in degrees)
        sep - separation of target (in pixels)
                sep and pa can be passed as single numbers, or as 2 element arrays,
                of the form [pixel offset, error]
        pscale - platescale to use (currently 'sph_h2', 'sph_k1', 'bb_h' available).
    return arguments:
                **all four return vectors returned as 2 element arrays,
                with the form [value, error]. If no errors passed to function,
                the errors will be returned as zeros.
        PAs - position angle of candidate - in degrees and measured anticlockwise of north
        sep_mas - the separation, in mas, between candidate and host star
        RA - the RA separation, in mas, between candidate and host star
        dec - the dec separation, in mas, between candidate and host star

    use header values and calibration values for VLT/SPHERE to convert from
    pixel separations to on-sky PA(degrees)+separation(mas), or RA(mas)+dec(mas)
    note both PA/sep *and* RA/dec are returned.

    """

    #### HERE are the relevant constants, from the SPHERE User Manual version p100; Maire et al 2016
    truenorth = [-1.75 * np.pi / 180, 0.08 * np.pi / 180]
    ## NOTE: no longer using different TN correction for 2015, since epsilon correction fixes that.
    if pscale == 'sph_h2':
        platescale = [12.255, 0.021]  # H2
    elif pscale == 'sph_k1':
        platescale = [12.267, 0.021]
    elif pscale == 'bb_h':
        platescale = [12.251, 0.021]
    else:
        return ('pscale {} not recognised'.format(pscale))
    pupiloffset = [135.99 * np.pi / 180, 0.11 * np.pi / 180]
    # platescale_ifs = [7.46, 0.02]
    # ifsoffset = [-100.48*np.pi/180, 0.13*np.pi/180]  # eso u.m. 100/p1,p2

    # make sep_in and pa_in two element, float, np.arrays
    sep = np.array(sep)
    pa = np.array(pa)
    sep_me = np.array([0., 0.])
    pa_me = np.array([0., 0.])
    if sep.size == 2:
        sep_me[0] = sep[0]
        sep_me[1] = sep[1]
    elif sep.size == 1:
        sep_me[0] = x
    else:
        return ("sep must have either 1 or 2 elements, see documentation")
    if pa.size == 2:
        pa_me[0] = pa[0]
        pa_me[1] = pa[1]
    elif pa.size == 1:
        pa_me[0] = pa[0]
    else:
        return ("pa must have either 1 or 2 elements, see documentation")

    # convert inputted position angle to radians
    pa_me = pa_me * np.pi / 180

    # these values are the PA_detector and the separation/pix
    sep = sep_me
    pa = pa_me

    # correct for anamorphic distortion - this is done elsewhere, but error is included here
    #y = y*1.0062p/m 0.0002
    anamorphic_error = [1.0062, 0.0002]
    new_platescale_error = np.sqrt((platescale[0] * anamorphic_error[1] /
                                    anamorphic_error[0])**2 + platescale[1]**2)
    platescale[1] = new_platescale_error

    # convert sep/pix to sep/mas
    sep_mas = np.array([0., 0.])
    sep_mas[0] = sep[0] * platescale[0]
    val = (sep[1] / sep[0])**2 + (platescale[1] / platescale[0])**2
    sep_mas[1] = sep_mas[0] * np.sqrt(val)

    # calculate PA_sky using formula on p64 of sphere user manual Jan 2016
    PAs = np.array([0., 0.])
    PAs[0] = pa[0] + truenorth[0] + pupiloffset[0] - (135.87) * np.pi / 180
    # ** note that 135.87 is hardcoded in BOTH arthur's and graphic's routines,
    # because it is the value that was in user manual v4 **
    PAs[1] = np.sqrt(pa[1]**2 + truenorth[1]**2 + pupiloffset[1]**2)

    # calculate the RA and dec using PAs and sep/mas
    RA = np.array([0., 0.])
    RA[0] = sep_mas[0] * np.sin(PAs[0])
    RA[1] = RA[0] * np.sqrt((sep_mas[1] / sep_mas[0])**2 +
                            (PAs[1] / np.tan(PAs[0]))**2)

    dec = np.array([0., 0.])
    dec[0] = sep_mas[0] * np.cos(PAs[0])
    dec[1] = dec[0] * np.sqrt((sep_mas[1] / sep_mas[0])**2 +
                              (PAs[1] * np.tan(PAs[0]))**2)

    #convert returned angles to degrees
    PAs *= 180 / np.pi

    # set the returned error numbers to 0 if an error was not passed
    if sep_me[1] <= 0 or pa_me[1] <= 0:
        sep_mas = [sep_mas[0], 0]
        PAs = [PAs[0], 0]
        RA = [RA[0], 0]
        dec = [dec[0], 0]

    return PAs, sep_mas, RA, dec


if __name__ == "__main__":

    print('HD4113 ep-2016')
    PAs, sep_mas, RA, dec = pa_IRDIS_seppa([39.50, 0.12], [43.66, 0.11],
                                           'sph_h2')
    print(sep_mas, PAs)
    print('-----------------------------------------')
