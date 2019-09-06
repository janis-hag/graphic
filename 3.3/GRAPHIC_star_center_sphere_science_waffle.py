#!/usr/bin/env python3
import numpy as np
import sys
from scipy.optimize import minimize
from scipy import signal
from scipy import ndimage
# from mpi4py import MPI
import graphic_nompi_lib_330 as graphic_nompi_lib
import argparse
import gaussfit_330 as gaussfit
import astropy.io.fits as pyfits
import time

# rank = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Detection of the star centre '
                                 + 'for corono images with the waffle pattern')
parser.add_argument('--pattern', action="store", dest="pattern",
                    default="cl_nomed_SPHER*STAR_CENTER",
                    help='cubes to apply the star centering')
parser.add_argument('-science_waffle', dest='science_waffle',
                    action='store_const', const=True, default=False,
                    help='Switch to science frame with waffle'
                    + ' (usually for high precision astrometry)')
parser.add_argument('-ifs', dest='ifs', action='store_const', const=True,
                    default=False, help='Switch for IFS data')
parser.add_argument('--lowpass_r', dest='lowpass_r', action='store', type=int,
                    default=50, help='Radius of low pass filter to apply'
                    + ' (in pixels) prior to the rough guess of the centre.'
                    + ' Default 50.')
parser.add_argument('-default_rough_centre', dest='default_centre',
                    action='store_const', const=True, default=False,
                    help='Rough position of star behind coronagraph. Use this'
                    + ' to overwrite the automatic rough centring before the'
                    + 'waffle positions are measured.')
parser.add_argument('--manual_rough_centre', dest='manual_rough_centre',
                    action='store', type=int, nargs='+', default=-1,
                    help='Rough position of star behind coronagraph. Use this'
                    + ' to overwrite the rough centring before the waffle'
                    + ' positions are measured.')
parser.add_argument('--ignore_frame', dest='ignore_frame', type=float,
                    default=None,
                    nargs='+', required=False, help='ds9 number of frames for'
                    + ' bad frames to be ignored')
parser.add_argument('-gaussian_fit', dest='gaussian_fit',
                    action='store_const', const=True, default=False,
                    help='Use the psf_gaussfit function from gaussfit to calculate'
                    + ' the centre of each waffle spot instead of a Moffat profile'
                    + ' and the error3 function in graphic_nompi_lib')

args = parser.parse_args()
pattern = args.pattern
science_waffle = args.science_waffle
lowpass_r = args.lowpass_r
default_centre = args.default_centre
manual_rough_centre = args.manual_rough_centre
ignore_frame = args.ignore_frame
gaussian_fit = args.gaussian_fit

rank = 0

if rank == 0:
    if ignore_frame is not None:
        ignore_frame = np.array(ignore_frame)
        bad_frame = True
    else:
        bad_frame = False
        print("No bad frame to ignore")

    def star_centre(key_word, science_waffle=False, ifs=False,
                    lowpass_r=50, manual_rough_centre=-1):
        """
        Determine the star position behind the coronograph using the waffle
        positions and computing a levenberg-marquardt fit of a
        moffat profile on the waffle. We calculate the star position taking
        the gravity centre of the waffles.
        Input:
        im_waffle: image with waffle pattern
        Output:
        creates an asci file with the position of the star in the image for
        the left and right image (the two filters of IRDIS)
        """
        # Extracting the image
        if science_waffle:
            cube_waffle, hdr = pyfits.getdata(key_word, header=True)
            size_cube = np.shape(cube_waffle)[0]
        else:
            count = 0
            # filenames = glob.glob(key_word)
            filenames_sorted = graphic_nompi_lib.create_dirlist(key_word)
            # filenames_sorted = np.sort(filenames)
            if filenames_sorted is None:
                sys.exit(1)
            print(filenames_sorted)
            for allfiles in filenames_sorted:
                if count == 0:
                    cube_waffle, hdr = pyfits.getdata(allfiles, header=True)
                else:
                    temp, hdr = pyfits.getdata(allfiles, header=True)
                    cube_waffle = np.append(cube_waffle, temp, axis=0)
                count += 1
            if bad_frame:
                print('Bad frame to be deleted ' + str(ignore_frame)+'\n')
                cube_waffle = np.delete(cube_waffle, ignore_frame-1, axis=0)
                star_centre_bad_frame_deleted_filename = 'STAR_CENTER_cube_bad_frame_del.fits'
#                pyfits.writeto(star_centre_bad_frame_deleted_filename, cube_waffle, header=hdr, clobber=True)
                graphic_nompi_lib.save_fits(
                        star_centre_bad_frame_deleted_filename , cube_waffle,
                        hdr=hdr,backend='pyfits')
            if cube_waffle.ndim > 2:
                # If it is a cube we take the median over frames
                print('More than one frame found, taking the median ')
                # taking the median
                cube_waffle = np.nanmedian(cube_waffle, axis=0)
            size_cube = 1

        # Work out how many wavelength channels there are
        if ifs:
            if cube_waffle.ndim == 2:
                n_channels = 1
            else:
                n_channels = cube_waffle.shape[0]
        else:
            n_channels = 2
            # Split the IRDIS data into its two channels
            if science_waffle:
                cube_waffle = np.array([cube_waffle[:, :, :1024],
                                        cube_waffle[:, :, 1024:]])
                # It should be nframes x nwav x ny x nx, but it is nwav x nframes x ny x nx
                cube_waffle = np.transpose(cube_waffle,axes=(1,0,2,3))

            else:
                cube_waffle = np.array([cube_waffle[:, :1024],
                                        cube_waffle[:, 1024:]])

        # Make sure cube_waffle is 4D so we can loop over frames and wavelength
        if cube_waffle.ndim == 2:
            cube_waffle = cube_waffle[np.newaxis, np.newaxis, :, :]
        elif cube_waffle.ndim == 3:
            cube_waffle = cube_waffle[np.newaxis, :, :]

        # Loop over frames
        for frame_ix in range(size_cube):

            # Loop over wavelength channels
            for channel_ix in range(n_channels):

                im_waffle = cube_waffle[frame_ix, channel_ix]

                # Rough approximation of the centre by detection of
                # the max after a low pass filter
                #low_pass_im = graphic_nompi_lib.low_pass(im_waffle,
                #                                         lowpass_r, 2, 100)
                low_pass_im = ndimage.filters.median_filter(
                    im_waffle, size=4, mode='reflect')

                if default_centre:
                    # Based on regular 1024x2048 frames
                    # channel 1 : 510, 486
                    # channel 2: 510, 1510
                    default_rough_centre = np.array([510, 486])
                    #default_rough_centre = np.array([486, 510])
                    print('Using default rough centre position: '
                          + str(default_rough_centre[0]) + ' '
                          + str(default_rough_centre[1]))
                    centre = default_rough_centre
                elif manual_rough_centre == -1:
                    # centre=np.array(scipy.ndimage.measurements.center_of_mass(np.nan_to_num(low_pass_im)))
                    centre = np.where(low_pass_im == np.nanmax(low_pass_im))
                    centre = [int(np.round(centre[1][0])),
                              int(np.round(centre[0][0]))]
                else:
                    print('')
                    print('Using manual rough centre position: '
                          + str(manual_rough_centre[0]) + ' '
                          + str(manual_rough_centre[1]))
                    centre = manual_rough_centre

                # Cut the image to find the centre faster
                if ifs:
                    window = 200  # H band will be outside of 128 pix, so take a larger area
                else:
                    window = 192  # 128
#                im = im_waffle[centre[1] - window//2:centre[1] + window//2,
#                               centre[0] - window//2:centre[0] + window//2]
                im = im_waffle[centre[0] - window//2:centre[0] + window//2,
                               centre[1] - window//2:centre[1] + window//2]

                # apply a donut shape mask on the central bright region to
                # find the waffles
                if ifs:
                    # What is the wavelength? We should have stored this during
                    # the re-cubing
                    diff_limit = (hdr['HIERARCH GC WAVELENGTH']*1e-6/8.2)*180/np.pi*3600*1000
                    diff_limit_pix = diff_limit / 7.46 # 7.46 mas/pix is the pixel scale.

                    # they're at about 14.5 lambda/D, so take 2 lambda/D
                    # each side to be safe
                    r_int = 12.5*diff_limit_pix
                    r_ext = 16.5*diff_limit_pix

                elif hdr["HIERARCH ESO INS COMB IFLT"] == 'DB_Y23':
                    r_int = 20
                    r_ext = 60
                elif hdr["HIERARCH ESO INS COMB IFLT"] == 'DB_J23':
                    r_int = 25
                    r_ext = 80
                elif hdr["HIERARCH ESO INS COMB IFLT"] == 'DB_H23':
                    r_int = 35
                    r_ext = 60
                elif hdr["HIERARCH ESO INS COMB IFLT"] == 'DB_K12':
                    r_int = 45
                    r_ext = 80
                else:
                    print('Problem, filter mode not detected!')

                # Set all pixels outside the donut to 0
                x = np.arange(-window/2., window/2.)
                y = np.arange(-window/2., window/2.)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                donut = np.where(R > r_int, 1, 0)
                donut = np.where(R > r_ext, 0, donut)
                im_donut = donut*im

                # rough detection of the waffle positions for the initial guess
                # for the moffat fit with a max detection after a low pass
                # filter.
                # We hide the waffle detected under a mask to detect the next
                # one

                model = graphic_nompi_lib.twoD_Gaussian(X, Y, 1, 0, 0, 3, 3, 0)
                mask = np.where(R > 4, -1, 1)
                mask = np.where(R > 6, 0, mask)

                model = model*mask
                low_pass_im = signal.correlate2d(im_donut, model, "same")

                max_index_vec1 = []
                # Loop over the waffle spots
                # print X.shape,Y.shape,im_donut.shape
                for i in range(4):
                    # This loop records the 4 highest pixel values of the
                    # low_pass filtered image low_pass_im. Each time a spot
                    # is detected it is masked out in order to find the
                    # next highest peak in the following iteration.

                    # Search for the max value pixel
                    max_index = np.array(np.where(
                            low_pass_im == np.nanmax(low_pass_im)))

                    # Take the first one (in case there are multiple peaks
                    # with the same value)
                    # print(max_index, max_index[..., 0])
                    max_index = max_index[..., 0]
                    # We need to add 1 to each direction because the
                    # correlation above shifts the image by 1 pixel
                    max_index += [1, 1]
                    # Store the peak in a new array
                    max_index_vec1 = np.append(max_index_vec1, max_index)

                    # Mask out detected spot for next iteration
                    R = np.sqrt((X + (window/2 - max_index[1]))**2
                                + (Y + (window/2 - max_index[0]))**2)
#                    R = np.sqrt((X + (window/2 - max_index[0]))**2
#                                + (Y + (window/2 - max_index[1]))**2)
                    mask = np.where(R < 15, 0, 1)

                    # !!!!!! Quadrant should be masked out instead!
                    mask = np.ones(low_pass_im.shape)
                    # print(max_index[0], max_index[1], window/2)
                    if max_index[0] > window/2:
                        if max_index[1] > window/2:
                            mask[window//2:, window//2:] = 0
                        else:
                            mask[window//2:, :window//2] = 0
                    else:
                        if max_index[1] > window/2:
                            mask[:window//2, window//2:] = 0
                        else:
                            mask[:window//2, :window//2] = 0

                    low_pass_im = low_pass_im*mask

                # moffat fit with a levenberg-markardt arlgorithm on the
                # waffles to have the best positions
                par_vec = []
                par_init = []
                print('Fitting channel '+str(channel_ix))

                # Turn the area outside of the donut into NaNs to avoid
                # problems with the background
                im_donut[im_donut == 0] = np.NaN

                # Loop over waffle spots
                cutout_sz = 10  # pix, size around waffle to fit
                centre_spot = np.zeros((4, 2))

                for i in range(4):
                    print('\r Waffle '+str(i+1)+' of '+str(4))
                    centre_spot[i] = ([max_index_vec1[2*i],
                                    max_index_vec1[2*i+1]])

                    Prim_x = max_index_vec1[2*i+1]
                    Prim_y = max_index_vec1[2*i]
                    # print(Prim_x, Prim_y)
                    #print(int(Prim_y) - cutout_sz, int(Prim_y) + cutout_sz, int(Prim_x) - cutout_sz, int(Prim_x) + cutout_sz)

                    # cutting the image just around the waffle to make the fit faster
                    y0 = int(Prim_y) - cutout_sz
                    y1 = int(Prim_y) + cutout_sz

                    x0 = int(Prim_x) - cutout_sz
                    x1 = int(Prim_x) + cutout_sz

                    # im_temp = im_donut[y0: y1, x0: x1]
                    im_temp = im_donut[
                            int(centre_spot[i][0]) - cutout_sz:
                            int(centre_spot[i][0]) + cutout_sz,
                            int(centre_spot[i][1]) - cutout_sz:
                            int(centre_spot[i][1]) + cutout_sz]

                    # Prim_x_temp = np.shape(im_temp)[0]/2.
                    # Prim_y_temp = np.shape(im_temp)[0]/2.

                    # Test in which quadrant to initiate the angle
                    # if ((Prim_x < centre[0]) & (Prim_y < centre[1])) or ((Prim_x>centre[0]) & (Prim_y>centre[1])):
                    if (((centre_spot[i][0] < centre[0])
                        & (centre_spot[i][1] < centre[1]))
                        or ((centre_spot[i][0]>centre[0])
                        & (centre_spot[i][1]>centre[1]))):
                            theta_init = -40
                    else:
                            theta_init = 40

                    if gaussian_fit:
                        #Fit a Gaussian to it
                        im_temp = np.nan_to_num(im_temp)
                        fit=gaussfit.psf_gaussfit(im_temp,width=3.5,saturated=False)
                        fit_params=fit.parameters # (amplitude, x0, y0, sigmax, sigmay, theta)
                        # Convert to pixels in original image
                        centre_fit=[fit_params[2] - im_temp.shape[0]/2.+centre_spot[i][0] - im_donut.shape[0]/2. +centre[0],
                                    fit_params[1] - im_temp.shape[1]/2.+centre_spot[i][1] - im_donut.shape[1]/2. +centre[1]]

                        centre_spot[i] = centre_fit
                        correction = [fit_params[2]-im_temp.shape[0]/2.,fit_params[1]-im_temp.shape[1]/2.]

                    else:
                        # Do a Moffat profile fit

                        paramsinitial = [np.nanmedian(im_temp), np.nanmax(im_temp),
                                         0.1, 0.1, 7, 9, 7, theta_init]
                        # fitobj = kmpfit.Fitter(residuals=error, data=(im_temp,med))
                        # fitobj.fit(params0=paramsinitial)
                        res = minimize(graphic_nompi_lib.error3, paramsinitial,
                                       args=(im_temp), method='nelder-mead')
                        # The solution of the optimization:.A1, x01, y01, fwhm1
                        par_vec_temp = np.copy(res.x)[1:4]
                        par_vec_temp[1] = par_vec_temp[1] + cutout_sz#+int(Prim_x)#-10
                        par_vec_temp[2] = par_vec_temp[2] + cutout_sz#+int(Prim_y)#-10
                        par_vec = np.append(par_vec, par_vec_temp)
                        par_init = np.append(par_init, paramsinitial)

                        centre_spot[i] = ([centre[0] + centre_spot[i][0] + par_vec_temp[1] - np.shape(low_pass_im)[0]/2. - np.shape(im_temp)[0]/2.,
                                          centre[1] + centre_spot[i][1] + par_vec_temp[2] - np.shape(low_pass_im)[1]/2. - np.shape(im_temp)[1]/2.])


                        correction = np.array([par_vec_temp[1]-im_temp.shape[0]/2.,
                                               par_vec_temp[2]-im_temp.shape[0]/2.])

                    print('Correction to rough estimate: '+str(correction))


                # Take the mean along the "waffles" axis to have mean_x, mean_y
                star_centre_position = centre_spot.mean(axis=0)

                for s in range(4):
                    print("Spot ", str(s), " centre (x,y): (",
                          centre_spot[s][0], ",", centre_spot[s][1], ")")

                if ifs:
                    channel_name = 'wavelength_'+str(channel_ix)
                else:
                    if channel_ix == 0:
                        channel_name = 'left_im'
                    else:
                        channel_name = 'right_im'

                # !!!File is open for writing during the whole spot search
                # creating and filling the asci file with the star position
                with open('star_center.txt', 'a') as f:
                    f.write(channel_name+'\t' +
                            str(round(star_centre_position[0], 3)) + '\t' +
                            str(round(star_centre_position[1], 3)) + '\n')

                print('Star centre: ', channel_name, star_centre_position)

    # t0 = MPI.Wtime()
    t0 = time.time()
    print("Beginning of star centring")

    f = open('star_center.txt', 'w')
    f.write('image'+'\t'+'x_axis'+'\t'+'y_axis'+'\n')
    f.write('----------------------------'+'\n')
    f.close()

    if science_waffle:
        print("Science waffle frames: finding the centre for each frame in\
              the cubes")
        allfiles = graphic_nompi_lib.create_dirlist(pattern)
        for this_file in allfiles:
            with open('star_center.txt', 'a') as f:
                f.write(this_file+' \n')

            star_centre(this_file, science_waffle=True, ifs=args.ifs,
                        lowpass_r=lowpass_r,
                        manual_rough_centre=manual_rough_centre)
    else:
        star_centre(pattern+'*', ifs=args.ifs, lowpass_r=lowpass_r,
                    manual_rough_centre=manual_rough_centre)

    # print("Total time: " + graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
    print("Total time: " + graphic_nompi_lib.humanize_time((time.time()-t0)))
    print("Star centre detected and file <<star_center.txt>> created.")
    # MPI_Finalize()
    # sys.exit(1)
    # os._exit(1)s
    sys.exit(0)
else:
    # sys.exit(1)
    sys.exit(0)
