#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline for
High-contrast Imaging of planetary Companions".
It creates a list containing information on each frame: frame quality,
gaussian fitted position of star, parallactic angle.

The output tables contain the following columns:
frame_number, psf_barycentre_x, psf_barycentre_y, psf_pixel_size,
psf_fit_centre_x, psf_fit_centre_y, psf_fit_height, psf_fit_width_x,
psf_fit_width_y,  frame_number, frame_time, paralactic_angle

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

__version__ = '3.3'
__subversion__ = '0'

import numpy, glob, os, sys, argparse, fnmatch
## pickle, tables, argparse
from mpi4py import MPI
#from gaussfit_nosat import fitgaussian_nosat
#from gaussfit import fitgaussian, i_fitmoffat, moments
import gaussfit_330 as gaussfit
#from scipy import stats
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
#from graphic_mpi_lib_330 import dprint
import numpy as np
from astropy.io import fits as pyfits
import bottleneck
#from scipy import ndimage
import dateutil.parser

nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser(
        description=
        'GRAPHIC:\n The Geneva Reduction and Analysis Pipeline for High-contrast Imaging of planetary Companions.\n\n\
This program creates a list containing information on each frame: frame quality, gaussian fitted position of star, parallactic angle.'
)
parser.add_argument('--debug', action="store", dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern", type=str,
                    required=True, help='Filename pattern')
parser.add_argument('--log_file', action="store", dest="log_file",
                    default='GRAPHIC', help='Log filename')
parser.add_argument(
        '--chuck', action="store", dest="hiciao_filename",
        help='HICIAO fits file to use as reference for chuckcam data')
parser.add_argument(
        '--search_region', action="store", dest="search_region", type=int,
        default=0, help=
        'Set this to a size (in pixels) to cut out a region around the centre of the frame before calculating the centre. Useful to ignore edge of detector artefacts'
)

parser.add_argument(
        '--smooth_width', action="store", dest="smooth_width", type=float,
        default=3, help=
        'The std dev of the Gaussian kernel used to smooth the data before measuring the centre'
)
parser.add_argument('--psf_width', action="store", dest="psf_width", type=float,
                    default=3, help='An initial guess for the psf width')

# No argument options...
parser.add_argument(
        '-saturated', dest='saturated', action='store_const', const=True,
        default=False, help=
        'Use a saturated psf model and fit to the saturation level (in counts)')
parser.add_argument('-once_per_cube', dest='once_per_cube',
                    action='store_const', const=True, default=False,
                    help='Centre only once per cube')

parser.add_argument('-nofit', dest='nofit', action='store_const', const=True,
                    default=False, help='No PSF fitting performed.')
parser.add_argument('-stat', dest='stat', action='store_const', const=True,
                    default=False, help='Print benchmarking statistics')
parser.add_argument('-drh', dest='spherepipe', action='store_const', const=True,
                    default=False,
                    help='Switch for data pre-processed by the SPHERE DRH')
parser.add_argument('-naco', dest='naco', action='store_const', const=True,
                    default=False, help='Switch for NACO data')
parser.add_argument('-sphere', dest='sphere', action='store_const', const=True,
                    default=False, help='Switch for SPHEREdata')
parser.add_argument('-nirc2', dest='nirc2', action='store_const', const=True,
                    default=False, help='Switch for NIRC2 data')
parser.add_argument('-scexao', dest='scexao', action='store_const', const=True,
                    default=False, help='Switch for SCExAO data')
parser.add_argument('-no_psf', action='store_const', dest='no_psf', const=True,
                    default=False,
                    help='Do not look for a PSF, assume it is already centred.')
parser.add_argument(
        '-agpm_centre', action='store_const', dest='agpm_centre', const=True,
        default=False, help=
        'Use the double-Gaussian fit for AGPM coronagraph images. Positive Gaussian for star, negative Gaussian for coro.'
)
parser.add_argument(
        '-remove_striping', action='store_const', dest='remove_striping',
        const=True, default=False, help=
        'Remove the median of each row before calculating the centre position')

args = parser.parse_args()
d = args.d
pattern = args.pattern
spherepipe = args.spherepipe
naco = args.naco
scexao = args.scexao
nirc2 = args.nirc2
log_file = args.log_file
nofit = args.nofit
saturated = args.saturated
no_psf = args.no_psf
hiciao_filename = args.hiciao_filename
agpm_centre = args.agpm_centre
smooth_width = args.smooth_width
psf_width = args.psf_width
once_per_cube = args.once_per_cube
search_region = args.search_region
remove_striping = args.remove_striping

# if moffat:
# header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
# 'frame_num', 'frame_time', 'paralactic_angle']
# else:
header_keys = [
        'frame_number', 'psf_barycentre_x', 'psf_barycentre_y',
        'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y',
        'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y', 'frame_num',
        'frame_time', 'paralactic_angle'
]

if hiciao_filename is None:
    chuck = False
else:
    chuck = True

target_dir = "."
backup_dir = "prev"
positions_dir = "cube-info"
iterations = 1

comments = None

# sys.setrecursionlimit(recurs)
t_init = MPI.Wtime()

# Set the system exception hook to this one, so that if an MPI process fails
# they all exit instead of freezing

sys.excepthook = graphic_mpi_lib.global_except_hook


def fit_frame(centre_est, image, psf_width, saturated):
    cut_size = 16
    refpoint = centre_est - cut_size // 2
    cut_frame = image[refpoint[0]:refpoint[0] + cut_size,
                      refpoint[1]:refpoint[1] + cut_size]
    # Handle the case where the estimate is near the edges of the image
    cut_frame = np.zeros((cut_size, cut_size)) + np.nan
    miny_in = np.max([0, refpoint[0]])
    maxy_in = np.min([image.shape[0], refpoint[0] + cut_size])
    minx_in = np.max([0, refpoint[1]])
    maxx_in = np.min([image.shape[1], refpoint[1] + cut_size])

    miny_out = np.max([0, -refpoint[0]])
    maxy_out = np.min([cut_size, image.shape[0] - (refpoint[0])])
    minx_out = np.max([0, -refpoint[1]])
    maxx_out = np.min([cut_size, image.shape[1] - (refpoint[1])])
    cut_frame[miny_out:maxy_out, minx_out:maxx_out] = image[miny_in:maxy_in,
                                                            minx_in:maxx_in]

    # Fit a Gaussian to it
    fit = gaussfit.psf_gaussfit(cut_frame, width=psf_width, saturated=saturated)
    fit_params = fit.parameters  # (amplitude, x0, y0, sigmax, sigmay, theta)
    # Convert to pixels in original image
    centre_fit = fit_params[2:0:-1] + refpoint

    return centre_fit, fit_params


if rank == 0:  # Master process
    graphic_nompi_lib.print_init()
    # try:
    t0 = MPI.Wtime()
    skipped = 0

    dirlist = glob.glob(pattern + '*.fits')
    dirlist.sort()  # Sort the list alphabetically

    if spherepipe:
        fctable_list = glob.glob(positions_dir + os.sep + 'SPHER*fctable.rdb')
        fctable_list.sort()
    elif chuck:
        # Read the IERS_A table
        iers_a = graphic_nompi_lib.read_iers_a()
        if os.access(hiciao_filename, os.F_OK):
            hdr = pyfits.getheader(hiciao_filename)
        else:
            print(hiciao_filename + ' not found. Interrupting!')
            comm.Abort()
            sys.exit(1)
    elif scexao and not chuck:
        fctable_list = glob.glob(positions_dir + os.sep + 'scexao_parang_*.rdb')
        fctable_list.sort()

    if len(dirlist) == 0:
        print("No files found!")
        comm.Abort()
        ## for n in range(nprocs-1):
        ## comm.send("over", dest = n+1 )
        ## comm.send("over", dest = n+1 )
        sys.exit(1)

    for i in range(len(dirlist)):
        # Read cube header and data
        #header_in=pyfits.getheader(dirlist[i])
        t_cube = MPI.Wtime()
        #check if already processed
        ## if hdf5:
        ## filename='all_info_'+str(thres_coefficient)+"_"+str(min_size)+"_"+str(max_size)+"_"+dirlist[i][:-5]+'.hdf5'
        ## else:
        # filename='all_info_'+str(thres_coefficient)+"_"+str(min_size)+"_"+str(max_size)+"_"+dirlist[i][:-5]+'.rdb'
        if nofit:
            fitname = 'nofit'
        else:
            fitname = 'fit'
        if no_psf:
            psfname = 'nopsf'
        else:
            psfname = 'psf'

        filename = 'all_info_' + str(
                psf_width
        ) + "_" + psfname + '_' + fitname + '_' + dirlist[i][:-5] + '.rdb'

        if os.access(positions_dir + os.sep + filename, os.F_OK):
            print("[" + str(i + 1) + "/" + str(len(dirlist)) + "]: " +
                  filename + " already exists. SKIPPING")
            skipped = skipped + 1
            continue

        print("[" + str(i + 1) + "/" + str(len(dirlist)) + "]: Processing " +
              str(dirlist[i]))

        if not os.access(dirlist[i], os.F_OK):  # Check if file exists

            print("Error: cannot access file " + dirlist[i])

            skipped = skipped + 1
            continue
        else:
            cube, cube_header = pyfits.getdata(dirlist[i], header=True)
            if not chuck:
                hdr = cube_header
            else:  #Creating a header for the empty chuck cam headers
                cube_header['OBS-MOD'] = hdr['OBS-MOD']
                cube_header.comments['OBS-MOD'] = 'Observation mode'
                cube_header['P_TRMODE'] = hdr['P_TRMODE']
                cube_header.comments['P_TRMODE'] = 'Tracking mode of Lyot stop'
                cube_header['DATA-TYP'] = hdr['DATA-TYP']
                cube_header.comments[
                        'DATA-TYP'] = 'Type / Characteristics of this data'
                cube_header['OBJECT'] = hdr['OBJECT']
                cube_header.comments['OBJECT'] = 'Target Description'
                cube_header['RADECSYS'] = hdr['RADECSYS']
                cube_header.comments[
                        'RADECSYS'] = 'The equitorial coordinate system'
                cube_header['RA'] = hdr['RA']
                cube_header.comments['RA'] = 'HH:MM:SS.SSS RA pointing'
                cube_header['DEC'] = hdr['DEC']
                cube_header.comments['DEC'] = '+/-DD:MM:SS.SS DEC pointing'
                cube_header['EQUINOX'] = hdr['EQUINOX']
                cube_header.comments['EQUINOX'] = 'Standard FK5 (years)'
                cube_header['RA2000'] = hdr['RA2000']
                cube_header.comments[
                        'RA2000'] = 'HH:MM:SS.SSS RA (J2000) pointing)'
                cube_header['DEC2000'] = hdr['DEC2000']

                graphic_nompi_lib.save_fits('h_' + dirlist[i], cube,
                                            hdr=cube_header, backend='pyfits',
                                            verify='warn')

        #######
        # Currently crashes if not rdb file found. Should print an error instead and continue.
        ######
        if spherepipe:
            fctable_filename = fnmatch.filter(fctable_list, '*' +
                                              dirlist[i][-40:-10] + '*')[0]
            fctable = graphic_nompi_lib.read_rdb(fctable_filename)
            parang_list = None
            if not 'Angle_deg' in fctable.keys():
                print(
                        str(fctable_filename) +
                        ' does not contain Angle_deg in keys: ' +
                        str(fctable.keys()))
            for i in range(len(fctable['Angle_deg'])):
                jdate = graphic_nompi_lib.datetime2jd(
                        dateutil.parser.parse(fctable['Time-UT'][i]))
                if parang_list is None:
                    parang_list = numpy.array([
                            i, jdate, fctable['Angle_deg'][i]
                    ])
                    ## utcstart=datetime2jd(dateutil.parser.parse(hdr['DATE']+"T"+hdr['UT']))
                else:
                    parang_list = numpy.vstack(
                            (parang_list, [i, jdate, fctable['Angle_deg'][i]]))
        elif 'INSTRUME' in cube_header.keys(
        ) and cube_header['INSTRUME'] == 'SPHERE':
            parang_list = graphic_nompi_lib.create_parang_list_sphere(
                    cube_header)
        elif scexao and not chuck:
            fctable_filename = fnmatch.filter(
                    fctable_list,
                    '*' + dirlist[i].split('_')[-1][:-5] + '.rdb')[0]
            fctable = graphic_nompi_lib.read_rdb(fctable_filename)
            parang_list = np.array([
                    fctable['frame_num'][:], fctable['frame_time'][:],
                    fctable['paralactic_angle'][:]
            ])
            parang_list = (np.rollaxis(parang_list, 1))
        elif chuck:
            ## 'ircam'+string.split(tfile,'ircam')[1]
            #            frame_text_info=string.replace('ircam'+string.split(dirlist[i],'ircam')[1],'fits','txt')
            frame_text_info = 'ircam' + dirlist[i].split('ircam')[1].replace(
                    'fits', 'txt')
            if os.access(frame_text_info, os.F_OK | os.R_OK):
                f = open(frame_text_info)
                timestamps = f.readlines()
                parang_list = graphic_nompi_lib.create_parang_scexao_chuck(
                        timestamps, hdr, iers_a)
            else:
                print('No ' + frame_text_info + ' file found. Skipping ' +
                      dirlist[i])
                continue
        elif naco:
            # Creates a 2D array [frame_number, frame_time, paralactic_angle]
            parang_list = graphic_nompi_lib.create_parang_list_naco(cube_header)
        elif nirc2:
            # Creates a 2D array [frame_number, frame_time, paralactic_angle]
            parang_list = graphic_nompi_lib.create_parang_list_nirc2(
                    cube_header)
        else:
            print('Unknown instrument. Please specify using a the available command switches.'
                  )
            comm.Abort()
            sys.exit(1)

        print('Parang list generated')

        if no_psf:
            print('no psf')
            comm.bcast('over', root=0)
            cent_list = np.ones((cube.shape[0], 9))
            cent_list[:, 0] = np.arange(cube.shape[0])  # Frame number
            cent_list[:, 1] = cube.shape[1] / 2.  # X centre
            cent_list[:, 2] = cube.shape[2] / 2.  # Y centre
            del cube
        else:
            # ACC: Find the rough centre position
            # Do we want to cut out the centre of the image before finding the rough centre position?
            #  For the AGPM, the region outside the mask has a different offset and can cause weird edge effects when smoothing
            mean_cube = np.mean(cube, axis=0)
            if search_region > 1:

                mean_cube = mean_cube[
                        mean_cube.shape[0] // 2 -
                        search_region // 2:mean_cube.shape[0] // 2 +
                        search_region // 2, mean_cube.shape[1] // 2 -
                        search_region // 2:mean_cube.shape[1] // 2 +
                        search_region // 2]

                if remove_striping:
                    for row in mean_cube:
                        row -= np.median(row)
                centre_est = gaussfit.rough_centre(mean_cube,
                                                   smooth_width=smooth_width)
                # And convert to pixels in the original image
                centre_est[0] += cube.shape[1] // 2 - search_region // 2
                centre_est[1] += cube.shape[2] // 2 - search_region // 2
            else:
                centre_est = gaussfit.rough_centre(mean_cube,
                                                   smooth_width=smooth_width)

            if remove_striping:
                for frame in cube:
                    for row in frame:
                        row -= np.median(row)

            if cube.shape[0] == 1:
                centre_fit, fit_params = fit_frame(centre_est, cube[0],
                                                   psf_width, saturated)
                cluster_array_ref = np.array([
                        0, centre_est[0], centre_est[1], 0., centre_fit[0],
                        centre_fit[1], fit_params[0], fit_params[3],
                        fit_params[4]
                ])

                cent_list = [cluster_array_ref]
                comm.bcast('over', root=0)

            else:
                # Send the rough centre to the processes
                comm.bcast(centre_est, root=0)

                # send_frames...
                graphic_mpi_lib.send_frames(cube)
                del cube
                # Prepare the centroid array:
                # [frame_number, psf_barycentre_x, psf_barycentre_y,
                # psf_pixel_size, psf_fit_centre_x, psf_fit_centre_y,
                # psf_fit_height, psf_fit_width_x, psf_fit_width_y]
                cent_list = None

                # Receive data back from slaves
                for n in range(nprocs - 1):
                    data_in = None
                    data_in = comm.recv(source=n + 1)
                    if data_in is None:
                        continue
                    elif cent_list is None:
                        cent_list = data_in.copy()
                    else:
                        cent_list = np.vstack((cent_list, data_in))

        if not os.path.isdir(positions_dir):  # Check if positions dir exists
            os.mkdir(positions_dir)

        if cent_list is None:
            print("No centroids list generated for " + str(dirlist[i]))
            continue

        if d > 2:
            print("parang_list " + str(parang_list.shape) + " : " +
                  str(parang_list))
            print("cent_list " + str(cent_list.shape) + " :" + str(cent_list))

        #Create the final list:
        # [frame_number, psf_barycentre_x, psf_barycentre_y, psf_pixel_size, psf_fit_centre_x, psf_fit_centre_y, psf_fit_height, psf_fit_width_x, psf_fit_width_y  ,  frame_number, frame_time, paralactic_angle]

        cent_list = np.hstack((cent_list, parang_list))

        # Set last frame to invalid if it's the cube-median
        if ('ESO DET NDIT' in cube_header.keys()) and (not no_psf) and (
                cube_header['NAXIS3'] != cube_header['ESO DET NDIT']):
            cent_list[-1] = -1

        # Set first frame to invalid for L_prime band due to cube reset effects
        if not no_psf and 'ESO INS OPTI6 ID' in cube_header.keys(
        ) and cube_header['ESO INS OPTI6 ID'] == 'L_prime':
            cent_list[0] = -1

        if comments is None and not 'ESO ADA PUPILPOS' in cube_header.keys():
            comments = "Warning! No ESO ADA PUPILPOS keyword found. Is it ADI? Using 89.44\n"

        ## if hdf5:
        ## # Open a new empty HDF5 file
        ## f = tables.openFile(positions_dir+os.sep+filename, mode = "w")
        ## # Get the root group
        ## hdfarray = f.createArray(f.root, 'centroids', cent_list, "List of centroids for cube "+str(dirlist[i]))
        ## f.close()
        ## else:
        graphic_nompi_lib.write_array2rdb(positions_dir + os.sep + filename,
                                          cent_list, header_keys)

        if d > 2:
            print("saved cent_list " + str(cent_list.shape) + " :" +
                  str(cent_list))

        sys.stdout.write('\n\n')
        sys.stdout.flush()

        if not no_psf:
            bad = np.where(cent_list[:, 6] == -1)[0]
            print(dirlist[i] + " total frames: " + str(cent_list.shape[0]) +
                  ", rejected: " + str(len(bad)) + " in " +
                  str(MPI.Wtime() - t_cube) + " seconds.")
        del cent_list

        t_cube = MPI.Wtime() - t_cube
        # print(" ETA: "+humanize_time(t_cube*(len(dirlist)-i-1)))
        print(" Remaining time: " +
              graphic_nompi_lib.humanize_time((MPI.Wtime() - t0) *
                                              (len(dirlist) - i - 1) /
                                              (i + 1 - skipped)))

    if len(
            dirlist
    ) == skipped:  # Nothing to be done, so close the slave processes (ACC edit May 2017)
        comm.bcast("over", root=0)
        for n in range(nprocs - 1):
            comm.send("over", dest=n + 1)
            comm.send("over", dest=n + 1)
        MPI.Finalize()
        sys.exit(0)

    print("")
    print(" Total time: " + graphic_nompi_lib.humanize_time(MPI.Wtime() - t0))
    if not len(dirlist) - skipped == 0:
        print(" Average time per cube: " +
              graphic_nompi_lib.humanize_time((MPI.Wtime() - t0) /
                                              (len(dirlist) - skipped)) +
              " = " + str((MPI.Wtime() - t0) /
                          (len(dirlist) - skipped)) + " seconds.")

    ## if 'ESO OBS TARG NAME' in hdr.keys():
    ## log_file=log_file+"_"+string.replace(hdr['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
    ## elif 'OBJECT' in hdr.keys():
    ## log_file=log_file+"_"+string.replace(hdr['OBJECT'],' ','')+"_"+str(__version__)+".log"
    ## else:
    ## log_file=log_file+"_UNKNOW_TARGET_"+str(__version__)+".log"

    # print("the end!")
    graphic_nompi_lib.write_log_hdr((MPI.Wtime() - t_init), log_file, hdr,
                                    comments, nprocs=nprocs)
    ## graphic_nompi_lib.write_log((MPI.Wtime()-t_init), log_file, comments, nprocs=nprocs)
    # Stop slave processes
    comm.bcast("over", root=0)
    for n in range(nprocs - 1):
        comm.send("over", dest=n + 1)
        comm.send("over", dest=n + 1)
    MPI.Finalize()
    sys.exit(0)

#except:
## print "Unexpected error:", sys.exc_info()[0]
## for n in range(nprocs-1):
## comm.send("over", dest = n+1 )
## comm.send("over", dest = n+1 )

## for n in range(nprocs-1):
## check=comm.recv(source = n+1)
## if not check=="OK":
## print("Unexpected reply from slave("+str(n+1)+"). Expected OK, recieved:"+str(check))
## print("Ignoring error.")

## sys.exit(1)

else:  # Slave processes
    ## print(rank)

    # Receive the rough centre position
    centre_est = comm.bcast(None, root=0)

    startframe = comm.recv(source=0)  # get number of first frame
    data_in = comm.recv(source=0)
    if data_in is None:
        print('Rank ' + str(rank) + ': no data received for processing.')
    else:
        print('Rank ' + str(rank) + ': processing.')
    ## print('startframe, data_in'+str(startframe)+', '+str(data_in))
    cube_count = 1
    centre = None
    x0_i = 0
    y0_i = 0

    while not type(data_in) == type("over"):
        if data_in is not None and isinstance(data_in, np.ndarray):

            if agpm_centre:
                # image=np.mean(data_in,axis=0)
                # Two step-centring.
                # 1. Rough centring using peak of smoothed, summed cube
                # centre_est=gaussfit.rough_centre(image)
                # centre_est=np.array([393,437])
                # centre_est=np.array([300,300])

                if once_per_cube:
                    cen_frame = np.mean(data_in, axis=0)
                    #cen_frame - bottleneck.nanmedian(cen_frame)
                    cutsz = 16
                    cen_frame = cen_frame[centre_est[0] -
                                          cutsz // 2:centre_est[0] + cutsz // 2,
                                          centre_est[1] -
                                          cutsz // 2:centre_est[1] + cutsz // 2]
                    # Run the agpm fit
                    fit = gaussfit.agpm_gaussfit(cen_frame)
                    # Now save the results [frame #, rough x cen, rough y cen, psf pixel size?, cen x, cen y, amplitude, x width, y width]
                    star_params = fit.parameters[
                            0:6]  # (amplitude, x0, y0, sigmax, sigmay, theta)
                    agpm_params = fit.parameters[
                            6:]  # (amplitude, x0, y0, sigmax, sigmay, theta)
                    # centre_fit=agpm_params[1:3]+ centre_est - cutsz/2 # the output of agpm_gaussfit is relative to the edge of the cut frame
                    # centre_fit=agpm_params[2:0:-1]+ centre_est - cutsz/2 # the output of agpm_gaussfit is relative to the edge of the cut frame
                    centre_fit = star_params[
                            2:0:
                            -1] + centre_est - cutsz // 2  # the output of agpm_gaussfit is relative to the edge of the cut frame

            for frame in range(data_in.shape[0]):
                sys.stdout.write('\n  [Rank ' + str(rank) + ', cube ' +
                                 str(cube_count) + ']  Frame ' +
                                 str(frame + startframe) + ' of ' +
                                 str(startframe + data_in.shape[0]))
                sys.stdout.flush()

                image = data_in[frame]

                if once_per_cube:
                    cluster_array_ref = np.array([
                            frame + startframe, centre_est[0], centre_est[1],
                            0., centre_fit[0], centre_fit[1], star_params[0],
                            star_params[3], star_params[4]
                    ])

                elif agpm_centre and not (nofit):
                    # # 2. Now use the initial estimate as a starting position for the agpm fit
                    # # Median subtract to make the fitting easier
                    cen_frame = image - bottleneck.nanmedian(image)
                    # Cut out a small region to make the fitting more reliable and fast
                    cutsz = 16
                    cen_frame = cen_frame[centre_est[0] -
                                          cutsz // 2:centre_est[0] + cutsz // 2,
                                          centre_est[1] -
                                          cutsz // 2:centre_est[1] + cutsz // 2]
                    # Run the agpm fit
                    fit = gaussfit.agpm_gaussfit(cen_frame)

                    # Now save the results [frame #, rough x cen, rough y cen, psf pixel size?, cen x, cen y, amplitude, x width, y width]
                    star_params = fit.parameters[
                            0:6]  # (amplitude, x0, y0, sigmax, sigmay, theta)
                    agpm_params = fit.parameters[
                            6:]  # (amplitude, x0, y0, sigmax, sigmay, theta)
                    # centre_fit=agpm_params[1:3]+ centre_est - cutsz/2 # the output of agpm_gaussfit is relative to the edge of the cut frame
                    # centre_fit=agpm_params[2:0:-1]+ centre_est - cutsz/2 # the output of agpm_gaussfit is relative to the edge of the cut frame
                    centre_fit = star_params[
                            2:0:
                            -1] + centre_est - cutsz // 2  # the output of agpm_gaussfit is relative to the edge of the cut frame
                    cluster_array_ref = np.array([
                            frame + startframe, centre_est[0], centre_est[1],
                            0., centre_fit[0], centre_fit[1], star_params[0],
                            star_params[3], star_params[4]
                    ])

                else:
                    # Measure a rough centre position by smoothing the image
                    # and taking the peak (now done outside the loop)
                    # centre_est=gaussfit.rough_centre(image,smooth_width=smooth_width)

                    if not nofit:
                        #                        # Now cut out a small region of the image to make the fitting more reliable and fast
                        #                        cut_size=16
                        #                        refpoint=centre_est-cut_size//2
                        #                        cut_frame=image[refpoint[0]:refpoint[0]+cut_size,refpoint[1]:refpoint[1]+cut_size]
                        #                        # Handle the case where the estimate is near the edges of the image
                        #                        cut_frame = np.zeros((cut_size,cut_size))+np.nan
                        #                        miny_in = np.max([0,refpoint[0]])
                        #                        maxy_in = np.min([image.shape[0],refpoint[0]+cut_size])
                        #                        minx_in = np.max([0,refpoint[1]])
                        #                        maxx_in = np.min([image.shape[1],refpoint[1]+cut_size])
                        #
                        #                        miny_out = np.max([0,-refpoint[0]])
                        #                        maxy_out = np.min([cut_size,image.shape[0]-(refpoint[0])])
                        #                        minx_out = np.max([0,-refpoint[1]])
                        #                        maxx_out = np.min([cut_size,image.shape[1]-(refpoint[1])])
                        #                        cut_frame[miny_out:maxy_out,minx_out:maxx_out] = image[miny_in:maxy_in,minx_in:maxx_in]
                        #
                        #                        #Fit a Gaussian to it
                        #                        fit=gaussfit.psf_gaussfit(cut_frame,width=psf_width,saturated=saturated)
                        #                        fit_params=fit.parameters # (amplitude, x0, y0, sigmax, sigmay, theta)
                        #                        # Convert to pixels in original image
                        #                        centre_fit=fit_params[2:0:-1]+refpoint
                        centre_fit, fit_params = fit_frame(
                                centre_est, image, psf_width, saturated)
                        # Save the params
                        cluster_array_ref = np.array([
                                frame + startframe, centre_est[0],
                                centre_est[1], 0., centre_fit[0], centre_fit[1],
                                fit_params[0], fit_params[3], fit_params[4]
                        ])
                    else:
                        # If we dont want to fit to the image, use the estimated centre and set most params to zero,
                        cluster_array_ref = np.array([
                                frame + startframe, centre_est[0],
                                centre_est[1], 0., centre_est[0], centre_est[1],
                                frame.max(), 0., 0.
                        ])

                # This is the end of the "if agpm_centre" statement

                # if rank==1:
                if centre is None:
                    centre = cluster_array_ref
                else:
                    centre = np.vstack((centre, cluster_array_ref))
            if d > 3:
                print(str(rank) + ": " + str(centre))
            comm.send(centre, dest=0)
        else:
            if d > 3:
                print(str(rank) + ": " + str(centre))
            comm.send(None, dest=0)
        cube_count = cube_count + 1
        centre_est = comm.bcast(None, root=0)
        startframe = comm.recv(source=0)  # get number of first frame
        data_in = comm.recv(source=0)
        centre = None
        x_0 = 0
        y_0 = 0
    else:
        comm.send("OK", dest=0)
        sys.exit(0)
