#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.
First preparation step for ADI reduction.

Its purpose is to subtract the PCA generated sky from each frame .

If you find any bugs or have any suggestions email:
janis.hagelberg@unige.ch
"""

__version__ = '3.3'
__subversion__ = '0'

import glob, os, sys, argparse
import numpy as np
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
import pca
from mpi4py import MPI
## from astropy.io import fits as pyfits
import astropy.io.fits as pyfits
import bottleneck
from scipy.interpolate import interp1d

nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

target_dir = "."
backup_dir = "prev"

parser = argparse.ArgumentParser(
        description='Subtracts the sky on each frame of the cube.')
parser.add_argument('--debug', action="store", dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern", default='*',
                    help='Filename pattern')
parser.add_argument('--sky_pattern', action="store", dest="sky_pattern",
                    help='Sky file pattern')
parser.add_argument('--sky_dir', action="store", dest="sky_dir",
                    default='sky-OB', help='Give alternative sky directory')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",
                    default='all_info', help='Info filename pattern.')
parser.add_argument('--info_dir', action="store", dest="info_dir",
                    default='cube-info', help='Info directory')
parser.add_argument('-noinfo', dest='noinfo', action='store_const', const=True,
                    default=False, help='Do not use PSF fitting values.')
parser.add_argument('-s', dest='stat', action='store_const', const=True,
                    default=False, help='Print benchmarking statistics')
parser.add_argument('--log_file', action="store", dest="log_file",
                    default='GRAPHIC', help='Log filename')
parser.add_argument('-nofit', dest='fit', action='store_const', const=False,
                    default=True, help='Do not use PSF fitting values.')
parser.add_argument('-norm', dest='normalise', action='store_const', const=True,
                    default=False, help='Normalise the sky before subtracting.')
parser.add_argument('-nici', dest='nici', action='store_const', const=True,
                    default=False, help='Switch for GEMINI/NICI data')
## parser.add_argument('-hdf5', dest='hdf5', action='store_const',
## const=True, default=False,
## help='Switch to use HDF5 tables')
parser.add_argument('-interactive', dest='interactive', action='store_const',
                    const=True, default=False,
                    help='Switch to set execution to interactive mode')
parser.add_argument(
        '--flat_filename', dest='flat_filename', action='store', default=None,
        help=
        'Name of flat field to be used. If this argument is not set, the data will not be flat fielded'
)
parser.add_argument(
        '--sky_interp', dest='sky_interp', action='store', type=int, default=1,
        help=
        'Number of sky files to interpolate when doing the sky subtraction. Default is to use 1 file only (no interpolation).'
)
parser.add_argument('-sphere', dest='sphere', action='store_const', const=True,
                    default=False, help='Switch for raw SPHERE data')
parser.add_argument(
        '--pca_modes', dest='pca_modes', action='store', default=None, type=int,
        help=
        'Subtract the sky by PCA, using the principal components calculated on the sky frames to subtract from the data. Set to the number of modes.'
)
parser.add_argument(
        '--star_window', dest='star_window', action='store', default=0,
        type=int, help=
        'Size of the window used to block out the star when calculating the PCA coefficients. Assumes the star is centred in the array.'
)
parser.add_argument(
        '-fix_naco_bad_columns', dest='fix_naco_bad_columns',
        action='store_const', default=False, const=True,
        help='Fix the second quadrant of bad columns on the NACO detector.')

args = parser.parse_args()
d = args.d
pattern = args.pattern
sky_pattern = args.sky_pattern
sky_dir = args.sky_dir
info_pattern = args.info_pattern
info_dir = args.info_dir
stat = args.stat
log_file = args.log_file
fit = args.fit
nici = args.nici
flat_filename = args.flat_filename
sky_interp = args.sky_interp
## hdf5=args.hdf5
sphere = args.sphere
pca_modes = args.pca_modes
star_window = args.star_window
fix_naco_bad_columns = args.fix_naco_bad_columns

header_keys = [
        'frame_number', 'psf_barycentre_x', 'psf_barycentre_y',
        'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y',
        'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y', 'frame_num',
        'frame_time', 'paralactic_angle'
]

skipped = 0
header = None

t_init = MPI.Wtime()

if args.noinfo:
    infolist = None
    cube_list = None

# If a flat field was provided, load it
if flat_filename:
    flat = pyfits.getdata(flat_filename)
else:
    flat = 1.

if rank == 0:
    graphic_nompi_lib.print_init()

    dirlist = graphic_nompi_lib.create_dirlist(pattern)

    if dirlist == None or len(dirlist) < 1:
        print("No files found")
        MPI.Finalize()
        sys.exit(1)

    ## if hdf5:
    ## infolist=glob.glob(info_dir+os.sep+info_pattern+'*.hdf5')
    ## else:
    if not args.noinfo:
        if info_pattern == 'all_info':
            print('Warning, using default value: info_pattern=\"all_info\" wrong info file may be used.'
                  )
        infolist = graphic_nompi_lib.create_dirlist(
                info_dir + os.sep + info_pattern, extension='.rdb')
        cube_list, dirlist = graphic_nompi_lib.create_megatable(
                dirlist, infolist, keys=header_keys, nici=nici, fit=fit,
                sphere=sphere)

    print('Distributing dirlist to slaves.')
    print(rank, nprocs)
    start, dirlist = graphic_mpi_lib.send_dirlist(dirlist)

    comm.bcast(cube_list, root=0)

    skyls = glob.glob(sky_dir + os.sep + sky_pattern + "*.fits")
    skyls.sort()

    sky_array = []

    sky_obstimes = {}

    if len(skyls) < 1:
        print("No sky file found")
        comm.bcast("over", root=0)
        sys.exit(1)
    elif len(skyls) == 1:
        print("Error! Only 1 sky frame found. Need at least 2...")
        comm.bcast("over", root=0)
        sys.exit(1)
    elif len(skyls) < pca_modes:
        print(" Warning! Less sky frames than desired pca modes! Using all available modes: "
              + str(len(skyls)))
        pca_modes = len(skyls)

    if d > 2:
        print(" Using " + str(pca_modes) + " pca modes for sky subtraction")
        print(" With " + str(len(skyls)) + " sky files")

    # Load the data into a single array
    # Loop over all the sky files and make a big array containing all of them.
    for skyfits in skyls:
        sky_data, sky_hdr = pyfits.getdata(skyfits, header=True)

        sky_array.append(sky_data / flat)

    sky_array = np.array(sky_array)

    # PCA code
    # Make the sky cube 1D (since PCA only works in 1D)
    orig_sky_shape = sky_array.shape
    sky_array = np.reshape(sky_array, (orig_sky_shape[0],
                                       orig_sky_shape[1] * orig_sky_shape[2]))

    # Calculate the principal components
    sky_pcomps = pca.principal_components(sky_array, n_modes=pca_modes)

    print("  Max: " + str(np.max(sky_array)) + " Min: " +
          str(np.min(sky_array)))
    print("  Nans: " + str(np.sum(np.isnan(sky_pcomps))))

    comm.bcast(sky_pcomps, root=0)

    # Create directory to store reduced data
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

if not rank == 0:
    dirlist = comm.recv(source=0)
    if dirlist == None:
        sys.exit(0)

    start = int(comm.recv(source=0))

    cube_list = comm.bcast(None, root=0)
    # skylist=comm.bcast(None, root=0)
    # sky_obstimes=comm.bcast(None,root=0)
    sky_pcomps = comm.bcast(None, root=0)

skyfile = None
t0 = MPI.Wtime()
print(dirlist)
# Loop through the files and do the sky subtraction
for i in range(len(dirlist)):
    targetfile = "no" + sky_pattern.replace('_', '') + "_" + dirlist[i]
    if os.access(target_dir + os.sep + targetfile, os.F_OK | os.R_OK):
        print('Already processed: ' + targetfile)
        skipped = skipped + 1
        continue

    ###################################################################
    #
    # Read cube header and data
    #
    ###################################################################

    print(
            str(rank) + ': [' + str(start + i) + '/' +
            str(len(dirlist) + start) + "] " + dirlist[i] +
            " Remaining time: " +
            graphic_nompi_lib.humanize_time((MPI.Wtime() - t0) *
                                            (len(dirlist) - i) /
                                            (i + 1 - skipped)))
    cube, header = pyfits.getdata(dirlist[i], header=True)

    found = False

    if not args.noinfo:
        all_info = cube_list['info'][cube_list['cube_filename'].index(
                dirlist[i])]
    else:
        all_info = 'empty'

    # Flat field
    cube /= flat

    # Make the cube 2D
    orig_cube_shape = cube.shape
    cube = np.reshape(cube, (orig_cube_shape[0],
                             orig_cube_shape[1] * orig_cube_shape[2]))

    # Subtract the PCA modes
    if star_window == 0:
        cube = pca.subtract_principal_components(sky_pcomps, cube)
    else:
        # Generate a mask to choose which pixels are used when calculating the PCA coefficients
        mask = np.zeros(orig_cube_shape[1:], dtype=np.bool) + True
        mask[orig_cube_shape[1] // 2 - star_window:orig_cube_shape[1] // 2 +
             star_window, orig_cube_shape[2] // 2 -
             star_window:orig_cube_shape[2] // 2 + star_window] = False
        # Make it 1D
        mask = np.reshape(mask, (orig_cube_shape[1] * orig_cube_shape[2]))
        cube = pca.subtract_principal_components(sky_pcomps, cube, mask=mask)

    # Make 3D again
    cube = np.reshape(cube, orig_cube_shape)

    # Fix the NACO bad columns if requested
    if fix_naco_bad_columns:
        # Sometimes when cropping the images we might want to centre it away from the centre of the detector
        # This should be saved in the header if this is the case
        if 'GC RM_OVERSCAN_CENTRE_OFFSET_X' in header.keys():
            offset_x = header['HIERARCH GC RM_OVERSCAN_CENTRE_OFFSET_X']
            offset_y = header['HIERARCH GC RM_OVERSCAN_CENTRE_OFFSET_Y']
        else:
            offset_x = 0
            offset_y = 0
        cube = graphic_nompi_lib.fix_naco_second_bad_columns(
                cube, offset=(offset_y, offset_x))

    # Flat field the data if a filename was provided
    if flat_filename:
        header['HIERARCH GC FLAT_FIELD'] = (flat_filename,
                                            'Filename of flat field')

    header['HIERARCH GC PCA_SUB_SKY'] = (__version__ + '.' + __subversion__, '')
    header['HIERARCH GC SKY_PCA_MODES'] = (sky_pcomps.shape[0],
                                           '# of sky pca modes subtracted')

    graphic_nompi_lib.save_fits(
            targetfile, cube.astype(np.float32), hdr=header, backend='pyfits'
    )  # ACC removed verify='warn' because NACO files have a PXSPACE card that is non-standard

if rank == 0:
    if not header == None:
        ## if 'ESO OBS TARG NAME' in header.keys():
        ## log_file=log_file+"_"+string.replace(header['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
        ## else:
        ## log_file=log_file+"_"+string.replace(header['OBJECT'],' ','')+"_"+str(__version__)+".log"
        ## graphic_nompi_lib.write_log((MPI.Wtime()-t_init),log_file)

        graphic_nompi_lib.write_log_hdr((MPI.Wtime() - t_init), log_file,
                                        header, comments=None, nprocs=nprocs)

print(
        str(rank) + ": Total time: " +
        graphic_nompi_lib.humanize_time((MPI.Wtime() - t0)))
sys.exit(0)
