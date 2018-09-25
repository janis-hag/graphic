#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline
for High-contrast Imaging of planetary Companions".

Its purpose is to recentre the frames in the cubes, and collapse them to create
smaller cubes with less frames. It is part of the pipeline's quick-look branch.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""
import glob
import os
import sys
import time
import numpy as np
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
from scipy import ndimage
from mpi4py import MPI
import argparse
from graphic_mpi_lib_330 import dprint
import astropy.io.fits as pyfits

__version__ = '3.3'
__subversion__ = '0'

nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD


parser = argparse.ArgumentParser(description='Creates cubes with less frames \
                                 by median-combining frames.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern",  default='*',
                    help='Filename pattern')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",
                    required=True, help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",
                    default='cube-info', help='Info directory')
parser.add_argument('--log_file', action="store", dest="log_file",
                    default='GRAPHIC', help='Log filename')
parser.add_argument('--naxis3', action="store", dest="naxis3", default=4,
                    type=int,
                    help='The number of files to use to generate one cube')
parser.add_argument('--lmax', action="store", dest="l_max", type=int,
                    default=0, help='Shape of the final image. If not \
                    specified it will be calculated to fit all the images.')

parser.add_argument('-nofft', dest='nofft', action='store_const',
                    const=True, default=False,
                    help='Use interpolation instead of Fourier shift')
parser.add_argument('-nofit', dest='fit', action='store_const',
                    const=False, default=True,
                    help='Do not use PSF fitting values.')
parser.add_argument('-nocollapse', dest='collapse', action='store_const',
                    const=False, default=True,
                    help='Do not collapse cubes.')
parser.add_argument('-combine_frames', dest='combine_frames',
                    action='store_const', const=True, default=False,
                    help='Mean combine the frames so that the output is a \
                    single 2D frame.')

parser.add_argument('-s', dest='stat', action='store_const',
                    const=True, default=False,
                    help='Print benchmarking statistics')
parser.add_argument('-nici', dest='nici', action='store_const',
                    const=True, default=False,
                    help='Switch for GEMINI/NICI data')
parser.add_argument('-bottleneck', dest='use_bottleneck', action='store_const',
                    const=True, default=False, help='Use bottleneck module \
                    instead of numpy for nanmedian.')
parser.add_argument('-update_crpix', dest='update_crpix', action='store_const',
                    const=True, default=False,
                    help='Update the CRPIX values from the header')
parser.add_argument('--output_n_frames', dest='output_n_frames',
                    action='store', default=0, type=int, help="Combine frames \
                    in the input cubes to give this many frames per cube in \
                    the output. Default (0) takes all frames")

args = parser.parse_args()
d = args.d
pattern = args.pattern
info_pattern = args.info_pattern
info_dir = args.info_dir
log_file = args.log_file
nofft = args.nofft
fit = args.fit
collapse = args.collapse
nici = args.nici
naxis3 = args.naxis3
use_bottleneck = args.use_bottleneck
l_max = args.l_max
update_crpix = args.update_crpix
output_n_frames = args.output_n_frames

if use_bottleneck:
    from bottleneck import median as median
    from bottleneck import nanmedian as nanmedian
else:
    from numpy import nanmedian
    from numpy import median as median

header_keys = ['frame_number', 'psf_barycentre_x', 'psf_barycentre_y',
               'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y',
               'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
               'frame_num', 'frame_time', 'paralactic_angle']

target_dir = "."
if collapse:
    target_pattern = "rc"
else:
    target_pattern = "r"

# Save the requested output size so we can crop the images later
l_max_in = l_max


def read_recentre_cube(rcn, cube, rcube_list, l_max):
    t0_trans = MPI.Wtime()
    # Send the cube to be recentreed
    comm.bcast("recentre", root=0)
    comm.bcast(l_max, root=0)
    comm.bcast(rcube_list['info'][rcn], root=0)
    graphic_mpi_lib.send_frames_async(cube)

    cube = None
    if args.stat is True:
        print("\n STAT: Data upload took: "
              + graphic_nompi_lib.humanize_time(MPI.Wtime()-t0_trans))
        t0_trans = MPI.Wtime()

    # Recover data from slaves
    for n in range(nprocs-1):
        data_in = comm.recv(source=n+1)
        if data_in is None:
            continue
        sys.stdout.write('\r\r\r Recentreed data from '+str(n+1)+' received                                        = >')
        sys.stdout.flush()

        if cube is None:
            cube = data_in
        else:
            cube = np.concatenate((cube, data_in))
    return cube, t0_trans


t_init = MPI.Wtime()

if rank == 0:
    print(sys.argv[0]+' started on ' + time.strftime("%c"))
    hdr = None

    dirlist = graphic_nompi_lib.create_dirlist(pattern)

    infolist = glob.glob(info_dir+os.sep+'*'+info_pattern+'*.rdb')
    infolist.sort()  # Sort the list alphabetically


    cube_list,dirlist = graphic_nompi_lib.create_megatable(dirlist,infolist,keys = header_keys,nici = nici,fit = fit)

    skipped = 0

    # 0: frame_number, 1: psf_barycentre_x, 2: psf_barycentre_y, 3: psf_pixel_size,
    # 4: psf_fit_centre_x, 5: psf_fit_centre_y, 6: psf_fit_height,
    # 7: psf_fit_width_x, 8: psf_fit_width_y,
    # 9: frame_number, 10: frame_time, 11: paralactic_angle
    # l_max = 0

    # Loop over the output cubes
    for c in range(0, len(cube_list['cube_filename']), naxis3):
        t0_cube = MPI.Wtime()
        if nici: # Convert JD to unix time to get rid of day change issues
            cube_list['info'][c][:, 10] = (cube_list['info'][c][:, 10]-2440587.5)*86400

        psf_sub_filename = target_pattern+"_"+cube_list['cube_filename'][c]
        info_filename = "all_info_"+psf_sub_filename[:-5]+".rdb"

        # Check if already processed
        if os.access(target_dir+os.sep+psf_sub_filename, os.F_OK | os.R_OK):
            print('Already processed: '+psf_sub_filename)
            skipped = skipped+1
            continue
        # Check if already processed
        elif os.access(target_dir +os.sep + psf_sub_filename+'.EMPTY',
                       os.F_OK | os.R_OK):
            print('Already processed, but no cube created: ' + psf_sub_filename)
            skipped = skipped+1
            continue

        new_cube = None
        new_info = None
        # Loop over the files we need to bin together for this output cube
        for n in range(naxis3):
            if c+n == len(cube_list['cube_filename']):
                break
            print("Processing cube [" + str(c+n+1) + "/" + str(len(cube_list['cube_filename'])) + "]: " + str(cube_list['cube_filename'][c]))

            cube, hdr = pyfits.getdata(cube_list['cube_filename'][c+n],
                                       header=True)

            p0 = float(cube_list['info'][c+n][1][11])

            input_cube_shape = cube.shape

            if l_max == 0:
                for i in range(len(cube_list['info'])):
                    if not cube_list['info'][i][
                            len(cube_list['info'][i])//2, 4] == -1:
                        length = 2*graphic_nompi_lib.get_max_dim(
                                cube_list['info'][i][len(cube_list['info'][i])//2, 4],
                                cube_list['info'][i][len(cube_list['info'][i])//2, 5],
                                int(hdr['NAXIS1']), cube_list['info'][i][:, 11]-p0)
                        if length > l_max:
                            l_max = length

                dprint(d > 1, 'l_max: ' + str(l_max))
                l_max = np.int(np.floor(l_max))
            if l_max == 0:
                l_max = 2*int(hdr['NAXIS1'])

            # If l_max is too small, increase it now
            max_dim = np.max([hdr['NAXIS1'], hdr['NAXIS2']])
            if l_max < max_dim:

                # Find the maximum amount of pixels that we need to shift the
                # image. Then we can make the output image big enough to store
                positions = [(this_cube_list[:, 4:6]) for this_cube_list in cube_list['info']]
                positions = np.array(positions)
                positions = positions[positions != -1]
                positions = np.abs(positions - max_dim/2)

                l_max = np.int(max_dim + np.max(positions))

            cube, t0_trans = read_recentre_cube(c+n, cube, cube_list, l_max)

            # Cut the cube down to the requested size
            cube = cube[:, int(cube.shape[1]//2-l_max_in//2):
                int(cube.shape[1]//2+l_max_in//2),
                int(cube.shape[2]//2-l_max_in//2):
                int(cube.shape[2]//2+l_max_in//2)]

            if collapse:
                if new_cube is None:
                    new_cube = np.ones((naxis3, cube.shape[1], cube.shape[2]))
                    new_info = np.NaN*np.ones((naxis3, len(cube_list['info'][c+n][0])))
                new_cube[n] = nanmedian(cube, axis=0)
                new_info[n] = nanmedian(np.where(cube_list['info'][c+n] == -1,
                        np.NaN, cube_list['info'][c+n]), axis=0)
                new_info[n] = np.where(np.isnan(new_info[n]), -1, new_info[n])
            else:
                psf_sub_filename = target_pattern + "_" + cube_list['cube_filename'][c+n]
                hdr['HIERARCH GC RECENTER'] = (str(__version__) + '.' + (__subversion__), "")
                hdr['HIERARCH GC LMAX'] = (l_max,"")
                ## hdr['CRPIX1'] = ('{0:14.7G}'.format(cube.shape[1]/2.+hdr['CRPIX1']-cube_list['info'][c+n][hdr['NAXIS3']/2,4]), "")
                ## hdr['CRPIX2'] = ('{0:14.7G}'.format(cube.shape[2]/2.+hdr['CRPIX2']-cube_list['info'][c+n][hdr['NAXIS3']/2,5]), "")
                hdr['CRPIX1'] = ('{0:14.7G}'.format(cube.shape[1]/2.+np.float(hdr['CRPIX1'])-median(cube_list['info'][c+n][np.where(cube_list['info'][c+n][:,4]>0),4])), "")
                hdr['CRPIX2'] = ('{0:14.7G}'.format(cube.shape[2]/2.+np.float(hdr['CRPIX2'])-median(cube_list['info'][c+n][np.where(cube_list['info'][c+n][:,5]>0),5])), "")

                hdr['history'] = 'Updated CRPIX1, CRPIX2'

                if output_n_frames != 0:
                    save_cube = np.zeros((output_n_frames,
                                          cube.shape[1], cube.shape[2]))

                    # Make some arrays to store some info for the output info files
                    fit_heights = []
                    frame_times = []
                    paralactic_angles = []
                    psf_fit_width_xs = []
                    psf_fit_width_ys = []

                    # Turn the bad entries into NaNs temporarily
                    cube_list['info'][c+n][(cube_list['info'][c+n]) == -1] = np.nan

                    # Divide the input cube into roughly equal chunks to be combined
                    end_ix = 0
                    frames_left = cube.shape[0]
                    for ix in range(output_n_frames):

                        start_ix = end_ix  # Start where we left off last loop
                        end_ix = start_ix + np.round(float(frames_left)/(output_n_frames-ix)).astype(int)
                        frames_left -= (end_ix-start_ix)

                        # Take the median of the frames to produce the new output frame
                        save_cube[ix] = nanmedian(cube[start_ix:end_ix], axis=0)

                        # Some info for the info files
                        fit_heights.append(nanmedian(cube_list['info'][c+n][start_ix:end_ix,6]))
                        frame_times.append(nanmedian(cube_list['info'][c+n][start_ix:end_ix,10]))
                        paralactic_angles.append(nanmedian(cube_list['info'][c+n][start_ix:end_ix,11]))
                        psf_fit_width_xs.append(nanmedian(cube_list['info'][c+n][start_ix:end_ix,7]))
                        psf_fit_width_ys.append(nanmedian(cube_list['info'][c+n][start_ix:end_ix,8]))

                    # And fix the cube_list file
                    new_info = np.zeros((output_n_frames,len(header_keys)))
                    new_info[:,0] = range(output_n_frames) # Frame number
                    new_info[:,9] = range(output_n_frames) #Frame Num
                    new_info[:,1] = np.repeat(cube.shape[1]/2,output_n_frames) # PSF Barycentre X (Since it has been recentred)
                    new_info[:,2] = np.repeat(cube.shape[1]/2,output_n_frames) # PSF Barycentre Y
                    new_info[:,4] = np.repeat(cube.shape[1]/2,output_n_frames) # PSF Fit centre X
                    new_info[:,5] = np.repeat(cube.shape[1]/2,output_n_frames) # PSF Fit centre Y
                    new_info[:,3] = np.repeat(0,output_n_frames) # PSF pixel size (?)
                    new_info[:,6] = fit_heights # PSF fit height
                    new_info[:,11] = paralactic_angles # parallactic angle
                    new_info[:,7] = psf_fit_width_xs # PSF fit width X
                    new_info[:,8] = psf_fit_width_ys # PSF fit width Y
                    new_info[:,10] = frame_times # Frame time

                    # Turn the bad entries back into -1
                    new_info[np.isnan(new_info)] = -1

                    # Now replace the old info array with the new one
                    cube_list['info'][c+n] = new_info
                    cube = save_cube

                    # Add some info to the header
                    hdr['HIERARCH GC INPUT_N_FRAMES'] = (input_cube_shape[0],
                       "Number of original frames in cube")
                    hdr['HIERARCH GC OUTPUT_N_FRAMES'] = (output_n_frames,
                       "Number of output frames requested")

                # Remove the bad frames that contain only NaNs
                bad_frames = (np.sum(np.isnan(cube) is False,axis=(1,2)) == 0)
                cube = cube[bad_frames is False]
                cube_list['info'][c+n] = cube_list['info'][c+n][bad_frames == False]
                cube_list['info'][c+n][:, 0] = range(cube.shape[0])
                cube_list['info'][c+n][:, 9] = range(cube.shape[0])
                hdr['HIERARCH GC NAN_FRAMES_REMOVED'] = (np.sum(bad_frames),
                   "NaN-only frames removed during recentring")

                # Convert it to 32bit float to save storage space
                # (we dont need 64 bit)
                cube = cube.astype(np.float32)

                # Only save if there are some frames to save:
                n_good_frames = np.sum(bad_frames is False)
                if n_good_frames == 0:
                    psf_sub_filename = psf_sub_filename+'.EMPTY'
                    open(psf_sub_filename, 'a').close()
                else:
                    graphic_nompi_lib.save_fits(psf_sub_filename, cube,
                                                target_dir=target_dir,
                                                hdr=hdr, backend='pyfits')
                    graphic_nompi_lib.write_array2rdb(
                            info_dir + os.sep + info_filename,
                            cube_list['info'][c+n], header_keys)

        if collapse:
            hdr['HIERARCH GC RECENTER'] = (str(__version__)+'.'+(__subversion__), "")
            hdr['HIERARCH GC LMAX'] = (l_max, "")
            ## hdr['CRPIX1'] = ('{0:14.7G}'.format(cube.shape[1]/2.+hdr['CRPIX1']-cube_list['info'][c][hdr['NAXIS3']/2,4]), "")
            ## hdr['CRPIX2'] = ('{0:14.7G}'.format(cube.shape[2]/2.+hdr['CRPIX2']-cube_list['info'][c][hdr['NAXIS3']/2,5]), "")

            ## ACC made the following 3 lines an option because they were not working correctly for NACO, and the header values arent used elsewhere.
            if update_crpix:
                hdr['CRPIX1'] = ('{0:14.7G}'.format(cube.shape[1]/2.+np.float(hdr['CRPIX1'])-median(cube_list['info'][c+n][np.where(cube_list['info'][c+n][:,4]>0),4])), "")
                hdr['CRPIX2'] = ('{0:14.7G}'.format(cube.shape[2]/2.+np.float(hdr['CRPIX2'])-median(cube_list['info'][c+n][np.where(cube_list['info'][c+n][:,5]>0),5])), "")
                hdr['history'] =  'Updated CRPIX1, CRPIX2'

            # Mean the frames if requested
            if args.combine_frames:
                new_cube = np.nanmean(new_cube,axis = 0)
            else:
                # Remove the bad frames that contain only NaNs
                bad_frames = (np.sum(np.isnan(new_cube) is False,
                                     axis=(1,2)) == 0)
                new_cube = new_cube[bad_frames is False]
                new_info = new_info[bad_frames is False]
                hdr['HIERARCH GC NAN_FRAMES_REMOVED'] = (np.sum(bad_frames),
                   "NaN-only frames removed during recentring")


            graphic_nompi_lib.save_fits(psf_sub_filename, new_cube,
                                        target_dir=target_dir, hdr=hdr,
                                        backend='pyfits')
            graphic_nompi_lib.write_array2rdb(info_dir + os.sep + info_filename,
                                              new_info, header_keys)

        sys.stdout.write("\n Saved: {name} .\n Processed in {human_time} at {rate:.2f} MB/s \n"
                         .format(name = psf_sub_filename, human_time = graphic_nompi_lib.humanize_time(MPI.Wtime()-t0_cube) ,
                                 rate = os.path.getsize(psf_sub_filename)/(1048576*(MPI.Wtime()-t0_cube))))
        sys.stdout.write("Remaining time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t_init)*(len(cube_list['cube_filename'])-c)/(c-skipped+1))+"\n")
        sys.stdout.flush()

        del cube

    print("\n Program finished, killing all the slaves...")
    print("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t_init)))
    comm.bcast("over", root=0)
    if skipped == len(cube_list['cube_filename']):
        sys.exit(0)

    if hdr is not None:
        if 'ESO OBS TARG NAME' in hdr.keys():
            log_file = log_file+"_" + hdr['ESO OBS TARG NAME'].replace(' ', '')+"_"+str(__version__)+".log"
        else:
            log_file = log_file+"_" + hdr['OBJECT'].replace(' ', '')+"_"+str(__version__)+".log"
        graphic_nompi_lib.write_log((MPI.Wtime()-t_init),log_file)
    sys.exit(0)

#################################################################################
#
# SLAVES
#
# slaves need to:
# receive stack and frame
# recentre frames in stack
# calculate median
# subtract median from frame
# improvement could be done by somehow keeping recentreed frames
else:
    ## nofft = comm.bcast(None,root = 0)
    todo = comm.bcast(None, root=0)

    while not todo == "over":
        if todo == "median":

            # Receive number of first column
            start_col = comm.recv(source = 0)
            # Receive stack to median
            stack = comm.recv(source = 0)
            if d>5:
                print("")
                print(str(rank)+" stack.shape: "+str(stack.shape))
            # Mask out the NaNs
            psf = nanmedian(stack, axis = 0)
            del stack
            comm.send(psf, dest = 0)
            del psf

        elif todo == "recentre":
            # Receive padding dimension
            l_max = comm.bcast(None,root = 0)
            # Receive info table used for recentreing
            info_stack = comm.bcast(None,root = 0)
            if d >2:
                print(str(rank)+" received info_stack: "+str(info_stack))

            # Receive number of first frame in cube
            s = comm.recv(source = 0)
            if d >2:
                print(str(rank)+" received first frame number: "+str(s))

            # Receive cube
            stack = comm.recv(source = 0)
            if d >2:
                if stack is None:
                    print(str(rank)+" received stack: "+str(stack))
                else:
                    print(str(rank)+" received stack, shape = "+str(stack.shape))


            if not stack is None:
                bigstack = np.zeros((stack.shape[0],l_max,l_max)) # ACC removed 2*

                # We need to account for odd-sized arrays which need an extra pixel here
                if (stack.shape[1] % 2)  == 0:
                    extra_pix = 0
                else:
                    extra_pix = 1

                bigstack[:,
                bigstack.shape[1]//2-stack.shape[1]//2:bigstack.shape[1]//2+stack.shape[1]//2+extra_pix,
                    bigstack.shape[2]//2-stack.shape[2]//2:bigstack.shape[2]//2+stack.shape[2]//2+extra_pix] = stack
                for fn in range(bigstack.shape[0]):
                    dprint(d>2, "recentreing frame: "+str(fn)+" with shape: "+str(bigstack[fn].shape))

                    if info_stack[s+fn,4] == -1 or info_stack[s+fn,5] == -1:
                        bigstack[fn] = np.NaN
                        continue
                    # Shift is given by (image centre position)-(star position)
                    if nofft == True: # Use interpolation
                        shift_amount = (stack.shape[1]/2.-info_stack[s+fn,4], stack.shape[2]/2.-info_stack[s+fn,5])
                        bigstack[fn] = ndimage.interpolation.shift(bigstack[fn], shift_amount, order = 3, mode = 'constant', cval = np.NaN, prefilter = False)
                    else: # Shift in Fourier space
                        left_edge = int(np.round(l_max//2 - info_stack[s+fn, 4]))
                        right_edge = int(np.round(l_max//2 - info_stack[s+fn, 4]+stack.shape[1]))
                        bottom_edge = int(np.round(l_max//2 - info_stack[s+fn, 5]))
                        top_edge = int(np.round(l_max//2 - info_stack[s+fn, 5]+stack.shape[2]))
                        # if (left_edge >0) & (right_edge < bigstack.shape[1]) & (bottom_edge > 0) & (top_edge <bigstack.shape[2]):
                        #     bigstack[fn,left_edge:right_edge,bottom_edge] = stack[fn,:,0]/2.
                        #     bigstack[fn,left_edge:right_edge,top_edge] = stack[fn,:,-1]/2.
                        #     bigstack[fn,left_edge,bottom_edge:top_edge] = stack[fn,0,:]/2.
                        #     bigstack[fn,right_edge,bottom_edge:top_edge] = stack[fn,-1,:]/2.
                        # Remove the NaNs
                        nan_mask = np.isnan(bigstack[fn])
                        bigstack[fn][nan_mask] = nanmedian(bigstack[fn])
                        bigstack[fn] = graphic_nompi_lib.fft_shift(bigstack[fn],
                                stack.shape[1]/2.-info_stack[s+fn, 4],
                                stack.shape[2]/2.-info_stack[s+fn, 5])

                    # Now turn all of the values that wrapped into NaNs.
                    # Due to a quirk of python's indexing, negative indexes wrap while those greater than the array size do not. So we need to be careful here not to NaN perfectly good data!
                    if left_edge >0:
                        bigstack[fn,:left_edge,:] = np.NaN
                    bigstack[fn,right_edge:,:] = np.NaN
                    if bottom_edge >0:
                        bigstack[fn,:,:bottom_edge] = np.NaN
                    bigstack[fn,:,top_edge:] = np.NaN
                dprint(d>2, "Sending back bigstack, shape = "+str(bigstack.shape))
                comm.send(bigstack, dest = 0)
                del bigstack
            else:
                comm.send(None, dest = 0 )

            del stack

        else:
            print(str(rank)+": received "+str(todo)+". Leaving....")
            comm.send(None, dest = 0 )

        todo = comm.bcast(None,root = 0)
