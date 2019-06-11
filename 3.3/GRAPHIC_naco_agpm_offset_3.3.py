#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline for
High-contrast Imaging of planetary Companions".

This program is designed for NACO AGPM data analysis. In NACO data, the AGPM coronagraph
moves with time. This program calculates the offset between the AGPM substrate and the 
coronagraph itself, using its thermal emission.
It is designed to work on sky frames (i.e. with no star). It calculates the individual 
offsets for each file, and averages them together to give a single offset value that 
can be used when registering the target frames to fix the position of the AGPM.


"""

__version__='3.3'
__subversion__='0'

import numpy, glob, sys,argparse
 ## pickle, tables, argparse
from mpi4py import MPI
import gaussfit_330 as gaussfit
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
import numpy as np
from astropy.io import fits as pyfits
from scipy import optimize
import matplotlib.pyplot as plt

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser(description='GRAPHIC:\n The Geneva Reduction and Analysis Pipeline for High-contrast Imaging of planetary Companions.\n\n\
This program calculates the mean offset between the AGPM substrate and the AGPM coronagraph itself, from a list of sky frames.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern", type=str, required=True, help='Filename pattern')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')

parser.add_argument('--psf_width', action="store", dest="psf_width", type=float, default=2, help='An initial guess for the psf width')

# No argument options...
parser.add_argument('-saturated', dest='saturated', action='store_const',
                   const=True, default=False,
                   help='Use a saturated psf model and fit to the saturation level (in counts)')
parser.add_argument('-fit_1d', dest='fit_1d', action='store_const',
                   const=True, default=False,
                   help='Fit to the 1D profile of the big circle rather than the 2D shape.')


args = parser.parse_args()
d=args.d
pattern=args.pattern
log_file=args.log_file
saturated=args.saturated
psf_width=args.psf_width
fit_1d = args.fit_1d

# if moffat:
    # header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
        # 'frame_num', 'frame_time', 'paralactic_angle']
# else:
header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
        'frame_num', 'frame_time', 'paralactic_angle']


target_dir = "."
backup_dir = "prev"
positions_dir = "cube-info"
iterations = 1
## args=6
comments=None

# Handle problems caused by one process exiting without cleaning up the others
sys.excepthook = graphic_mpi_lib.global_except_hook

# sys.setrecursionlimit(recurs)
t_init=MPI.Wtime()

print(rank, nprocs)
##################

if rank==0:  # Master process
    graphic_nompi_lib.print_init()
    # try:
    t0=MPI.Wtime()
    skipped=0


    dirlist=glob.glob(pattern+'*.fits')
    dirlist.sort() # Sort the list alphabetically

    # print 'HACK!'
    # dirlist = [dirlist[-3],dirlist[-3]]

    print('Found: '+str(len(dirlist))+'Files')

    if dirlist==None or len(dirlist)<1:
        print("No files found")

        # Close all of the slaves
        for ix in range(nprocs-1):
            comm.send([],dest=ix+1)
            comm.send(None,dest=ix+1)

        # comm.bcast(None)
        MPI.Finalize()
        sys.exit(0)


    print('Distributing dirlist to slaves.')
    start_ix,proc_dirlist=graphic_mpi_lib.send_dirlist(dirlist)

else: # Slave processes
    ## print(rank)

    # Receive dirlist and star frame number
    proc_dirlist=comm.recv(None)
    if proc_dirlist is None:
        sys.exit(0)
    proc_start_ix = comm.recv(None)

# Now process the files
nfiles = len(proc_dirlist) # number of files for this process
# Set up arrays to save the positions
proc_agpm_pos = np.zeros((nfiles,2))
proc_circle_pos = np.zeros((nfiles,2))
proc_offsets = np.zeros((nfiles,2))
proc_agpm_rad = np.zeros((nfiles))

for ix,filename in enumerate(proc_dirlist):

    print(str(rank)+": Processing "+str(filename))

    # Load the sky frame
    sky_frame,sky_header=pyfits.getdata(filename, header=True)

    # If it was a cube, then take the median over frames
    if sky_frame.ndim == 3:
        sky_frame = np.nanmedian(sky_frame,axis=0)

    # Remove the rows with huge offsets from the rest
    diffs = np.nanmean(sky_frame[:-2]-sky_frame[2:],axis=1)
    bad_rows = np.where(np.abs(diffs) > (3*np.std(diffs)))
    orig_sky = 1*sky_frame
    for row in bad_rows:
        sky_frame[row] = np.nan

    # Calculate the position of the big circle in this image
    big_circle_cen,agpm_rad = gaussfit.fit_to_big_circle(sky_frame,fit_1d=fit_1d)
    big_circle_cen_pix = np.round(big_circle_cen).astype(int)
        
    proc_circle_pos[ix] = big_circle_cen
    proc_agpm_rad[ix] = agpm_rad

    # Calculate the position of the AGPM in this image
    # First clean the image a bit by removing the bias in each row/column
    clean_sky = 1*orig_sky # don't ignore the bad rows since this causes problems in the fitting
    im = clean_sky[big_circle_cen_pix[0]-100:big_circle_cen_pix[0]+100,
             big_circle_cen_pix[1]-100:big_circle_cen_pix[1]+100]
    for row in range(im.shape[0]):
        im[row] -= np.nanmedian(im[row])
    
    for col in range(im.shape[1]):
        im[:,col] -= np.nanmedian(im[:,col])
    
    clean_sky[big_circle_cen_pix[0]-100:big_circle_cen_pix[0]+100,
             big_circle_cen_pix[1]-100:big_circle_cen_pix[1]+100] = im
        
    # Cut out a region around the agpm using the expected offset
    cut_sz = 6 
    rough_agpm_offset = [-10.447,11.711]# Nico and Johan's value

    rough_agpm_cen = np.array([big_circle_cen_pix[0]+rough_agpm_offset[0],
                               big_circle_cen_pix[1]+rough_agpm_offset[1]],dtype=int)
    im_cut = 1*clean_sky[rough_agpm_cen[0]-cut_sz:rough_agpm_cen[0]+cut_sz,
                 rough_agpm_cen[1]-cut_sz:rough_agpm_cen[1]+cut_sz]
    
    im_cut -= np.median(im_cut)
    
    # Fit to the position
    psf_fit = gaussfit.psf_gaussfit(im_cut,saturated=False,width=psf_width)
    
    agpm_offset = [rough_agpm_offset[0]+psf_fit.x_mean-cut_sz,
                   rough_agpm_offset[1]+psf_fit.y_mean-cut_sz]
    agpm_pos = np.array(rough_agpm_cen) + [psf_fit.y_mean-cut_sz,psf_fit.x_mean-cut_sz]

    
    agpm_offset = agpm_pos - big_circle_cen
    
    # Save it to the output arrays
    proc_agpm_pos[ix] = agpm_pos
    proc_offsets[ix] = agpm_offset

    # # # Hack to check it is working
    # x,y = np.indices(im_cut.shape)
    # model = psf_fit(y,x)
    # plt.figure(1)
    # plt.clf()
    # plt.subplot(131)
    # plt.imshow(im_cut,vmin=-5000,vmax=5000)
    # plt.subplot(132)
    # plt.imshow(model,vmin=-5000,vmax=5000)
    # plt.subplot(133)
    # plt.imshow(im_cut-model,vmin=-5000,vmax=5000)

    # plt.show()


#################
# Rank 0 tasks to compile the results of the other processes
#################
if rank ==0:

    nfiles_tot = len(dirlist)
    agpm_offsets = np.zeros((nfiles_tot,2))
    agpm_pos = np.zeros((nfiles_tot,2))
    circle_pos = np.zeros((nfiles_tot,2))
    agpm_rad =  np.zeros((nfiles_tot))

    # Add the data from the rank0 process
    nfiles_proc = proc_offsets.shape[0]

    agpm_offsets[start_ix:start_ix+nfiles_proc] = proc_offsets
    agpm_pos[start_ix:start_ix+nfiles_proc] = proc_agpm_pos
    circle_pos[start_ix:start_ix+nfiles_proc] = proc_circle_pos
    agpm_rad[start_ix:start_ix+nfiles_proc] = proc_agpm_rad

    # Recover data from slaves
    # How many datasets do we expect to receive?
    n_data = np.min([nprocs-1,nfiles_tot-1])
    for n in range(n_data):
        start_ix = comm.recv(source=n+1)
        proc_offsets = comm.recv(source=n+1)
        proc_agpm_pos = comm.recv(source=n+1)
        proc_circle_pos = comm.recv(source=n+1)
        proc_agpm_rad = comm.recv(source=n+1)

        sys.stdout.write('\r\r\r AGPM offsets from '+str(n+1)+' received.')
        sys.stdout.flush()

        if not (proc_offsets is None):
            nfiles_proc = proc_offsets.shape[0]

            agpm_offsets[start_ix:start_ix+nfiles_proc] = proc_offsets
            agpm_pos[start_ix:start_ix+nfiles_proc] = proc_agpm_pos
            circle_pos[start_ix:start_ix+nfiles_proc] = proc_circle_pos
            agpm_rad[start_ix:start_ix+nfiles_proc] = proc_agpm_rad


    # Save it
    # Array columns are: [frame index, agpm offset x, agpm offset y, agpm pos x, agpm pos y, circle pos x, circle pos y, agpm radius]
    output_array = np.zeros((nfiles_tot,8))
    output_array[:,0] = np.arange(nfiles_tot)
    output_array[:,1:3] = agpm_offsets
    output_array[:,3:5] = agpm_pos
    output_array[:,5:7] = circle_pos
    output_array[:,7] = agpm_rad

    output_header = 'Frame Index, AGPM offset x, AGPM offset y, AGPM pos x, AGPM pos y, circle pos x, circle pos y, AGPM radius'

    np.savetxt('all_agpm_offsets.txt',output_array,header=output_header,fmt='%.5e')

    # And save the mean offset
    mean_offset = np.mean(agpm_offsets,axis=0)
    std_offset = np.std(agpm_offsets,axis=0)
    print('Mean offset between AGPM and big circle:')
    print('X: '+str(np.round(mean_offset[0],4))+' +/- '+str(np.round(std_offset[0],4)))
    print('X: '+str(np.round(mean_offset[1],4))+' +/- '+str(np.round(std_offset[1],4)))

    np.savetxt('agpm_offset.txt',mean_offset,header='[X offset (pix), Y offset (pix)]. Mean offset between AGPM and the big circle taken from the sky sequence')

    print("")
    print(" Total time: "+graphic_nompi_lib.humanize_time(MPI.Wtime()-t0))

    print(" Average time per cube: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)/(len(dirlist)))+" = "+str((MPI.Wtime()-t0)/(len(dirlist)))+" seconds.")
    sys.stdout.write('\n\n')
    sys.stdout.flush()


    # print("the end!")
    # graphic_nompi_lib.write_log_hdr((MPI.Wtime()-t_init), log_file, hdr, comments, nprocs=nprocs)
    ## graphic_nompi_lib.write_log((MPI.Wtime()-t_init), log_file, comments, nprocs=nprocs)
    MPI.Finalize()
else:
    # Send the data to rank 0
    comm.send(proc_start_ix,dest=0)
    comm.send(proc_offsets,dest=0)
    comm.send(proc_agpm_pos,dest=0)
    comm.send(proc_circle_pos,dest=0)
    comm.send(proc_agpm_rad,dest=0)

sys.exit(0)
