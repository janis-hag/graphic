#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This program is part of GRAPHIC: "The Geneva Reduction and Analysis Pipeline for
High-contrast Imaging of planetary Companions".

This program is designed for NACO AGPM data analysis. In NACO data, the AGPM coronagraph
moves with time. This program uses the previously measured offset between the AGPM substrate 
and the coronagraph itself from GRAPHIC_naco_agpm_offset.
This program first calculates the centre of the AGPM subtrate, then uses the measured offset
to calculate the centre of the AGPM coronagraph. Then it performs a two-Gaussian fit to each
frame with the AGPM position fixed.

"""

__version__='3.3'
__subversion__='0'

import numpy, glob, sys,argparse,os
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

parser.add_argument('--psf_width', action="store", dest="psf_width", type=float, default=4, help='An initial guess for the psf width')
parser.add_argument('--agpm_width', action="store", dest="agpm_width", type=float, default=1.5, help='An initial guess for the agpm width')
parser.add_argument('--agpm_offset_file', action="store", dest="agpm_offset_file", type=str, default='../Sky/agpm_offset.txt', help='File containing measured AGPM offset')



# No argument options...
parser.add_argument('-saturated', dest='saturated', action='store_const',
                   const=True, default=False,
                   help='Use a saturated psf model and fit to the saturation level (in counts)')


args = parser.parse_args()
d=args.d
pattern=args.pattern
log_file=args.log_file
saturated=args.saturated
psf_width=args.psf_width
agpm_width=args.agpm_width
agpm_offset_file = args.agpm_offset_file

# if moffat:
    # header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
        # 'frame_num', 'frame_time', 'paralactic_angle']
# else:
header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
        'frame_num', 'frame_time', 'paralactic_angle']

target_dir = "."
backup_dir = "prev"
positions_dir = "cube-info"
## args=6
comments=None

# sys.setrecursionlimit(recurs)
t_init=MPI.Wtime()

print(rank, nprocs)

if rank==0:  # Master process
    graphic_nompi_lib.print_init()
    # try:
    t0=MPI.Wtime()
    skipped=0

    ## print(rank)

    ## print(thres_coefficient)

    dirlist=glob.glob(pattern+'*.fits')
    dirlist.sort() # Sort the list alphabetically

    if dirlist==None or len(dirlist)<1:
        print("No files found")

        # Close all of the slaves
        for ix in range(nprocs-1):
            comm.send([],dest=ix+1)
            comm.send(None,dest=ix+1)

        # comm.bcast(None)
        MPI.Finalize()
        sys.exit(0)

    # Load the agpm offset
    mean_agpm_offset = np.loadtxt(agpm_offset_file)
    print("Using AGPM offset of (x,y) = ("+str(mean_agpm_offset[0])+","+str(mean_agpm_offset[1])+")")

    for i in range(len(dirlist)):
        # Read cube header and data
        t_cube=MPI.Wtime()
        filename='all_info_'+str(psf_width)+"_"+dirlist[i][:-5]+'.rdb'

        print("["+str(i+1)+"/"+str(len(dirlist))+"]: Processing "+str(dirlist[i]))

        # We need to load both the sky subtracted and pre-sky-subtracted cubes
        cube,cube_header=pyfits.getdata(dirlist[i],header=True)
        pre_sky_name = dirlist[i].replace('nomed_','')
        pre_sky_cube = pyfits.getdata(pre_sky_name)

        #######
        # Currently crashes if not rdb file found. Should print an error instead and continue.
        ######
        # Creates a 2D array [frame_number, frame_time, paralactic_angle]
        parang_list=graphic_nompi_lib.create_parang_list_naco(cube_header)

        # ACC: Find the centre of the big circle using the median frame before sky subtraction
        mean_frame=np.median(pre_sky_cube,axis=0)

        # Calculate the centre of the big circle    
        big_circle_cen,agpm_rad = gaussfit.fit_to_big_circle(mean_frame)
        big_circle_cen_pix = np.round(big_circle_cen).astype(int)
        
        # Calculate the AGPM position using the known offset
        agpm_pos = big_circle_cen + mean_agpm_offset

        # Send the AGPM position to the processes
        comm.bcast(agpm_pos, root=0)

        # send_frames...
        graphic_mpi_lib.send_frames(cube)
        del cube
        # Prepare the centroid array:
        # [frame_number, psf_barycentre_x, psf_barycentre_y, psf_pixel_size, psf_fit_centre_x, psf_fit_centre_y, psf_fit_height, psf_fit_width_x, psf_fit_width_y]
        cent_list=None

        # Receive data back from slaves
        for n in range(nprocs-1):
            data_in=None
            data_in=comm.recv(source = n+1)
            if data_in is None:
                continue
            elif cent_list  is None:
                cent_list=data_in.copy()
            else:
                cent_list=np.vstack((cent_list,data_in))


        if not os.path.isdir(positions_dir): # Check if positions dir exists
            os.mkdir(positions_dir)

        if cent_list is None:
            print("No centroids list generated for "+str(dirlist[i]))
            continue

        if d>2:
            print("parang_list "+str(parang_list.shape)+" : "+str(parang_list))
            print("cent_list "+str(cent_list.shape)+" :" +str(cent_list))


        #Create the final list:
        # [frame_number, psf_barycentre_x, psf_barycentre_y, psf_pixel_size, psf_fit_centre_x, psf_fit_centre_y, psf_fit_height, psf_fit_width_x, psf_fit_width_y  ,  frame_number, frame_time, paralactic_angle]

        cent_list=np.hstack((cent_list,parang_list))

        # Set last frame to invalid if it's the cube-median
        if ('ESO DET NDIT' in cube_header.keys()) and (cube_header['NAXIS3']!=cube_header['ESO DET NDIT']):
            cent_list[-1]=-1

        # Set first frame to invalid for L_prime band due to cube reset effects
        if 'ESO INS OPTI6 ID' in cube_header.keys() and cube_header['ESO INS OPTI6 ID']=='L_prime':
            cent_list[0]=-1

        if comments is None and not 'ESO ADA PUPILPOS' in cube_header.keys():
            comments="Warning! No ESO ADA PUPILPOS keyword found. Is it ADI? Using 89.44\n"

        ## if hdf5:
            ## # Open a new empty HDF5 file
            ## f = tables.openFile(positions_dir+os.sep+filename, mode = "w")
            ## # Get the root group
            ## hdfarray = f.createArray(f.root, 'centroids', cent_list, "List of centroids for cube "+str(dirlist[i]))
            ## f.close()
        ## else:
        graphic_nompi_lib.write_array2rdb(positions_dir+os.sep+filename,cent_list,header_keys)

        if d>2:
            print("saved cent_list "+str(cent_list.shape)+" :" +str(cent_list))

        sys.stdout.write('\n\n')
        sys.stdout.flush()

        bad=np.where(cent_list[:,6]==-1)[0]
        print(dirlist[i]+" total frames: "+str(cent_list.shape[0])+", rejected: "+str(len(bad))+" in "+str(MPI.Wtime()-t_cube)+" seconds.")

        del cent_list

        t_cube=MPI.Wtime()-t_cube
        # print(" ETA: "+humanize_time(t_cube*(len(dirlist)-i-1)))
        print(" Remaining time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)*(len(dirlist)-i-1)/(i+1-skipped)))

    if len(dirlist)==skipped: # Nothing to be done, so close the slave processes (ACC edit May 2017)
        comm.bcast("over", root=0)
        for n in range(nprocs-1):
            comm.send("over", dest = n+1 )
            comm.send("over", dest = n+1 )
        MPI.Finalize()
        sys.exit(0)

    print("")
    print(" Total time: "+graphic_nompi_lib.humanize_time(MPI.Wtime()-t0))
    if not len(dirlist)-skipped==0:
        print(" Average time per cube: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)/(len(dirlist)-skipped))+" = "+str((MPI.Wtime()-t0)/(len(dirlist)-skipped))+" seconds.")


    ## if 'ESO OBS TARG NAME' in hdr.keys():
        ## log_file=log_file+"_"+string.replace(hdr['ESO OBS TARG NAME'],' ','')+"_"+str(__version__)+".log"
    ## elif 'OBJECT' in hdr.keys():
        ## log_file=log_file+"_"+string.replace(hdr['OBJECT'],' ','')+"_"+str(__version__)+".log"
    ## else:
        ## log_file=log_file+"_UNKNOW_TARGET_"+str(__version__)+".log"


    # print("the end!")
    graphic_nompi_lib.write_log_hdr((MPI.Wtime()-t_init), log_file, cube_header, comments, nprocs=nprocs)
    ## graphic_nompi_lib.write_log((MPI.Wtime()-t_init), log_file, comments, nprocs=nprocs)
    # Stop slave processes
    comm.bcast("over", root=0)
    for n in range(nprocs-1):
        comm.send("over", dest = n+1 )
        comm.send("over", dest = n+1 )
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


else: # Slave processes
    ## print(rank)

    # Receive the rough centre position
    agpm_pos = comm.bcast(None, root=0)

    startframe=comm.recv(source = 0) # get number of first frame
    data_in=comm.recv(source = 0)
    ## print('startframe, data_in'+str(startframe)+', '+str(data_in))
    cube_count=1
    centre=None
    x0_i=0
    y0_i=0

    while not type(data_in)==type("over"):
        if not data_in is None and isinstance(data_in, np.ndarray):


            for frame in range(data_in.shape[0]):
                sys.stdout.write('\n  [Rank '+str(rank)+', cube '+str(cube_count)+']  Frame '+str(frame+startframe)+' of '+str(startframe+data_in.shape[0]))
                sys.stdout.flush()

                image=data_in[frame]

                # Cut out a small region to do the fitting
                cut_sz = 6
                agpm_pos_pix = np.round(agpm_pos).astype(int)
                cut_agpm_pos = agpm_pos - agpm_pos_pix + cut_sz # this is the position of the agpm in im_cut

                cut_agpm_pos = cut_agpm_pos[::-1] # I think I got the axes wrong again...
                
                im_cut = image[agpm_pos_pix[0]-cut_sz:agpm_pos_pix[0]+cut_sz,
                                    agpm_pos_pix[1]-cut_sz:agpm_pos_pix[1]+cut_sz]
                
                # Do the fitting
                fit = gaussfit.fixed_agpm_gaussfit(im_cut,width=psf_width,agpm_width=agpm_width,
                                                  agpm_rejection_fraction=0.97,agpm_position=cut_agpm_pos)
                # The old fitting function, without fixed AGPM position
                # fit = gaussfit.agpm_gaussfit(im_cut,width=psf_width,agpm_width=agpm_width,
                                                  # agpm_rejection_fraction=0.97,agpm_position=cut_agpm_pos)



                # Now save the results [frame #, rough x cen, rough y cen, psf pixel size?, cen x, cen y, amplitude, x width, y width]
                star_params=fit.parameters[0:6] # (amplitude, x0, y0, sigmax, sigmay, theta)
                agpm_params=fit.parameters[6:]  # (amplitude, x0, y0, sigmax, sigmay, theta)
                centre_fit=star_params[2:0:-1]+ agpm_pos_pix - cut_sz # the output of agpm_gaussfit is relative to the edge of the cut frame
                # centre_fit=star_params[1:3]+ agpm_pos_pix - cut_sz # the output of agpm_gaussfit is relative to the edge of the cut frame
                cluster_array_ref=np.array([frame+startframe,agpm_pos[0],agpm_pos[1],0., centre_fit[0],centre_fit[1],star_params[0],star_params[3],star_params[4]])

                # y,x = np.indices(im_cut.shape)
                # model = fit(x,y)

                # plt.figure(1)
                # plt.clf()
                # plt.subplot(131)
                # plt.imshow(im_cut,vmin=-500,vmax=500)
                # plt.subplot(132)
                # plt.imshow(model,vmin=-500,vmax=500)
                # plt.subplot(133)
                # plt.imshow(im_cut-model,vmin=-500,vmax=500)

                # plt.show()



                # if rank==1:
                if centre is None:
                    centre=cluster_array_ref
                else:
                    centre=np.vstack((centre,cluster_array_ref))
            if d > 3:
                print(str(rank)+": "+str(centre))
            comm.send(centre, dest = 0)
        else:
            if d > 3:
                print(str(rank)+": "+str(centre))
            comm.send(None,dest=0)
        cube_count=cube_count+1
        agpm_pos=comm.bcast(None, root=0)
        startframe=comm.recv(source = 0) # get number of first frame
        data_in=comm.recv(source = 0)
        centre=None
        x_0=0
        y_0=0
    else:
        comm.send("OK", dest = 0)
        sys.exit(0)