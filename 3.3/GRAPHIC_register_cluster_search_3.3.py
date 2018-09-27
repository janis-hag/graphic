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

__version__='3.3'
__subversion__='0'

import numpy, glob, os, sys,argparse, fnmatch
 ## pickle, tables, argparse
from mpi4py import MPI
#from gaussfit_nosat import fitgaussian_nosat
#from gaussfit import fitgaussian, i_fitmoffat, moments
import gaussfit_330 as gaussfit
#from scipy import stats
import graphic_nompi_lib_330 as graphic_nompi_lib
import graphic_mpi_lib_330 as graphic_mpi_lib
from graphic_mpi_lib_330 import dprint
import numpy as np
from astropy.io import fits as pyfits
import bottleneck
from scipy import ndimage
import dateutil.parser

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser(description='GRAPHIC:\n The Geneva Reduction and Analysis Pipeline for High-contrast Imaging of planetary Companions.\n\n\
This program creates a list containing information on each frame: frame quality, gaussian fitted position of star, parallactic angle.')
parser.add_argument('--debug', action="store",  dest="d", type=int, default=0)
parser.add_argument('--pattern', action="store", dest="pattern", type=str, required=True, help='Filename pattern')
parser.add_argument('--win', action="store", dest="win", type=float, help='Gaussian fit window size')
parser.add_argument('--rmin', action="store", dest="rmin", type=float, default=5, help='The minimal radius in pixels to consider')
parser.add_argument('--rmax', action="store", dest="rmax", type=float, default=300, help='The Maximal spot size')
parser.add_argument('--t', action="store", dest="t", type=float, default=20, help='The threshold value')
parser.add_argument('--t_max', action="store", dest="t_max", type=float, help='The maximum deviation value')
parser.add_argument('--ratio', action="store", dest="ratio", type=float, help='PSF ellipticity ratio')
## parser.add_argument('--date', action="store", dest="date", help='Give the date in ISO format, e.g. 2015-11-30')
parser.add_argument('--log_file', action="store", dest="log_file",  default='GRAPHIC', help='Log filename')
parser.add_argument('--size', action="store", dest="size", type=int, default=-1, help='Filter size')
parser.add_argument('--chuck', action="store", dest="hiciao_filename", help='HICIAO fits file to use as reference for chuckcam data')

# No argument options...
parser.add_argument('-deviation', dest='deviation', action='store_const',
                   const=True, default=False,
                   help='Use a threshold based on standard deviation instead of an absolute threshold')
parser.add_argument('-sat', dest='nosat', action='store_const',
                   const=False, default=True,
                   help='Discard saturated pixels')
parser.add_argument('--recurs', dest='recurs', action='store',
                   type=int, default=2500,
                   help='Change the recursion limit for the centroid search')
parser.add_argument('-no_neg', dest='no_neg', action='store_const',
                   const=True, default=False,
                   help='Null negative pixels before PSF fitting')
parser.add_argument('-moffat', dest='moffat', action='store_const',
                   const=True, default=False,
                   help='Use moffat fitting instead of gaussian')
parser.add_argument('-nofit', dest='nofit', action='store_const',
                   const=True, default= False,
                   help='No PSF fitting performed.')
parser.add_argument('-stat', dest='stat', action='store_const',
                   const=True, default=False,
                   help='Print benchmarking statistics')
parser.add_argument('-drh', dest='spherepipe', action='store_const',
                   const=True, default=False,
                   help='Switch for data pre-processed by the SPHERE DRH')
parser.add_argument('-naco', dest='naco', action='store_const',
                   const=True, default=False,
                   help='Switch for NACO data')
parser.add_argument('-sphere', dest='sphere', action='store_const',
                   const=True, default=False,
                   help='Switch for RAW SPHERE data')
parser.add_argument('-scexao', dest='scexao', action='store_const',
                   const=True, default=False,
                   help='Switch for SCExAO data')
parser.add_argument('-naco_pack', dest='naco_pack', action='store_const',
                   const=True, default=False,
                   help='Switch for repacked single frame NACO data')
parser.add_argument('-no_psf', action='store_const', dest='no_psf',
                    const=True, default=False,
                    help='Do not look for a PSF, assume it is already centred.')
parser.add_argument('-keepfirst', action='store_const', dest='keepfirst',
                    const=True, default=False,
                    help='Keep first frame in NACO L-band cubes.')
parser.add_argument('-multispot', action='store_const', dest='multispot',
                    const=True, default=False,
                    help='Keep the brightest detected spot in case of multiple detections.')
## parser.add_argument('--centre_file', action="store", dest="centre_file", default=None, help='Centroids file generated by the SPHERE pipeline.')


args = parser.parse_args()
d=args.d
pattern=args.pattern
D=args.win
min_size=args.rmin
max_size=args.rmax
thres_coefficient=args.t
max_deviation=args.t_max
ratio=args.ratio
spherepipe=args.spherepipe
naco=args.naco
naco_pack=args.naco_pack
scexao=args.scexao
log_file=args.log_file
no_neg=args.no_neg
nosat=args.nosat
moffat=args.moffat
nofit=args.nofit
deviation=args.deviation
## hdf5=args.hdf5
recurs=int(args.recurs)
window_size=args.size
no_psf=args.no_psf
hiciao_filename=args.hiciao_filename
sphere=args.sphere

if moffat:
    header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
        'frame_num', 'frame_time', 'paralactic_angle']
else:
    header_keys=['frame_number', 'psf_barycentre_x', 'psf_barycentre_y', 'psf_pixel_size', 'psf_fit_centre_x', 'psf_fit_centre_y', 'psf_fit_height', 'psf_fit_width_x', 'psf_fit_width_y',
        'frame_num', 'frame_time', 'paralactic_angle']

if hiciao_filename is None:
    chuck=False
else:
    chuck=True

target_dir = "."
backup_dir = "prev"
positions_dir = "cube-info"
iterations = 1
## args=6
comments=None

sys.setrecursionlimit(recurs)
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

    if spherepipe:
        fctable_list=glob.glob(positions_dir+os.sep+'SPHER*fctable.rdb')
        fctable_list.sort()

    elif chuck:
        # Read the IERS_A table
        iers_a = graphic_nompi_lib.read_iers_a()
        if os.access(hiciao_filename, os.F_OK ):
            hdr=pyfits.getheader(hiciao_filename)
        else:
            print(hiciao_filename+' not found. Interrupting!')
            comm.Abort()
            sys.exit(1)
    elif scexao and not chuck:
        fctable_list=glob.glob(positions_dir+os.sep+'scexao_parang_*.rdb')
        fctable_list.sort()
    elif naco_pack:
        fctable_list=glob.glob(positions_dir+os.sep+'naco_parang_*.rdb')
        fctable_list.sort()

    if len(dirlist)==0 or ((spherepipe or chuck or scexao or naco_pack) and len(fctable_list)==0):
    ## if len(dirlist)==0 or interrupt==True: # or fctable_list is None or len(fctable_list)==0:
        print("No files found!")
        comm.Abort()
        ## for n in range(nprocs-1):
            ## comm.send("over", dest = n+1 )
            ## comm.send("over", dest = n+1 )
        sys.exit(1)

    for i in range(len(dirlist)):
        # Read cube header and data
        #header_in=pyfits.getheader(dirlist[i])
        t_cube=MPI.Wtime()
        #check if already processed
        ## if hdf5:
            ## filename='all_info_'+str(thres_coefficient)+"_"+str(min_size)+"_"+str(max_size)+"_"+dirlist[i][:-5]+'.hdf5'
        ## else:
        filename='all_info_'+str(thres_coefficient)+"_"+str(min_size)+"_"+str(max_size)+"_"+dirlist[i][:-5]+'.rdb'

        if os.access(positions_dir + os.sep +filename, os.F_OK ):
            print("["+str(i+1)+"/"+str(len(dirlist))+"]: "+filename+" already exists. SKIPPING")
            skipped=skipped+1
            continue

        print("["+str(i+1)+"/"+str(len(dirlist))+"]: Processing "+str(dirlist[i]))

        if not os.access(dirlist[i], os.F_OK ): # Check if file exists
            print("Error: cannot access file "+dirlist[i])
            skipped=skipped+1
            continue
        else:
            cube,cube_header=pyfits.getdata(dirlist[i], header=True)
            if not chuck:
                hdr=cube_header
            else: #Creating a header for the empty chuck cam headers
                cube_header['OBS-MOD'] = hdr['OBS-MOD']
                cube_header.comments['OBS-MOD'] = 'Observation mode'
                cube_header['P_TRMODE']= hdr['P_TRMODE']
                cube_header.comments['P_TRMODE']= 'Tracking mode of Lyot stop'
                cube_header['DATA-TYP']= hdr['DATA-TYP']
                cube_header.comments['DATA-TYP']= 'Type / Characteristics of this data'
                cube_header['OBJECT']  = hdr['OBJECT']
                cube_header.comments['OBJECT']  = 'Target Description'
                cube_header['RADECSYS']= hdr['RADECSYS']
                cube_header.comments['RADECSYS']= 'The equitorial coordinate system'
                cube_header['RA']    =hdr['RA']
                cube_header.comments['RA']    = 'HH:MM:SS.SSS RA pointing'
                cube_header['DEC']     =hdr['DEC']
                cube_header.comments['DEC']     = '+/-DD:MM:SS.SS DEC pointing'
                cube_header['EQUINOX'] =  hdr['EQUINOX']
                cube_header.comments['EQUINOX'] =  'Standard FK5 (years)'
                cube_header['RA2000']  = hdr['RA2000']
                cube_header.comments['RA2000']  = 'HH:MM:SS.SSS RA (J2000) pointing)'
                cube_header['DEC2000']= hdr['DEC2000']

                graphic_nompi_lib.save_fits('h_'+dirlist[i], cube, hdr=cube_header, backend='pyfits', verify='warn')


        #######
        # Currently crashes if not rdb file found. Should print an error instead and continue.
        ######
        if spherepipe:
            fctable_filename = fnmatch.filter(fctable_list,'*'+dirlist[i][-40:-10]+'*')[0]
            fctable=graphic_nompi_lib.read_rdb(fctable_filename)
            parang_list=None
            if not 'Angle_deg' in fctable.keys():
                print(str(fctable_filename)+' does not contain Angle_deg in keys: '+str(fctable.keys()))
            for i in range(len(fctable['Angle_deg'])):
                jdate = graphic_nompi_lib.datetime2jd(dateutil.parser.parse(fctable['Time-UT'][i]))
                if parang_list is None:
                    parang_list=numpy.array([i,jdate,fctable['Angle_deg'][i]])
                    ## utcstart=datetime2jd(dateutil.parser.parse(hdr['DATE']+"T"+hdr['UT']))
                else:
                    parang_list=numpy.vstack((parang_list,[i,jdate,fctable['Angle_deg'][i]]))
        elif 'INSTRUME' in cube_header.keys() and cube_header['INSTRUME']=='SPHERE':
            #parang_list=graphic_nompi_lib.create_parang_list_sphere(cube_header)
            parang_list=np.atleast_2d(graphic_nompi_lib.create_parang_list_sphere(cube_header))
        elif sphere:
            parang_list=np.atleast_2d(graphic_nompi_lib.create_parang_list_sphere(cube_header))
        elif naco_pack or (scexao and not chuck):
            fctable_filename= fnmatch.filter(fctable_list,'*'+dirlist[i].split('_')[-1][:-5]+'.rdb')[0]
            fctable=graphic_nompi_lib.read_rdb(fctable_filename)
            parang_list=np.array([fctable['frame_num'][:],fctable['frame_time'][:],fctable['paralactic_angle'][:]])
            parang_list=(np.rollaxis(parang_list,1))
        elif chuck:
            ## 'ircam'+string.split(tfile,'ircam')[1]
#            frame_text_info=string.replace('ircam'+string.split(dirlist[i],'ircam')[1],'fits','txt')
            frame_text_info='ircam'+dirlist[i].split('ircam')[1].replace('fits','txt')
            if os.access(frame_text_info, os.F_OK | os.R_OK):
                f=open(frame_text_info)
                timestamps=f.readlines()
                parang_list=graphic_nompi_lib.create_parang_scexao_chuck(timestamps, hdr, iers_a)
            else:
                print('No '+frame_text_info+' file found. Skipping '+dirlist[i])
                continue
        elif naco:
            # Creates a 2D array [frame_number, frame_time, paralactic_angle]
            parang_list=graphic_nompi_lib.create_parang_list_naco(cube_header)
        else:
            print('Unknown instrument. Please specify one using the available command switches.')
            comm.Abort()
            sys.exit(1)

        print('Parang list generated')

        if no_psf:
            print('no psf')
            comm.bcast('over', root=0)
            cent_list=np.ones((cube.shape[0],9))
            cent_list[:,0]=np.arange(cube.shape[0]) # Frame number
            cent_list[:,1]=cube.shape[1]/2. # X centre
            cent_list[:,2]=cube.shape[2]/2. # Y centre
            del cube
        else:
            sat=-1
            if nosat:
                # Get saturation level
                sat=graphic_nompi_lib.get_saturation(cube_header)
            comm.bcast(sat, root=0)
            ## print('saturation broadcasted: '+str(sat))
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
        if 'ESO DET NDIT' in cube_header.keys() and not no_psf and cube_header['NAXIS3']==cube_header['ESO DET NDIT']:
            cent_list[-1]=-1

        # Set first frame to invalid for L_prime band due to cube reset effects
        if not no_psf and not args.keepfirst and 'ESO INS OPTI6 ID' in cube_header.keys() and cube_header['ESO INS OPTI6 ID']=='L_prime':
            cent_list[0]=-1

        if comments is None and not 'ESO ADA PUPILPOS' in cube_header.keys():
            comments="Warning! No ESO ADA PUPILPOS keyword found. Is it ADI? Using 89.44\n"

        graphic_nompi_lib.write_array2rdb(positions_dir+os.sep+filename,cent_list,header_keys)

        if d>2:
            print("saved cent_list "+str(cent_list.shape)+" :" +str(cent_list))

        sys.stdout.write('\r\r\r')
        sys.stdout.flush()

        if not no_psf:
            bad=np.where(cent_list[:,6]==-1)[0]
            print(dirlist[i]+" total frames: "+str(cent_list.shape[0])+", rejected: "+str(len(bad))+" in "+str(MPI.Wtime()-t_cube)+" seconds.")
        del cent_list

        t_cube=MPI.Wtime()-t_cube
        # print(" ETA: "+humanize_time(t_cube*(len(dirlist)-i-1)))
        print(" Remaining time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)*(len(dirlist)-i-1)/(i+1-skipped)))

    if len(dirlist)==skipped: # Nothing to be done.
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



    graphic_nompi_lib.write_log_hdr((MPI.Wtime()-t_init), log_file, hdr, comments, nprocs=nprocs)
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

    sat=comm.bcast(None, root=0)
    ## print('sat: '+str(sat))
    if sat=='over':
        sys.exit(0)
    startframe=comm.recv(source = 0) # get number of first frame
    data_in=comm.recv(source = 0)
    ## print('startframe, data_in'+str(startframe)+', '+str(data_in))
    cube_count=1
    centre=None
    x0_i=0
    y0_i=0

    while not data_in is "over":
        if not data_in is None and isinstance(data_in, np.ndarray):
            for frame in range(data_in.shape[0]):
                sys.stdout.write('\r\r\r [Rank '+str(rank)+', cube '+str(cube_count)+']  Frame '+str(frame+startframe)+' of '+str(startframe+data_in.shape[0]))
                sys.stdout.flush()

                if window_size>0:
                    data_in[frame]=data_in[frame]-ndimage.filters.median_filter(data_in[frame],size=(window_size,window_size),mode='reflect')
                if deviation:
                    sigma = bottleneck.nanstd(data_in[frame])
                    median = bottleneck.nanmedian(data_in[frame])
                    threshold = sigma*thres_coefficient
                    graphic_mpi_lib.dprint(d>1,"Sigma: "+str(sigma)+", median: "+str(median)+", threshold: "+str(threshold))
                    data_in[frame] = data_in[frame]-median
                    data_in[frame] = np.where(data_in[frame]>sigma*max_deviation, median, data_in[frame])
                else:
                    threshold=thres_coefficient

                # Starting rough centre search and quality check
                cluster_array_ref, ref_ima, count = graphic_mpi_lib.cluster_search(data_in[frame], threshold, min_size, max_size, x0_i , y0_i,d=d)
                dprint(d>3,"cluster_search, on frame["+str(frame)+"]: "+str(cluster_array_ref))
                if not count == 1: # Check if one and only one star has been found
                    if args.multispot and not cluster_array_ref is None:
                        ## print('Multispot:'+str(cluster_array_ref))
                        # Skip psf fitting procedure, and copy values
                        cluster_array_ref=np.append(cluster_array_ref,[cluster_array_ref[0],cluster_array_ref[1],cluster_array_ref[2], 1, 1])
                    else:
                        cluster_array_ref=np.array([-1 , -1, -1, -1 , -1 , -1, -1 , -1])
                elif nofit: # Skip psf fitting procedure, and copy values
                    cluster_array_ref=np.append(cluster_array_ref,[cluster_array_ref[0],cluster_array_ref[1],cluster_array_ref[2], 1, 1])
                else:
                    x0_i = np.ceil(cluster_array_ref[0])
                    y0_i = np.ceil(cluster_array_ref[1])
                    if d ==1:
                        print('\r [Rank '+str(rank)+', cube '+str(cube_count)+'] Number of spots detected in frame ('+str(frame+1)+'): '+str(count)+'\n')

                    # Check if D needs to be redefined
                    if D+x0_i>data_in[frame].shape[0]:
                        if d>2:
                            print("Redefined D because: D+x0_i>data_in[frame].shape[0] "+str(D)+"+"+str(x0_i)+"="+str(D+x0_i)+">"+str(data_in[frame].shape[0]))
                        D=data_in[frame].shape[0]-x0_i
                    elif D+y0_i>data_in[frame].shape[1]:
                        if d>2:
                            print("Redefined D because: D+y0_i>data_in[frame].shape[1] "+str(D)+"+"+str(y0_i)+"="+str(D+y0_i)+">"+str(data_in[frame].shape[1]))
                        D=data_in[frame].shape[1]-y0_i
                    elif  x0_i-D<0:
                        if d>2:
                            print("Redefined D because: D>y0_i "+str(D)+">"+str(x0_i))
                        D=x0_i
                    elif y0_i-D<0:
                        if d>2:
                            print("Redefined D because: D>y0_i "+str(D)+">"+str(y0_i))
                        D=y0_i

                    # Starting gaussian fitting for better centre determination
                    ## print(x0_i-D,x0_i+D,y0_i-D,y0_i+D)
                    centre_win=data_in[frame,x0_i-D:x0_i+D,y0_i-D:y0_i+D]
                    if no_neg: # put negative values to zero
                        centre_win=np.where(centre_win<0,0,centre_win)

                    ## print('\n'+str(frame+startframe)+': '+str(moments(centre_win))+', '+str(centre_win))

                    if d>1:
                        print("")
                        print("D: "+str(D)+", x0_i: "+str(x0_i)+", y0_i: "+str(y0_i))
                        print("centre_win.shape: "+str(centre_win.shape))
                    try:
                    ## if True:
                        if moffat:
                            g_param=gaussfit.i_fitmoffat(centre_win)
                        ## elif nosat:
                            ## g_param=gaussfit_247.fitgaussian_nosat(centre_win,sat)
                        else:
                            g_param=gaussfit.fitgaussian(centre_win)
                    except:
                        print("\n Something went wrong with the PSF fitting!")
                        g_param=np.array([-1 , -1, -1, -1 , -1 ])
                    if d>2:
                        print("")
                        print("height: "+str(g_param[0])+" x, y: "+str(g_param[1])+", "+str(g_param[2])+", width_x, width_y: "+str(g_param[3])+", "+str(g_param[4]))

                    x0_g=x0_i+g_param[1]-D
                    y0_g=y0_i+g_param[2]-D

                    if moffat:
                        cluster_array_ref=np.append(cluster_array_ref,[x0_g, y0_g, g_param[0], g_param[3], g_param[4]])
                    elif g_param[3]/g_param[4] < 1./ratio and g_param[3]/g_param[4] > ratio:
                        cluster_array_ref=np.append(cluster_array_ref,[x0_g, y0_g, g_param[0], g_param[3], g_param[4]])
                    else:
                        print("\n PSF shape rejection: "+str(g_param[3]/g_param[4]))
                        cluster_array_ref=np.append(cluster_array_ref,[-1 , -1, -1, -1 , -1 ])


                cluster_array_ref=np.append(frame+startframe,cluster_array_ref)


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
        sat=comm.bcast(None, root=0)
        startframe=comm.recv(source = 0) # get number of first frame
        data_in=comm.recv(source = 0)
        centre=None
        x_0=0
        y_0=0


    else:
        comm.send("OK", dest = 0)
        sys.exit(0)
