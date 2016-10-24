import numpy as np
import scipy, glob, sys
import astropy.io.fits as pyfits
import graphic_nompi_lib_330 as graphic_nompi_lib
from mpi4py import MPI
import argparse

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Converts images to the correct format for the PCA code')
parser.add_argument('--pattern', action="store", dest="pattern",  default="cl_nomed_SPHER*STAR_CENTER", help='cubes to apply the star centering')


args = parser.parse_args()
pattern=args.pattern

if rank==0:
	t0=MPI.Wtime()
	sys.stdout.write("beginning of convert")
	sys.stdout.write("\n")
	sys.stdout.flush()
	
	# Get the list of files
	dirlist=glob.glob(pattern+"*")
	dirlist.sort()
	for i,allfiles in enumerate(dirlist):
	    sys.stdout.write(allfiles)
	    sys.stdout.write('\n')
	sys.stdout.flush()
	
	# Loop through the files and combine them into a single cube
	count=1
	paralactic_angle_vec=[]
	for i,allfiles in enumerate(dirlist):
	    if count==1:
	        master_cube,hdr=pyfits.getdata(allfiles,header=True)
	    else:
	        cube_temp=pyfits.getdata(allfiles)
	        master_cube=np.append(master_cube,cube_temp,axis=0)
	    f=open(glob.glob("cube-info/*"+allfiles.replace(".fits",".rdb"))[0],'r')
	    lines=f.readlines()
	    f.close()
	
	    for line in lines:
	    	paralactic_angle=line.strip().split()[11]
	    	if ((not "paralactic_angle" in paralactic_angle) and (not "---" in paralactic_angle)):
	    		paralactic_angle_vec=np.append(paralactic_angle_vec,paralactic_angle)
	    count+=1
	
	pyfits.writeto("master_cube_PCA.fits",master_cube,header=hdr,clobber=True)
	f2=open("paralactic_angle.txt",'w')
	for i,paralactic_angle in enumerate(paralactic_angle_vec):
		f2.write(paralactic_angle+"\n")
	f2.close()

	sys.stdout.write("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
	sys.stdout.write("\n")
	sys.stdout.write("end of convert")
	sys.stdout.flush()
	# os._exit(1)
	sys.exit(0)
else:
	# sys.exit(1)
	sys.exit(0)

