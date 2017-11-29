import numpy as np
import scipy, glob, os, sys, shutil
import astropy.io.fits as pyfits
import graphic_nompi_lib_330 as graphic_nompi_lib
from mpi4py import MPI
import argparse

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Converts images to the correct format for the PCA code')
parser.add_argument('--pattern', action="store", dest="pattern",  default="cl_nomed_SPHER*STAR_CENTER",
	help='cubes to apply the star centering.')
parser.add_argument('--output_dir', action="store", dest="output_dir",  default="./",
	help='output directory for the cube and parallactic angle file.')
parser.add_argument('--output_file',action="store",dest='output_file', default="master_cube_PCA.fits",
	help='Filename of the output fits file containing the stacked cube.')
parser.add_argument('-skip_parang',action='store_const',dest='skip_parang',const=True,default=False,
	help='Skip the generation of the parallactic angle file.')
parser.add_argument('-collapse_cube',action='store_const',dest='collapse_cube',const=True,default=False,
	help='Collapse the image cube into a single frame (used to make the PSF frame).')


args = parser.parse_args()
pattern=args.pattern
output_dir = args.output_dir
output_file = args.output_file
skip_parang = args.skip_parang
collapse_cube = args.collapse_cube

if rank==0:
	t0=MPI.Wtime()
	sys.stdout.write("beginning of convert")
	sys.stdout.write("\n")
	sys.stdout.flush()
	
	# Get the list of files
	dirlist=glob.glob(pattern+"*")
	dirlist.sort()
	for i,allfiles in enumerate(dirlist):
		sys.stdout.write('  '+allfiles)
		sys.stdout.write('\n')
	sys.stdout.flush()

	# Check that the output directory exists, and make it if needed
	if not output_dir.endswith(os.sep):
		output_dir+=os.sep
	dir_exists=os.access(output_dir, os.F_OK)
	if not dir_exists:
		os.mkdir(output_dir)

	# Loop through the files and combine them into a single cube
	parallactic_angle_vec=[]
	for ix,allfiles in enumerate(dirlist):
		if ix==0:
			master_cube,hdr=pyfits.getdata(allfiles,header=True)
		else:
			cube_temp=pyfits.getdata(allfiles)
			master_cube=np.append(master_cube,cube_temp,axis=0)

		# Read in the cube info file to get the parallactic angles
		with open(glob.glob("cube-info/*"+allfiles.replace(".fits",".rdb"))[0],'r') as f:
			lines=f.readlines()

		for line in lines:
			parallactic_angle=line.strip().split()[11]
			if ((not "paralactic_angle" in parallactic_angle) and (not "---" in parallactic_angle)):
				parallactic_angle_vec=np.append(parallactic_angle_vec,parallactic_angle)

	if collapse_cube:
		master_cube = np.mean(master_cube,axis=0)
	
	# Write the output file with all of the frames in the cube
	# If there's only 1 file, just copy it rather than saving it with pyfits
	# This will be much quicker
	if len(dirlist) ==1 and not collapse_cube:
		shutil.copy(dirlist[0],output_dir+output_file)
	else:
		pyfits.writeto(output_dir+output_file,master_cube,header=hdr,clobber=True)

	# Write an output file with the parallactic angles
	if not skip_parang:
		with open(output_dir+"parallactic_angle.txt",'w') as f2:
			for i,parallactic_angle in enumerate(parallactic_angle_vec):
				f2.write(parallactic_angle+"\n")

	sys.stdout.write("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
	sys.stdout.write("\n")
	sys.stdout.write("end of convert\n")
	sys.stdout.flush()


	sys.exit(0)
else:
	sys.exit(0)

