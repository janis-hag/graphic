#!/usr/bin/python
import numpy as np
import astropy.io.fits as pyfits
import sys
from mpi4py import MPI
import graphic_nompi_lib_330 as graphic_nompi_lib
import argparse

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Supress the frame in the cubes and rdb files from the selection_frame file')
parser.add_argument('--pattern', action="store", dest="pattern",  default="sdi", help='cubes to apply the frame selections')
parser.add_argument('--info_pattern', action="store", dest="info_pattern", required=True, help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",  default='cube-info', help='Info directory')
args = parser.parse_args()
pattern=args.pattern
info_pattern=args.info_pattern
info_dir=args.info_dir


if rank==0:
	def frame_selec(frame_sel_filename,pattern):
		f=open(frame_sel_filename,'r')
		lines=f.readlines()
		f.close()

		#path_cube_info="cube-info/"
		#all_info_pattern="all_info_20_5_300_"
		path_cube_info=info_dir+"/"
		all_info_pattern=info_pattern

		for line in lines:
		    sys.stdout.write('\n'+line.strip().split()[0])
		    sys.stdout.flush()
		    if line.strip().split()[0]!='filename' and line.strip().split()[0]!='--------':
			filename=line.strip().split()[0]
			frame_to_del=line.strip().split("\t")[1]
			print 'frame_to_del:',frame_to_del
			frame_to_del=np.array(eval(frame_to_del))
		
			if "sdi" in pattern:
				if "sdi" in filename:
					cube,hdr=pyfits.getdata(filename,header=True)
				elif "left" in filename:
					filename=filename.replace("left","sdi")
				elif "right" in filename:
					filename=filename.replace("right","sdi")
			elif "left" in pattern:
				if "left" in filename:
					cube,hdr=pyfits.getdata(filename,header=True)
				elif "sdi" in filename:
					filename=filename.replace("sdi","left")
				elif "right" in filename:
					filename=filename.replace("right","left")
			elif "right" in pattern:
				if "right" in filename:
					cube,hdr=pyfits.getdata(filename,header=True)
				elif "sdi" in filename:
					filename=filename.replace("sdi","right")
				elif "left" in filename:
					filename=filename.replace("left","right")
			print "filename",filename
			cube,hdr=pyfits.getdata(filename,header=True)
			f=open(path_cube_info+all_info_pattern+filename.replace(".fits",".rdb"))
			lines=f.readlines()
			f.close()
		
			index_keep=np.arange(np.shape(cube)[0])
			counter1=0
			counter2=0
			if np.size(frame_to_del)!=0:
				index_del=index_keep[frame_to_del]
				index_keep=np.delete(index_keep,frame_to_del)
			else:
				index_del=np.array([])
				counter1=99999
			table=[]
			for line in lines:
			    if np.size(index_del)==0:
			        table=np.append(table,line)
			    elif counter1!=index_del[counter2]+2:
				table=np.append(table,line)
			    elif counter2<np.size(index_del)-1:
				counter2+=1
			    else:
				counter1=99999
			    counter1+=1
			# print 'Keeping:',index_keep
		
			cube=cube[index_keep,:,:]
		
			hdr['NAXIS3']=np.shape(cube)[0]
		
			pyfits.writeto("frame_sel_"+filename,cube,header=hdr,clobber=True)
			filename3=path_cube_info+all_info_pattern+"frame_sel_"+filename.replace(".fits",".rdb")
			f=open(filename3,'w')
			for i in range(np.size(table)):
			    f.write(table[i])
			f.close()

	frame_sel_filename="frame_selection.txt"
	#print "ERROR: No files for frame selection detected!"

	t0=MPI.Wtime()
	print("beginning of frame selection")
	frame_selec(frame_sel_filename,pattern)


	print("Total time: "+graphic_nompi_lib.humanize_time((MPI.Wtime()-t0)))
	print("frame selection finished")
	sys.exit(0)
else:
	sys.exit(0)
