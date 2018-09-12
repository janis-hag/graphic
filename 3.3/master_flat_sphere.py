#!/usr/bin/python
import numpy as np
import scipy, glob, sys, os
import astropy.io.fits as pyfits
#import pyfits
#import graphic_lib_330
import argparse
from mpi4py import MPI

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Create a master flat from flatfields of different exposure time')
parser.add_argument('--pattern', action="store", dest="pattern",  default="SPHER*FLAT_FIELD", help='cubes to use to create the master flat')

args = parser.parse_args()
pattern=args.pattern

if rank==0:
	def master_flat_sphere(key_word):
		"""
		produce a master_flat by combining flats with different exposure times and taking the slope. To achieve this it resolves the matrix linear system.
		created by Sebastien Peretti
		"""
		count=0
		hdulist=pyfits.PrimaryHDU()
		hdr_masterflat = hdulist.header
		hdr_masterflat['Routine used'] = "master_flat_func routine from Sebastien P"  # Add a new keyword
		for allfiles in glob.iglob(key_word+'*'):
			print(allfiles)
			hdr_masterflat['flat nbr '+np.str(count+1)+' used'] = allfiles  # Add a new keyword
			if count==0:
				temp,hdr=pyfits.getdata(allfiles, header=True)
				#print "shape(temp)",np.shape(temp)
				y=np.ones(((1,np.shape(temp)[-2],np.shape(temp)[-1])))
				#print "shape(y)",np.shape(y)
				if np.size(np.shape(temp))>2:
					if np.shape(temp)[0]>1:
						y[0]=np.median(temp,axis=0)
					else:
						y[0]=temp[0]
				else:
					y[0]=temp
				time_exp=np.array(hdr["HIERARCH ESO DET SEQ1 DIT"])
			else:
				temp,hdr=pyfits.getdata(allfiles, header=True)
				if np.size(np.shape(temp))>2:
					if np.shape(temp)[0]>1:
						temp[0]=np.median(temp,axis=0)
						temp=temp[:1,:,:]
				else:
					temp=np.zeros(((1,np.shape(temp)[0],np.shape(temp)[1])))+temp
				y=np.append(y,temp,axis=0)
				time_exp=np.append(time_exp,hdr["HIERARCH ESO DET SEQ1 DIT"])
			hdr_masterflat['Dit for flat nbr '+np.str(count+1)] = (hdr["HIERARCH ESO DET SEQ1 DIT"], "s.") # Add a new keyword
			count+=1

		shape_y=np.shape(y)
		x = np.append(np.ones((np.size(time_exp),1)),np.reshape(np.array(time_exp),(np.size(time_exp),1)),axis=1)
		y=np.reshape(y,(np.shape(y)[0],shape_y[1]*shape_y[2]))
		a,c=np.linalg.lstsq(x, y)[0]
		master_flat=np.reshape(c,(shape_y[1],shape_y[2]))
		master_flat=master_flat/np.median(master_flat)
		pyfits.writeto("master_flat.fits",master_flat,header=hdr_masterflat,output_verify='warn',clobber=True)
		sys.stdout.write("saving flats")
		sys.stdout.flush()


	sys.stdout.write('beginning of a master flat creation\n')
	sys.stdout.flush()

	master_flat_sphere(pattern)

	sys.stdout.write("\nmaster flat created -> end of master flat\n")
	sys.stdout.flush()
	sys.exit(0)
else:
	sys.exit(0)
