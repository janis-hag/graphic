import numpy as np
import scipy, pyfits
from scipy import fftpack
import sys, glob
from mpi4py import MPI
import argparse

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
parser = argparse.ArgumentParser(description='Supress the frame in the cubes and rdb files from the selection_frame file')
parser.add_argument('--pattern', action="store", dest="pattern",  default="sdi", help='cubes to apply the frame selections')
args = parser.parse_args()
pattern=args.pattern


if rank==0:
	def mad(arr):
	    """ Median Absolute Deviation: a "Robust" version of standard deviation.
		Indices variabililty of the sample.
		https://en.wikipedia.org/wiki/Median_absolute_deviation 
	    """
	    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
	    med = np.median(arr)
	    return np.median(np.abs(arr - med))

	def frame_selection(filename):
	    cut=80
	    cube,hdr=pyfits.getdata(filename,header=True)
	    
	    cube=cube[:,np.shape(cube)[1]/2-cut:np.shape(cube)[1]/2+cut,np.shape(cube)[1]/2-cut:np.shape(cube)[1]/2+cut]

	    med=np.nanmedian(cube,axis=0)
	    cube=cube-med+np.median(med)
	    #cube=cube2-np.median(np.min(cube2,axis=0))

	    x=np.arange(-np.shape(cube)[1]/2,np.shape(cube)[1]/2)
	    y=np.arange(-np.shape(cube)[1]/2,np.shape(cube)[1]/2)
	    X,Y=np.meshgrid(x,y)
	    R=np.sqrt(X**2+Y**2)
	    r_int=30
	    r_ext=80
	    donut=np.where(R<r_int,np.nan,1)
	    donut=np.where(R>r_ext,np.nan,donut)

	    cube=cube*donut
	    #cube2=cube2*donut

	    #pyfits.writeto("cube_sub.fits",cube,clobber=True)
	    quad1=cube[:,:np.shape(cube)[1]/2,:np.shape(cube)[2]/2]
	    quad2=cube[:,np.shape(cube)[1]/2:,:np.shape(cube)[2]/2]
	    quad3=cube[:,:np.shape(cube)[1]/2,np.shape(cube)[2]/2:]
	    quad4=cube[:,np.shape(cube)[1]/2:,np.shape(cube)[2]/2:]

	    med_cube=np.zeros(np.shape(cube)[0])

	    sum1=np.zeros(np.shape(quad1)[0])
	    sum2=np.zeros(np.shape(quad1)[0])
	    sum3=np.zeros(np.shape(quad1)[0])
	    sum4=np.zeros(np.shape(quad1)[0])



	    for i in range(np.shape(cube)[0]):
		med_cube[i]=np.nanmedian(cube[i,:,:])
	    
		sum1[i]=np.nansum(quad1[i,:,:])
		sum2[i]=np.nansum(quad2[i,:,:])
		sum3[i]=np.nansum(quad3[i,:,:])
		sum4[i]=np.nansum(quad4[i,:,:])
	    

	    bad_frames=[]
	    for i in range(np.shape(cube)[0]):
		index_quad_min=np.where(np.array([sum1[i],sum2[i],sum3[i],sum4[i]])==np.min([sum1[i],sum2[i],sum3[i],sum4[i]]))[0][0]
		if index_quad_min==0:
		    quad_min=quad1[i]
		elif index_quad_min==1:
		    quad_min=quad2[i]
		elif index_quad_min==2:
		    quad_min=quad3[i]
		elif index_quad_min==3:
		    quad_min=quad4[i]
		mean_quad_min=np.nanmean(quad_min)
		sum_quad_mean=np.nansum(quad_min)
		std_quad_min=np.nanstd(quad_min)
		sigma=3
		if np.nansum(np.where(quad1[i]>mean_quad_min+3*std_quad_min,quad1[i],np.nan))>0.1*sum_quad_mean:
		    bad_frames=bad_frames+[int(i+1)]
		elif np.nansum(np.where(quad2[i]>mean_quad_min+3*std_quad_min,quad2[i],np.nan))>0.1*sum_quad_mean:
		    bad_frames=bad_frames+[int(i+1)]
		elif np.nansum(np.where(quad3[i]>mean_quad_min+3*std_quad_min,quad3[i],np.nan))>0.1*sum_quad_mean:
		    bad_frames=bad_frames+[int(i+1)]
		elif np.nansum(np.where(quad4[i]>mean_quad_min+3*std_quad_min,quad4[i],np.nan))>0.1*sum_quad_mean:
		    bad_frames=bad_frames+[int(i+1)]
		elif med_cube[i]>np.median(med_cube)+(sigma+2)*mad(med_cube):
		    bad_frames=bad_frames+[int(i+1)]
	    return bad_frames

	#pattern="left_cl_nomed*SCIENCE"
	length=0

	for allfiles in glob.iglob(pattern+'*'):
	    sys.stdout.write(allfiles)
	    sys.stdout.write('\n')
	    #sys.stdout.write(allfiles.replace('left','right'))
	    #sys.stdout.write('\n')
	    length+=1
	sys.stdout.flush()

	bad_frames_cube=range(length)

	for i,allfiles in enumerate(glob.iglob(pattern+'*')):
	    bad_frames_cube[i]=frame_selection(allfiles)

	f=open('frame_selection.txt','w')
	f.write("filename\tframe_to_delete\n")
	f.write("--------\t---------------\n")
	for i,allfiles in enumerate(glob.iglob(pattern+'*')):
	    f.write(allfiles+'\t'+str(bad_frames_cube[i])+'\n')
	f.close()
	
	sys.exit(0)
else:
	sys.exit(0)



