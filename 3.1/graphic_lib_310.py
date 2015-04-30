"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of the Parallel Angular Differential Imaging Pipeline (PADIP).
These are function common to different programs of the pipeline.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""

import numpy, os, shutil, sys, glob, math
import numpy as np
## from scipy.signal import correlate2d
from mpi4py import MPI
from gaussfit_310 import fitgaussian
from scipy import ndimage, fftpack
#import astropy.io.fits as pyfits
from astropy.io import fits

## sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
## import bottleneck

nprocs = MPI.COMM_WORLD.Get_size()
rank   = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD


def bcast_window_coord(x0=None,y0=None,xs=None,xe=None,ys=None,ye=None,ref_window=None):
	'''
	Get broadcast the reference window limits and the reference window

	To send x0,y0,xs,xe,ys,ye,ref_window call bcast_window_coord(x0=x0,y0=y0,xs=xs,xe=xe,ys=ys,ye=ye,ref_window=ref_window)
	To receive x0,y0,xs,xe,ys,ye,ref_window call x0,y0,xs,xe,ys,ye,ref_window=bcast_window_coord()

	Returns x0,y0,xs,xe,ys,ye,ref_window
	'''

	# Broadcast the reference position, reference window
	x0 = comm.bcast(x0, root=0)
	y0 = comm.bcast(y0, root=0)
	# Broadcast the reference window limits and the reference window
	xs = comm.bcast(xs, root=0)
	xe = comm.bcast(xe, root=0)
	ys = comm.bcast(ys, root=0)
	ye = comm.bcast(ye, root=0)
	ref_window = comm.bcast(ref_window, root=0)

	return x0,y0,xs,xe,ys,ye,ref_window


## def calc_parang(hdr):
##	 """
##	 Read a header and calculates the paralactic angle
##	 """
##	 from numpy import sin, cos, tan, arctan, pi
##	 geolat_rad=float(hdr['HIERARCH ESO TEL GEOLAT'])*pi/180.
##	 dec_rad=float(hdr['DEC'])*pi/180.
##	 ra_seconds=float(hdr['RA'])*3600./15.
##	 ha_seconds=float(hdr['LST'])-ra_seconds
##	 ha_rad=(ha_seconds*15.*pi)/(3600*180)
##	 parang_deg=arctan((-sin(ha_rad)*cos(ha_rad))/sin(dec_rad)-cos(dec_rad)*tan(geolat_rad))*180/pi

##	 return parang_deg

def calc_parang(hdr):
	"""
	Read a header and calculates the paralactic angle, using method derived by Arthur Vigan
	"""
	from numpy import sin, cos, tan, arctan, pi

	r2d = 180/pi
	d2r = pi/180

	ra_deg = float(hdr['RA'])
	dec_deg = float(hdr['DEC'])

	geolat_rad=float(hdr['HIERARCH ESO TEL GEOLAT'])*d2r #*r2d

	ha_deg=(float(hdr['LST'])*15./3600)-ra_deg

	# VLT TCS formula
	f1 = cos(geolat_rad) * sin(d2r*ha_deg)
	f2 = sin(geolat_rad) * cos(d2r*dec_deg) - cos(geolat_rad) * sin(d2r*dec_deg) * cos(d2r*ha_deg)

	parang_deg=r2d*arctan(f1/f2)

	return parang_deg

def chi_diff(shift, r_im, s_im ,x0, y0, r, R):
	"""
	Calculate the difference between the reference window (rw) and the shifted image (shift_im)

	-shift tuple containing x and y shift to apply
	-xs,ys x position of the reference window start
	-xe,ye y position of the reference window end
	-r inner annulus radius
	-R outer annulus radius
	"""

	s=5

	if x0+R+s+1 >= s_im.shape[0]:
		for i in range(s,0):
		   if x0+R+i+1 <= s_im.shape[0]:
			   s=1
			   break
		   elif i==0:
			   print("Error, star at edge of frame! "+str(x0)+" "+str(y0))

	elif x0-R-s <= 0:
		for i in range(s,0):
			if x0-R-i >= 0:
				s=i
				break
			elif i==0:
				print("Error, star at edge of frame! "+str(x0)+" "+str(y0))

	if y0+R+s+1 >= s_im.shape[1]:
		for i in range(s,0):
			if x0+R+i+1 <= s_im.shape[1]:
				s=i
				break
			elif i==0:
				print("Error, star at edge of frame! "+str(x0)+" "+str(y0))

	elif y0-R-s <= 0:
		for i in range(s,0):
			if x0-R-i >= 0:
				s=i
				break
			elif i==0:
				print("Error, star at edge of frame! "+str(x0)+" "+str(y0))

	# Cut window in reference image
	# rw = r_im[x0-R:x0+R+1,y0-R:y0+R+1]

	# Generate annulus mask
	annulus=numpy.zeros((2*R+1,2*R+1))

	for x in range(-R,R+1):
		for y in range(-R,R+1):
			if (x**2+y**2) <= R**2 and (x**2+y**2) >= r**2:
				annulus[x+R+x0][y+R+y0]=1

	chi2=numpy.zeros(11,11)
	for xs in range(-s,s+1):
		for ys in range(-s,s+1):
			iw=s_im[x0-R+xs:x0+R+1+xs,y0-R+ys:y0+R+1+ys] # Sliding search window
			chi2[xs,ys]=sum((rw[annulus]-iw[annulus])**2)

	## best=numpy.where(chi2==numpy.max(chi2))
	## level=numpy.median(chi2)

	params=fitgaussian(chi2)

	return params

def cluster_search(image, thres_coef, min, max, x_i, y_i, d=0):
	"""Search pixel above threshold by running along the image.

	Warning: python inverts x and y
	Threshold could be given as argument...
	-im: the image to be analysed
	-spot_ary: the 2d array of spots [psf_barycenter_x, psf_barycenter_y, psf_pixel_size]
	"""
	import sys, numpy

	ima=image.copy()
	from sys import setrecursionlimit
	setrecursionlimit(25000)

	check_ima=numpy.zeros_like(1.*ima) #Create an image for checking
	cl_cnt=0	#cluster counter
	spot_ary=None
	thres=thres_coef # using thres_coef directly as threshold value because mean is nearly equal 0
	smallest=0
	biggest=ima.size

	if ima[x_i , y_i] > thres: #check if above thresh
		xpix_cnt = x_i
		ypix_cnt = y_i
		new_cluster=numpy.array([]) # create a new empty cluster array
		new_cluster=cluster_build(xpix_cnt,ypix_cnt,ima,new_cluster,thres)
		#remove cluster from img to prevent double detection, by
		#check the size of the spot
		if new_cluster.size < min:
			if smallest<new_cluster.size:
				smallest=new_cluster.size
			pass
		elif new_cluster.size > max:
			if biggest>new_cluster.size:
			   biggest=new_cluster.size
			pass
		# find the centroid of the cluster and add the position to
		# spot_ary
		# compute total luminosity of the spot
		else:
			tot_lum=numpy.sum(numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))
			# find the xcenter of mass
			xcenter=numpy.sum(new_cluster[i,0]*numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))/tot_lum
			# find the ycenter of mass
			ycenter=numpy.sum(new_cluster[i,1]*numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))/tot_lum
			#check if it is the first spot
			if spot_ary is None: spot_ary=numpy.array([xcenter,ycenter,new_cluster.size]) # initialise
			elif new_cluster.size>spot_ary[2]:
				# More than one centroid detected. Keep only the biggest.
				#spot_ary=numpy.vstack((spot_ary,[xcenter,ycenter,new_cluster.size]))
				dprint(d>0,'Found bigger spot, old: '+str(spot_ary)+', new: '+str([xcenter,ycenter,new_cluster.size]))
				spot_ary=numpy.array([xcenter,ycenter,new_cluster.size])
			if  check_ima[numpy.round(xcenter),numpy.round(ycenter)]==0:
				check_ima[numpy.round(xcenter),numpy.round(ycenter)]=tot_lum #Add a point on the checking image
			else:
				check_ima[numpy.round(xcenter)+1,numpy.round(ycenter)+1]=tot_lum
				print "Double detection"
			cl_cnt=cl_cnt+1
	else:
		for ypix_cnt in range(ima.shape[1]):
			for xpix_cnt in range(ima.shape[0]):
				if ima[xpix_cnt , ypix_cnt] > thres: #check if above thresh
					new_cluster=numpy.array([]) # create a new empty cluster array
					new_cluster=cluster_build(xpix_cnt,ypix_cnt,ima,new_cluster,thres)
					#remove cluster from img to prevent double detection, by
					#check the size of the spot
					if new_cluster.size < min:
						if smallest<new_cluster.size:
							smallest=new_cluster.size
						continue
					if new_cluster.size > max:
						if biggest>new_cluster.size:
							biggest=new_cluster.size
						continue
						#do some thing.....
					#find the centroid of the cluster and add the position to
					#spot_ary
					# compute total luminosity of the spot
					tot_lum=numpy.sum(numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))
					# find the xcenter of mass
					xcenter=numpy.sum(new_cluster[i,0]*numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))/tot_lum
					 # find the ycenter of mass
					ycenter=numpy.sum(new_cluster[i,1]*numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))/tot_lum
					#check if it is the first spot
					if spot_ary is None: spot_ary=numpy.array([xcenter,ycenter,new_cluster.size]) # initialise
					elif new_cluster.size>spot_ary[2]:
						# More than one centroid detected. Keep only the biggest.
						#spot_ary=numpy.vstack((spot_ary,[xcenter,ycenter,new_cluster.size]))
						dprint(d>0,'Found bigger spot, old: '+str(spot_ary)+', new: '+str([xcenter,ycenter,new_cluster.size]))
						spot_ary=numpy.array([xcenter,ycenter,new_cluster.size])
					if  check_ima[numpy.round(xcenter),numpy.round(ycenter)]==0:
						check_ima[numpy.round(xcenter),numpy.round(ycenter)]=tot_lum #Add a point on the checking image
					else:
						check_ima[numpy.round(xcenter)+1,numpy.round(ycenter)+1]=tot_lum
						print "Double detection"
					cl_cnt=cl_cnt+1
			else: continue
	if cl_cnt==0:
		print(" No centroids found. Clusters closest to size limits: "+str(smallest)+" < [ "+str(min)+" : "+str(max)+" ] < "+str(biggest)+" \n")
		## if not smallest==0:
		##	 sys.stdout.write("Biggest candidate below limit size: "+str(smallest)+".\n")
		## if not biggest==ima.size:
		##	  sys.stdout.write("Smallest candidate above size: "+str(biggest)+".\n")
		# sys.stdout.write("\n")
		sys.stdout.flush()
	elif cl_cnt>1:
		print(" Multiple centroids found. Clusters closest to size limits: "+str(smallest)+" < [ "+str(min)+" : "+str(max)+" ] < "+str(biggest)+" \n")
		dprint(d>0,str(spot_ary))
		## if not smallest==0:
		##	 sys.stdout.write("Biggest candidate below limit size: "+str(smallest)+".\n")
		## if not biggest==ima.size:
		##	  sys.stdout.write("Smallest candidate above size: "+str(biggest)+".\n")
		# sys.stdout.write("\n")
		sys.stdout.flush()

	return spot_ary, ima, cl_cnt

def cluster_search_multi(image, thres_coef, min, max, x_i, y_i):
	"""Search pixel above threshold by running along the image.

	Warning: python inverts x and y
	Threshold could be given as argument...
	-im: the image to be analysed
	-spot_ary: the 2d array of spots [psf_barycenter_x, psf_barycenter_y, psf_pixel_size]
	"""
	import sys, numpy

	ima=image.copy()
	from sys import setrecursionlimit
	setrecursionlimit(25000)

	check_ima=numpy.zeros_like(1.*ima) #Create an image for checking
	cl_cnt=0	#cluster counter
	spot_ary=None
	thres=thres_coef # using thres_coef directly as threshold value because mean is nearly equal 0
	smallest=0
	biggest=ima.size

	if ima[x_i , y_i] > thres: #check if above thresh
		xpix_cnt = x_i
		ypix_cnt = y_i
		new_cluster=numpy.array([]) # create a new empty cluster array
		new_cluster=cluster_build(xpix_cnt,ypix_cnt,ima,new_cluster,thres)
		#remove cluster from img to prevent double detection, by
		#check the size of the spot
		if new_cluster.size < min:
			if smallest<new_cluster.size:
				smallest=new_cluster.size
			pass
		elif new_cluster.size > max:
			if biggest>new_cluster.size:
			   biggest=new_cluster.size
			pass
		# do some thing.....
		# find the centroid of the cluster and add the position to
		# spot_ary
		# compute total luminosity of the spot
		else:
			tot_lum=numpy.sum(numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))
			# find the xcenter of mass
			xcenter=numpy.sum(new_cluster[i,0]*numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))/tot_lum
			# find the ycenter of mass
			ycenter=numpy.sum(new_cluster[i,1]*numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))/tot_lum
			#check if it is the first spot
			if spot_ary is None: spot_ary=numpy.array([xcenter,ycenter,new_cluster.size]) # initialise
			else: spot_ary=numpy.vstack((spot_ary,[xcenter,ycenter,new_cluster.size]))
			if  check_ima[numpy.round(xcenter),numpy.round(ycenter)]==0:
				check_ima[numpy.round(xcenter),numpy.round(ycenter)]=tot_lum #Add a point on the checking image
			else:
				check_ima[numpy.round(xcenter)+1,numpy.round(ycenter)+1]=tot_lum
				print "Double detection"
			cl_cnt=cl_cnt+1
	else:
		for ypix_cnt in range(ima.shape[1]):
			for xpix_cnt in range(ima.shape[0]):
				if ima[xpix_cnt , ypix_cnt] > thres: #check if above thresh
					new_cluster=numpy.array([]) # create a new empty cluster array
					new_cluster=cluster_build(xpix_cnt,ypix_cnt,ima,new_cluster,thres)
					#remove cluster from img to prevent double detection, by
					#check the size of the spot
					if new_cluster.size < min:
						if smallest<new_cluster.size:
							smallest=new_cluster.size
						continue
					if new_cluster.size > max:
						if biggest>new_cluster.size:
							biggest=new_cluster.size
						continue
						#do some thing.....
					#find the centroid of the cluster and add the position to
					#spot_ary
					# compute total luminosity of the spot
					tot_lum=numpy.sum(numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))
					# find the xcenter of mass
					xcenter=numpy.sum(new_cluster[i,0]*numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))/tot_lum
					 # find the ycenter of mass
					ycenter=numpy.sum(new_cluster[i,1]*numpy.abs(ima[new_cluster[i,0],new_cluster[i,1]]) for i in range(new_cluster.shape[0]))/tot_lum
					#check if it is the first spot
					if spot_ary is None: spot_ary=numpy.array([xcenter,ycenter,new_cluster.size]) # initialise
					else: spot_ary=numpy.vstack((spot_ary,[xcenter,ycenter,new_cluster.size]))
					if  check_ima[numpy.round(xcenter),numpy.round(ycenter)]==0:
						check_ima[numpy.round(xcenter),numpy.round(ycenter)]=tot_lum #Add a point on the checking image
					else:
						check_ima[numpy.round(xcenter)+1,numpy.round(ycenter)+1]=tot_lum
						print "Double detection"
					cl_cnt=cl_cnt+1
			else: continue
	if cl_cnt==0:
		sys.stdout.write(" No centroids found. Clusters closest to size limits: "+str(smallest)+" < [ "+str(min)+" : "+str(max)+" ] < "+str(biggest)+" \n")
		## if not smallest==0:
		##	 sys.stdout.write("Biggest candidate below limit size: "+str(smallest)+".\n")
		## if not biggest==ima.size:
		##	  sys.stdout.write("Smallest candidate above size: "+str(biggest)+".\n")
		# sys.stdout.write("\n")
		sys.stdout.flush()
	elif cl_cnt>1:
		sys.stdout.write(" Multiple centroids found. Clusters closest to size limits: "+str(smallest)+" < [ "+str(min)+" : "+str(max)+" ] < "+str(biggest)+" \n")
		sys.stdout.write(str(spot_ary))
		## if not smallest==0:
		##	 sys.stdout.write("Biggest candidate below limit size: "+str(smallest)+".\n")
		## if not biggest==ima.size:
		##	  sys.stdout.write("Smallest candidate above size: "+str(biggest)+".\n")
		# sys.stdout.write("\n")
		sys.stdout.flush()

	return spot_ary, ima, cl_cnt

def cluster_build(xpos,ypos,image,cl_ary,thr):
	"""Build a cluster at xpos, ypos, sourced in image.

	Build a cluster by adding all pixel in ima above threshold in contact
	with the starting pixel at (xpos,ypos). The cluster number is given by
	cl_num.
	Arguments:
	-xpos, ypos: x and y position of the starting pixel in the image
	-ima: the array containing the image
	-cl_ary: the array of cluster this cluster should be added to
	should rather be the cluster array
	-thr: threshold
	"""
	from sys import setrecursionlimit
	## setrecursionlimit(50000)
	setrecursionlimit(25000)

	if image[xpos,ypos] > thr:
		#negate the pixel value to prevent double detection and infinite loops
		image[xpos,ypos]=-image[xpos,ypos]
		if cl_ary.size > 0: # see if cl_ary has already been filled
			cl_ary=numpy.vstack((cl_ary,[xpos,ypos]))
		else: cl_ary=numpy.array([xpos,ypos]) #initiate the array
		#print cl_ary.shape
		#check if contiguous pixel still in the image, then search in this pixel
		if xpos-1 in range(image.shape[0]):
			cl_ary=cluster_build(xpos-1,ypos,image,cl_ary,thr) # left
		if xpos+1 in range(image.shape[0]):
			cl_ary=cluster_build(xpos+1,ypos,image,cl_ary,thr) # right
		if ypos-1 in range(image.shape[1]):
			cl_ary=cluster_build(xpos,ypos-1,image,cl_ary,thr) # up
		if ypos+1 in range(image.shape[1]):
			cl_ary=cluster_build(xpos,ypos+1,image,cl_ary,thr) # down
	return cl_ary

def create_dirlist(pattern, target_dir='.', extension='.fits', target_pattern=None, interactive=False):
	"""
	Generate a dirlist and checks for file acces rights
	"""
	# import glob
	import string

	dirlist=glob.glob(pattern+'*'+extension)
	dirlist.sort() # Sort the list alphabetically

	if len(dirlist)>0:
		iprint(interactive, '\r\r\r Found '+str(len(dirlist))+' files.\n')
		## sys.stdout.write("\r\r\r Found "+str(len(dirlist))+" files.\n")
		## sys.stdout.flush()
	else:
		return None

	# Check values in dirlist and remove dodgy files.
	for i in range(len(dirlist)):
		if not os.access(dirlist[i], os.F_OK | os.R_OK): # Check if file exists
			## print("")
			print(str(rank)+': Error, cannot access: '+dirlist[i])
			dirlist[i]=None
			continue
		if not target_pattern==None:
			if os.access(target_dir+os.sep+target_pattern+string.split(dirlist[i], os.sep)[-1], os.F_OK | os.R_OK): # Check if file exists
				print(dirlist[i]+' already processed.')
				dirlist[i]=None
				continue

	# Clean dirlist of discarded values:
	dirlist.sort()
	skipped=0
	for i in range(len(dirlist)):
		if dirlist[0]==None:
			dirlist.pop(0)
			skipped=skipped+1
		else: break
	if skipped>0:
		print(" Skipped "+str(skipped)+" files.")

	if len(dirlist)==0:
		return None
	else:
		return dirlist

def create_megatable(dirlist,infolist,skipped=None,keys=None, nici=False, fit=True, nonan=True, interactive=False):
	"""
	Read all the all_info tables an create a mega-table adding a column with filename
	for filename in dirlist search coresponding all_info table
	add filename column
	stack the tables
	"""
	import fnmatch, os, tables, sys

	if skipped==None:
		skip=0
	else:
		skip=skipped

	key=['cube_filename','info']
	cube_list = {}
	for i in range(len(key)): cube_list[key[i]] = []

	for i in range(len(dirlist)):
		if nici:
			info_filename = fnmatch.filter(infolist,'*'+dirlist[i][-19:-5]+'*')
		else:
			info_filename = fnmatch.filter(infolist,'*'+dirlist[i][-28:-5]+'*')


		if len(info_filename)==0:
			if nici:
				print("\n"+str(rank)+": No centroids list found for "+dirlist[i][-19:-5])
			else:
				print("\n"+str(rank)+": No centroids list found for "+dirlist[i][-28:-5])
			skip=skip+1
			dirlist[i]=None
			continue
		elif not len(info_filename)==1:
			## print("")
			print("\n"+str(rank)+": More than one centroids list found for "+dirlist[i])
			print(str(rank)+": Using first occurence: "+info_filename[0])
		if not os.access(info_filename[0], os.F_OK | os.R_OK): # Check if file exists
			print("\n"+str(rank)+': Error, cannot access: '+info_filename[0])
			## sys.stdout(str(rank)+'\n Error, cannot access: '+centroname[0])
			## sys.stdout.flush()
			skip=skip+1
			dirlist[i]=None
			continue

		iprint(interactive,"\r\r\r Reading info files "+str(i+1)+"/"+str(len(dirlist)))
		## sys.stdout.flush()
		# Read centroids list
		if info_filename[0][-3:]=='rdb':
			rdb_info=read_rdb(info_filename[0])
			if fit:
				rdb_info['psf_barycenter_x']=rdb_info['psf_fit_center_x']
				rdb_info['psf_barycenter_y']=rdb_info['psf_fit_center_y']
			else:
				rdb_info['psf_fit_center_x']=rdb_info['psf_barycenter_x']
				rdb_info['psf_fit_center_y']=rdb_info['psf_barycenter_y']
			all_info=np.array(rdb_info['frame_number'])
			for k in keys[1:]:
				all_info=np.vstack((all_info,np.array(rdb_info[k])))
			all_info=np.swapaxes(all_info,0,1)

		else:
			f=tables.openFile(info_filename[0])
			all_info=f.root.centroids.read()
			f.close()

		if len(all_info)==0 and not np.all(all_info[:,1]==1) and not np.all(np.isnan(all_info[:,1])): # Check if there is any valid data
			dirlist[i]=None
		else:
			## print(len(all_info))
			cube_list['cube_filename'].append(dirlist[i])
			if nonan:
				all_info=np.where(np.isnan(all_info),-1,all_info)
			cube_list['info'].append(all_info)

	dirlist_clean=[]
	for i in range(len(dirlist)):
		if not dirlist[i]==None:
			dirlist_clean.append(dirlist[i])
	## dirlist=dirlist_clean

	sys.stdout.write("\n")
	sys.stdout.flush()
	if skipped==None:
		return cube_list, dirlist_clean
	else:
		return cube_list, dirlist_clean, skip

def create_parang_list(hdr):
	"""
	Reads the header and creates an array giving the paralactic angle for each frame

	The columns of the output array contains:
	frame_number, frame_time, paralactic_angle
	"""

	from numpy import sin, cos, tan, arctan, pi

	r2d = 180/pi
	d2r = pi/180

	ra_deg = float(hdr['RA'])
	dec_deg = float(hdr['DEC'])

	geolat_rad=float(hdr['ESO TEL GEOLAT'])*r2d

	dit=float(hdr['ESO DET DIT'])
	dit_delay=0
	if 'ESO DET DITDELAY' in hdr.keys():
		dit_delay=float(hdr['ESO DET DITDELAY'])
	## else:
		## sys.stdout.write('\n Warning! No HIERARCH ESO DET DITDELAY keyword found, using 0. Is it ADI?\n')
		## sys.stdout.flush()

	ha_deg=(float(hdr['LST'])*15./3600)-ra_deg

	# VLT TCS formula
	f1 = cos(geolat_rad) * sin(d2r*ha_deg)
	f2 = sin(geolat_rad) * cos(d2r*dec_deg) - cos(geolat_rad) * sin(d2r*dec_deg) * cos(d2r*ha_deg)

	parang_array=numpy.array([0,float(hdr['LST']),r2d*arctan(f1/f2)])

	for i in range(1,hdr['NAXIS3']):
		ha_deg=((float(hdr['LST'])+i*(dit+dit_delay))*15./3600)-ra_deg

		# VLT TCS formula
		f1 = cos(geolat_rad) * sin(d2r*ha_deg)
		f2 = sin(geolat_rad) * cos(d2r*dec_deg) - cos(geolat_rad) * sin(d2r*dec_deg) * cos(d2r*ha_deg)

		parang_array=numpy.vstack((parang_array,[i,float(hdr['LST'])+i*(dit+dit_delay),r2d*arctan(f1/f2)]))

	return parang_array

def create_parang_list_ada(hdr):
	"""
	Reads the header and creates an array giving the paralactic angle for each frame,
	taking into account the inital derotator position.

	The columns of the output array contains:
	frame_number, frame_time, paralactic_angle
	"""

	from numpy import sin, cos, tan, arctan, pi
	import dateutil.parser

	r2d = 180/pi
	d2r = pi/180

	ra_deg = float(hdr['RA'])
	dec_deg = float(hdr['DEC'])

	geolat_rad=float(hdr['ESO TEL GEOLAT'])*d2r
	if 'ESO ADA PUPILPOS' in hdr.keys():
		ADA_PUPILPOS=float(hdr['ESO ADA PUPILPOS'])
	else:
		ADA_PUPILPOS=89.44
		sys.stdout.write("\n Warning! No ESO ADA PUPILPOS keyword found, using 89.44. Is it ADI?\n")
		sys.stdout.flush()

	ROT_PT_OFF=179.44-ADA_PUPILPOS # from NACO manual v90 p.85

	dit=float(hdr['ESO DET DIT'])
	if 'ESO DET DITDELAY' in hdr.keys():
		dit_delay=float(hdr['ESO DET DITDELAY'])
	else:
		## sys.stdout.write('\n Warning! No HIERARCH ESO DET DITDELAY keyword found, using 0. Is it ADI?\n')
		## sys.stdout.flush()
		dit_delay=0

	ha_deg=(float(hdr['LST'])*15./3600)-ra_deg

	# VLT TCS formula
	f1 = cos(geolat_rad) * sin(d2r*ha_deg)
	f2 = sin(geolat_rad) * cos(d2r*dec_deg) - cos(geolat_rad) * sin(d2r*dec_deg) * cos(d2r*ha_deg)

	mjdstart=float(hdr['MJD-OBS'])

	if 'ESO TEL ROT ALTAZTRACK' in hdr.keys() and hdr['ESO TEL ROT ALTAZTRACK']==True:

		parang_array=numpy.array([0,mjdstart,r2d*arctan(f1/f2)+ROT_PT_OFF])
		## utcstart=datetime2jd(dateutil.parser.parse(hdr['DATE']+"T"+hdr['UT']))

		for i in range(1,hdr['NAXIS3']):
			ha_deg=((float(hdr['LST'])+i*(dit+dit_delay))*15./3600)-ra_deg

			# VLT TCS formula
			f1 = cos(geolat_rad) * sin(d2r*ha_deg)
			f2 = sin(geolat_rad) * cos(d2r*dec_deg) - cos(geolat_rad) * sin(d2r*dec_deg) * cos(d2r*ha_deg)

			## parang_array=numpy.vstack((parang_array,[i,float(hdr['LST'])+i*(dit+dit_delay),r2d*arctan((f1)/(f2))+ROT_PT_OFF]))
			parang_array=numpy.vstack((parang_array,[i,mjdstart+i*(dit+dit_delay)/86400.,r2d*arctan(f1/f2)+ROT_PT_OFF]))
	else:
		if 'ARCFILE' in hdr.keys():
			print(hdr['ARCFILE']+' does seem to be taken in pupil tracking.')
		else:
			print('Data does not seem to be taken in pupil tracking.')

		parang_array=numpy.array([0,mjdstart,0])
		## utcstart=datetime2jd(dateutil.parser.parse(hdr['DATE']+"T"+hdr['UT']))

		for i in range(1,hdr['NAXIS3']):
			parang_array=numpy.vstack((parang_array,[i,mjdstart+i*(dit+dit_delay)/86400.,0]))

	return parang_array


def create_parang_list_nici(hdr):
	"""
	Reads the header and creates an array giving the paralactic angle for each frame,
	taking into account the inital derotator position.

	The columns of the output array contains:
	frame_number, frame_time, paralactic_angle
	"""

	from numpy import sin, cos, tan, arctan, pi
	import dateutil.parser

	r2d = 180/pi
	d2r = pi/180

	utcstart=datetime2jd(dateutil.parser.parse(ndata[0].header['DATE']+"T"+ndata[0].header['UT']))
	utcend=datetime2jd(dateutil.parser.parse(ndata[0].header['DATE']+"T"+ndata[0].header['UTEND']))
	if k==1:
		cent_list[k,l,9:]=[l,utcstart+(utcend-utcstart)/2,
			-1*parang(ndata[0].header['DEC'], hms2d(ndata[0].header['HA'],':'), -30.24075)]
	else:
		cent_list[k,l,9:]=[l,utcstart+(utcend-utcstart)/2,
			parang(ndata[0].header['DEC'], hms2d(ndata[0].header['HA'],':'), -30.24075)]


	ra_deg = float(hdr['RA'])
	dec_deg = float(hdr['DEC'])

	geolat_rad=float(hdr['ESO TEL GEOLAT'])*d2r
	if 'ESO ADA PUPILPOS' in hdr.keys():
		ADA_PUPILPOS=float(hdr['ESO ADA PUPILPOS'])
	else:
		ADA_PUPILPOS=89.44
		sys.stdout.write("\n Warning! No ESO ADA PUPILPOS keyword found, using 89.44. Is it ADI?\n")
		sys.stdout.flush()

	ROT_PT_OFF=179.44-ADA_PUPILPOS # from NACO manual v90 p.85

	dit=float(hdr['ESO DET DIT'])
	if 'ESO DET DITDELAY' in hdr.keys():
		dit_delay=float(hdr['ESO DET DITDELAY'])
	else:
		## sys.stdout.write('\n Warning! No HIERARCH ESO DET DITDELAY keyword found, using 0. Is it ADI?\n')
		## sys.stdout.flush()
		dit_delay=0

	ha_deg=(float(hdr['LST'])*15./3600)-ra_deg

	# VLT TCS formula
	f1 = cos(geolat_rad) * sin(d2r*ha_deg)
	f2 = sin(geolat_rad) * cos(d2r*dec_deg) - cos(geolat_rad) * sin(d2r*dec_deg) * cos(d2r*ha_deg)

	parang_array=numpy.array([0,float(hdr['LST']),r2d*arctan(f1/f2)+ROT_PT_OFF])
	## utcstart=datetime2jd(dateutil.parser.parse(hdr['DATE']+"T"+hdr['UT']))
	mjdstart=float(hdr['MJD-OBS'])

	for i in range(1,hdr['NAXIS3']):
		ha_deg=((float(hdr['LST'])+i*(dit+dit_delay))*15./3600)-ra_deg

		# VLT TCS formula
		f1 = cos(geolat_rad) * sin(d2r*ha_deg)
		f2 = sin(geolat_rad) * cos(d2r*dec_deg) - cos(geolat_rad) * sin(d2r*dec_deg) * cos(d2r*ha_deg)

		## parang_array=numpy.vstack((parang_array,[i,float(hdr['LST'])+i*(dit+dit_delay),r2d*arctan((f1)/(f2))+ROT_PT_OFF]))
		parang_array=numpy.vstack((parang_array,[i,mjdstart+i*(dit+dit_delay)/86400.,r2d*arctan(f1/f2)+ROT_PT_OFF]))

	return parang_array

def datetime2jd(t):
	"""
	Converts a datetime object into julian date.
	"""
	from datetime import datetime

	a=(14-t.month)/12
	y=t.year+4800-a
	m=t.month+12*a-3
	JDN=t.day+((153*m+2)/5)+(365*y)+(y/4)-(y/100)+(y/400)-32045
	JD=JDN+((t.hour-12.)/24.)+(t.minute/1440.)+(t.second/86400.)
	# microseconds not used yet!

	return JD

def cut_cube(centroname,cube_in, R, d):
	import tables

	if not os.access(centroname[0], os.F_OK | os.R_OK): # Check if file exists
		sys.stdout('\n Error, cannot access: '+centroname[0])
		sys.stdout.flush()
		for n in range(nprocs-1):
			comm.send("over", dest = n+1 )
			comm.send("over", dest = n+1 )
			sys.exit(1)

	# Read the centroids list
	f=tables.openFile(centroname[0])
	tab=f.root.centroids.read()
	f.close()

	x0=tab[0,0]
	y0=tab[0,1]
	if d>0:
		print(x0,y0)

	# Cut window in first frame and broadcast
	# Defined edges
	xs=numpy.ceil(x0-R)
	xe=numpy.floor(x0+R)
	ys=numpy.ceil(y0-R)
	ye=numpy.floor(y0+R)

	# Check if window limits not out of bounds
	if x0-R < 0:
		xs=0
	if x0+R > cube_in[0].shape[0]:
		xe= cube_in[0].shape[0]
	if y0-R < 0:
		ys=0
	if y0+R > cube_in[0].shape[1]:
		ye= cube_in[0].shape[1]

	if d>0:
		print("xs: "+str(xs)+" xe: "+str(xe)+" ys: "+str(ys)+" ye: "+str(ye)+" x0: "+str(x0)+" y0: "+str(y0))
	## if d>0:
	##	 print("Sending chunks")

	# Mask window containing the star
	mask=numpy.empty(cube_in.shape,dtype=bool)
	mask[:]=False
	mask[:,xs:xe,ys:ye]=True
	masked_cube=numpy.ma.array(cube_in,mask=mask,fill_value=numpy.NaN)
	#cube_in[:,xs:xe,ys:ye]=scipy.NaN

	return masked_cube

## def frame_recenter(frame,dx,dy):
	## print("frame.shape: "+str(frame.shape))
	## frame=ndimage.interpolation.shift(frame, (dx,dy), order=3, mode='constant', cval=numpy.NaN, prefilter=False)
	## return frame

def dprint(D, text):
	"""
	Prints debug text prepended by rank number

	"""
	if D:
		from mpi4py import MPI
		from sys import stdout
		stdout.flush()
		print(str(MPI.COMM_WORLD.Get_rank())+": "+text)

def fft_3shear_rotate(in_frame, alpha,x1,x2,y1,y2):
	"""
	3 FFT shear based rotation, following Larkin et al 1997

	in_frame: the numpy array which has to be rotated
	alpha: the rotation alpha in radians
	x1,x2: the borders of the original image in x
	y1,y2: the borders of the original image in y

	Return the rotated array
	"""
	import numpy as np
	from numpy import fft
	from scipy import ndimage, fftpack
	# Check alpha validity and correct if needed
	alpha=1.*alpha-360*np.floor(alpha/360)
	# FFT rotation only work in the -45:+45 range
	if alpha > 45 and alpha < 135:
		in_frame=np.rot90(in_frame, k=1)
		alpha=-np.deg2rad(alpha-90)
	elif alpha > 135 and alpha < 225:
		in_frame=np.rot90(in_frame, k=2)
		alpha=-np.deg2rad(alpha-180)
	elif alpha > 225 and alpha < 315:
		in_frame=np.rot90(in_frame)
		alpha=-np.deg2rad(alpha-270, k=3)
	else:
		alpha=-np.deg2rad(alpha)
	naxis=x2-x1
	pad=4.
	px1=(pad*naxis/2.)-(in_frame.shape[0]/2)+x1
	px2=(pad*naxis/2.)-(in_frame.shape[0]/2)+x2
	py1=(pad*naxis/2.)-(in_frame.shape[1]/2)+y1
	py2=(pad*naxis/2.)-(in_frame.shape[1]/2)+y2
	pad_frame=np.zeros((naxis*pad,naxis*pad))
	## pad_mask=pad_frame==0 # Ugly way to create a boolean mask, should be changed
	pad_mask=np.ones((pad_frame.shape), dtype=bool)
	pad_frame[
		(pad*naxis/2.)-in_frame.shape[0]/2:(pad*naxis/2.)+in_frame.shape[0]/2,
		(pad*naxis/2.)-in_frame.shape[1]/2:(pad*naxis/2.)+in_frame.shape[1]/2]=in_frame
	## pad_mask=np.ones((in_frame.shape[0]*pad,in_frame.shape[1]*pad))*np.NaN
	pad_mask[
		(pad*naxis/2.)-in_frame.shape[0]/2:(pad*naxis/2.)+in_frame.shape[0]/2,
		(pad*naxis/2.)-in_frame.shape[1]/2:(pad*naxis/2.)+in_frame.shape[1]/2]=np.where(np.isnan(in_frame),True,False)
	# Rotate the mask, to know what part is actually the image
	pad_mask=ndimage.interpolation.rotate(pad_mask, np.rad2deg(-alpha),reshape=False, order=0, mode='constant', cval=True, prefilter=False)
	## print(pad_mask)
	# Replace part outside the image which are NaN by 0, and go into Fourier space.
	pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)
	pad_frame[px1,py1:py2]=pad_frame[px1,py1:py2]/2.
	pad_frame[px2,py1:py2]=pad_frame[px2,py1:py2]/2.
	pad_frame[px1:px2,py1]=pad_frame[px1:px2,py1]/2.
	pad_frame[px1:px2,py2]=pad_frame[px1:px2,py2]/2.
	#pad_frame=np.fftpack.fftshift(np.fftpack.fft2(np.where(np.isnan(pad_frame),0.,pad_frame)))
	# Rotation in Fourier space
	a=np.tan(alpha/2.)
	b=-np.sin(alpha)
	M=-2j*np.pi*np.ones(pad_frame.shape)
	N=fftpack.fftfreq(pad_frame.shape[0])
	X=np.arange(-pad_frame.shape[0]/2.,pad_frame.shape[0]/2.)#/pad_frame.shape[0]
	pad_x=fftpack.ifft((fftpack.fft(pad_frame, axis=0,overwrite_x=True).T*np.exp(a*((M*N).T*X).T)).T, axis=0,overwrite_x=True)
	pad_xy=fftpack.ifft(fftpack.fft(pad_x,axis=1,overwrite_x=True)*np.exp(b*(M*X).T*N), axis=1,overwrite_x=True)
	pad_xyx=fftpack.ifft((fftpack.fft(pad_xy, axis=0,overwrite_x=True).T*np.exp(a*((M*N).T*X).T)).T,axis=0,overwrite_x=True)

	# Go back to real space
	# Put back to NaN pixels outside the image.
	pad_xyx[pad_mask]=np.NaN

	return np.real(pad_xyx[
		((pad*naxis)/2.)-in_frame.shape[0]/2:((pad*naxis)/2.)+in_frame.shape[0]/2,
		((pad*naxis)/2.)-in_frame.shape[1]/2:((pad*naxis)/2.)+in_frame.shape[1]/2]).copy()

def fft_3shear_rotate_pad(in_frame, alpha, pad=4,x1=0,x2=0,y1=0,y2=0):
	"""
	3 FFT shear based rotation, following Larkin et al 1997

	in_frame: the numpy array which has to be rotated
	alpha: the rotation alpha in degrees
	pad: the padding factor
	x1,x2: the borders of the original image in x
	y1,y2: the borders of the original image in y


	Return the rotated array
	"""
	import numpy as np
	from numpy import fft
	from scipy import ndimage, fftpack

	#################################################
	# Check alpha validity and correcting if needed
	#################################################
	alpha=1.*alpha-360*np.floor(alpha/360)

	# FFT rotation only work in the -45:+45 range
	if alpha > 45 and alpha < 135:
		in_frame=np.rot90(in_frame, k=1)
		alpha_rad=-np.deg2rad(alpha-90)
	elif alpha > 135 and alpha < 225:
		in_frame=np.rot90(in_frame, k=2)
		alpha_rad=-np.deg2rad(alpha-180)
	elif alpha > 225 and alpha < 315:
		in_frame=np.rot90(in_frame, k=3)
		alpha_rad=-np.deg2rad(alpha-270)
	else:
		alpha_rad=-np.deg2rad(alpha)

	###################################
	# Preparing the frame for rotation
	###################################
	## naxis=None
	if x1>0:
		## naxis=x2-x1
		## print(naxis,x1,x2,y1,y2)
		## px1=(pad*naxis/2.)-(in_frame.shape[0]/2)+x1
		## px2=(pad*naxis/2.)-(in_frame.shape[0]/2)+x2
		## py1=(pad*naxis/2.)-(in_frame.shape[1]/2)+y1
		## py2=(pad*naxis/2.)-(in_frame.shape[1]/2)+y2
		px1=(pad*in_frame.shape[0]/2)-x1
		px2=(pad*in_frame.shape[0]/2)+x1
		py1=(pad*in_frame.shape[1]/2)-y1
		py2=(pad*in_frame.shape[1]/2)+y1
	pad_frame=np.ones((in_frame.shape[0]*pad,in_frame.shape[1]*pad))*np.NaN
	## pad_frame=np.zeros((in_frame.shape[0]*pad,in_frame.shape[1]*pad))
	## pad_mask=pad_frame==0 # Ugly way to create a boolean mask, should be changed
	pad_mask=np.ones((pad_frame.shape), dtype=bool)
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame
	## pad_mask=np.ones((in_frame.shape[0]*pad,in_frame.shape[1]*pad))*np.NaN
	pad_mask[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=np.where(np.isnan(in_frame),True,False)
	# Rotate the mask, to know what part is actually the image
	pad_mask=ndimage.interpolation.rotate(pad_mask, np.rad2deg(-alpha_rad),reshape=False, order=0, mode='constant', cval=True, prefilter=False)
	## print(pad_mask)
	# Replace part outside the image which are NaN by 0, and go into Fourier space.
	pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)
	if x1==0:
		pad_frame[
			((pad-1)/2.)*in_frame.shape[0]-1,
			((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame[0,:]/2.
		pad_frame[
			((pad+1)/2.)*in_frame.shape[0],
			((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame[-1,:]/2.
		pad_frame[
			((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
			((pad-1)/2.)*in_frame.shape[1]-1]=in_frame[:,0]/2.
		pad_frame[
			((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
			((pad+1)/2.)*in_frame.shape[1]] =in_frame[:,-1]/2.
	elif x1>0:
		pad_frame[px1,py1:py2]=pad_frame[px1,py1:py2]/2.
		pad_frame[px2,py1:py2]=pad_frame[px2,py1:py2]/2.
		pad_frame[px1:px2,py1]=pad_frame[px1:px2,py1]/2.
		pad_frame[px1:px2,py2]=pad_frame[px1:px2,py2]/2.
	#pad_frame=np.fftpack.fftshift(np.fftpack.fft2(np.where(np.isnan(pad_frame),0.,pad_frame)))

	###############################
	# Rotation in Fourier space
	###############################
	a=np.tan(alpha_rad/2.)
	b=-np.sin(alpha_rad)

	M=-2j*np.pi*np.ones(pad_frame.shape)
	N=fftpack.fftfreq(pad_frame.shape[0])

	X=np.arange(-pad_frame.shape[0]/2.,pad_frame.shape[0]/2.)#/pad_frame.shape[0]

	pad_x=fftpack.ifft((fftpack.fft(pad_frame, axis=0,overwrite_x=True).T*np.exp(a*((M*N).T*X).T)).T, axis=0,overwrite_x=True)
	pad_xy=fftpack.ifft(fftpack.fft(pad_x,axis=1,overwrite_x=True)*np.exp(b*(M*X).T*N), axis=1,overwrite_x=True)
	pad_xyx=fftpack.ifft((fftpack.fft(pad_xy, axis=0,overwrite_x=True).T*np.exp(a*((M*N).T*X).T)).T,axis=0,overwrite_x=True)

	# Go back to real space
	# Put back to NaN pixels outside the image.

	pad_xyx[pad_mask]=np.NaN

	## if x1==0:
	return np.real(pad_xyx[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]).copy()
	## else:
		## return np.real(pad_xyx[
			## ((pad*naxis)/2.)-in_frame.shape[0]/2:((pad*naxis)/2.)+in_frame.shape[0]/2,
			## ((pad*naxis)/2.)-in_frame.shape[1]/2:((pad*naxis)/2.)+in_frame.shape[1]/2]).copy()
		## return np.real(pad_xyx[px1:px2,py1:py2]).copy()

def fft_rotate(frame, angle):
	# Rotate a mask, to know what part is actually the image
	mask=ndimage.interpolation.rotate(numpy.where(numpy.isnan(frame),True,False), angle,reshape=False, order=0, mode='constant', cval=True, prefilter=False)
	# Replace part outside the image which are NaN by 0, and go into Fourier space.
	frame=numpy.fft.fftshift(numpy.fft.fft2(numpy.where(numpy.isnan(frame),0.,frame)))
	# Rotation in Fourier space
	frame.imag=ndimage.interpolation.rotate(frame.imag,angle,reshape=False, order=3, mode='constant', cval=0, prefilter=False)
	frame.real=ndimage.interpolation.rotate(frame.real,angle,reshape=False, order=3, mode='constant', cval=0, prefilter=False)
	# Go back to real space
	frame=numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(frame)))
	# Put back to NaN pixels outside the image.
	frame[mask]=numpy.NaN

	return frame

def fft_rotate_pad(in_frame, angle):
	pad=4.
	pad_frame=np.zeros((in_frame.shape[0]*pad,in_frame.shape[1]*pad))
	pad_mask=pad_frame>0 # Ugly way to create a boolean mask, should be changed
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame
	## pad_mask=np.ones((in_frame.shape[0]*pad,in_frame.shape[1]*pad))*np.NaN
	pad_mask[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=numpy.where(numpy.isnan(in_frame),True,False)
	# Rotate a mask, to know what part is actually the image
	pad_mask=ndimage.interpolation.rotate(pad_mask, angle,reshape=False, order=0, mode='constant', cval=True, prefilter=False)
	print(pad_mask)
	# Replace part outside the image which are NaN by 0, and go into Fourier space.
	pad_frame=numpy.fft.fftshift(numpy.fft.fft2(numpy.where(numpy.isnan(pad_frame),0.,pad_frame)))
	# Rotation in Fourier space
	pad_frame.imag=ndimage.interpolation.rotate(pad_frame.imag,angle,reshape=False, order=3, mode='constant', cval=0, prefilter=False)
	pad_frame.real=ndimage.interpolation.rotate(pad_frame.real,angle,reshape=False, order=3, mode='constant', cval=0, prefilter=False)
	# Go back to real space
	pad_frame=numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(pad_frame)))
	# Put back to NaN pixels outside the image.
	pad_frame[pad_mask]=numpy.NaN

	return pad_frame[
		((pad-1)/2.)*in_frame.shape[0]/2:((pad+1)/2.)*in_frame.shape[0]/2,
		((pad-1)/2.)*in_frame.shape[1]/2:((pad+1)/2.)*in_frame.shape[1]/2]

def fft_shift(in_frame, dx, dy):
	f_frame=numpy.fft.fft2(in_frame)
	N=numpy.fft.fftfreq(in_frame.shape[0])
	v=numpy.exp(-2j*numpy.pi*dx*N)
	u=numpy.exp(-2j*numpy.pi*dy*N)
	f_frame=f_frame*u
	f_frame=(f_frame.T*v).T

	return numpy.real(numpy.fft.ifft2(f_frame))

def fft_shift_fpad(in_frame, dx, dy, pad=4.):
	f_frame=np.zeros((in_frame.shape[0]*pad,in_frame.shape[1]*pad),dtype=complex)
	f_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=fftpack.fftshift(fftpack.fft2(in_frame.astype(float)))
	del in_frame
	N=fftpack.fftfreq(f_frame.shape[0])
	v=np.ones((f_frame.shape))*np.exp(-2j*numpy.pi*dx*pad*N)
	u=np.ones((f_frame.shape))*np.exp(-2j*numpy.pi*dy*pad*N)
	f_frame=fftpack.ifftshift(f_frame)
	f_frame=f_frame*u
	f_frame=f_frame*v.T
	## M, N = a.shape
	## m, n = new_shape
	## if m<M:
		## return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
	M,N=f_frame.shape
	## m=M/pad

	return numpy.real(fftpack.ifft2(f_frame)).reshape((M/pad,pad,N/pad,pad)).mean(3).mean(1)
	## return numpy.real(fftpack.ifft2(f_frame))

def fft_shift_pad(in_frame, dx, dy):
	pad=4.
	pad_frame=np.zeros((in_frame.shape[0]*pad,in_frame.shape[1]*pad))
	pad_mask=pad_frame==0 # Ugly way to create a boolean mask, should be changed
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame
	## pad_mask=np.ones((in_frame.shape[0]*pad,in_frame.shape[1]*pad))*np.NaN
	pad_mask[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=np.where(np.isnan(in_frame),True,False)
	# Shift the mask, to know what part is actually the image
	pad_mask=ndimage.interpolation.shift(pad_mask, (dx,dy) , mode='constant', cval=True, prefilter=False)
	## print(pad_mask)
	# Replace part outside the image which are NaN by 0, and go into Fourier space.
	pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]-1,
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame[0,:]/2.
	pad_frame[
		((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame[-1,:]/2.
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]-1]=in_frame[:,0]/2.
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad+1)/2.)*in_frame.shape[1]]=in_frame[:,-1]/2.
	f_frame=np.fft.fft2(pad_frame)
	N=np.fft.fftfreq(pad_frame.shape[0])
	v=np.exp(-2j*numpy.pi*dx*N)
	u=np.exp(-2j*numpy.pi*dy*N)
	f_frame=f_frame*u
	f_frame=(f_frame.T*v).T

	pad_frame=numpy.real(numpy.fft.ifft2(f_frame))
	pad_frame[pad_mask]=np.NaN
	pad_frame=pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]
	return pad_frame


def fftpack_shift_pad(in_frame, dx, dy, pad=4):
	###################################
	# Preparing the frame for shift
	###################################
	## pad=4.
	pad_frame=np.zeros((in_frame.shape[0]*pad,in_frame.shape[1]*pad))
	pad_mask=np.ones((pad_frame.shape), dtype=bool)

	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame
	pad_mask[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=False
		## ((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=np.where(np.isnan(in_frame),True,False)
	# Shift the mask, to know what part is actually the image
	pad_mask=ndimage.interpolation.shift(pad_mask, (dx,dy) ,order=0, mode='constant', cval=True, prefilter=False)
	# Replace part outside the image which are NaN by 0, and go into Fourier space.
	pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]-1,
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame[0,:]/2.
	pad_frame[
		((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame[-1,:]/2.
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]-1]=in_frame[:,0]/2.
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad+1)/2.)*in_frame.shape[1]]=in_frame[:,-1]/2.

	###############################
	# Shift in Fourier space
	###############################
	f_frame=fftpack.fft2(pad_frame)
	N=fftpack.fftfreq(pad_frame.shape[0])
	M=-2j*np.pi*np.ones((pad_frame.shape))
	uv=np.exp((dx*(M*N).T)+dy*M*N)
	## uv=np.exp((dx*(M*N).T).T)
	## uv=np.exp(-2j*np.pi*((N*dx*np.ones((pad_frame.shape))).T+(N*dy*np.ones((pad_frame.shape)))))
	## u=np.exp(-2j*np.pi*dy*N*np.ones((pad_frame.shape)))
	## v=np.exp(-2j*np.pi*dx*N*np.ones((pad_frame.shape))).T
	## u=np.exp(-2j*np.pi*dy*N*np.ones((pad_frame.shape)))
	## v=np.exp(-2j*numpy.pi*dx*N)
	## u=np.exp(-2j*numpy.pi*dy*N)
	## f_frame=f_frame*u*v
	f_frame=f_frame*uv
	## f_frame=f_frame*v

	pad_frame=numpy.real(fftpack.ifft2(f_frame))
	pad_frame[pad_mask]=np.NaN
	pad_frame=pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]
	return pad_frame

def get_max_dim(x,y,shape,alpha):
	"""
	Returns the maximal dimension needed given a position x,y in a frame of dimension shape, with a rotation alpha
	"""
	alpha=(np.pi/180)*alpha
	if x<shape/2:
		x=shape-x
	if y<shape/2:
		y=shape-y
	l=np.sqrt(x**2+y**2)
	return np.ceil(np.max([np.cos(alpha)*l, np.sin(alpha)*l]))


def get_saturation(header):
	'''
	Reads the header and gives the saturation as output

	Using values from the CONICA manual. Satuaration is considered to be attained at
	1/3 of Full Well Depth (ADU):

	Instrument mode  	Readout mode  	Detector mode  	Readout noise (ADU)  	Gain (e-/ADU)  	Full Well (ADU)  	Min DIT (Sec)
	SW				  FowlerNsamp  	HighSensitivity 1.3  	12.1  	9200  	1.7927
	SW				  Double_RdRstRd 	HighDynamic 	4.2 	11.0 	15000 	0.3454
	LW NB imaging 	Uncorr 			HighDynamic 	4.4 	11.0 	15000 	0.1750
	LW L_prime imaging 	Uncorr 			HighWellDepth 	4.4 	9.8 	22000 	0.1750
	LW M_prime imaging 	Uncorr 			HighBackground 	4.4 	9.0 	28000 	0.1750

	Returns sat, the saturation level
	'''

	if header['ESO DET NCORRS NAME']=='Double_RdRstRd':
		sat=15000/3.
	elif header['ESO DET NCORRS NAME']=='FowlerNsamp':
		sat=9200/3.
	elif header['ESO DET NCORRS NAME']=='Uncorr':
		if header['ESO DET MODE NAME']=='HighDynamic':
			sat=15000/3.
		elif header['ESO DET MODE NAME']=='HighWellDepth':
			sat=22000/3.
		elif header['ESO DET MODE NAME']=='HighBackground':
			sat=28000/3.

	return sat

## def get_shift(rw, shift_im ,x_start, x_end, y_start, y_end,R):
	## """
	## Calculate the shift between the reference window (rw) and the shifted image (shift_im)
##
	## -xs,ys x position of the reference window start
	## -xe,ye y position of the reference window end
	## -R the window size
	## """
##
	## # Cut out the window
	## sw = shift_im[x_start:x_end,y_start:y_end]
	## # Set to zero saturated and background pixels
	## sw = numpy.where(sw>0.28,0,sw) #desaturate
	## sw = numpy.where(sw<0.02,0,sw) #set background to 0
	## rw = numpy.where(rw>0.28,0,rw) #desaturate
	## rw = numpy.where(rw<0.02,0,rw) #set background to 0
##
	## cor=correlate2d(rw,sw,mode='same')
	## mass=cor.sum()
	## xc=0
	## yc=0
	## for i in range(cor.shape[0]):
		## xc=xc+(i+1-cor.shape[0]/2.)*cor[i,:].sum()
	## for i in range(cor.shape[1]):
		## yc=yc+(i+1-cor.shape[1]/2.)*cor[:,i].sum()
##
## ##	 for i in range(cor.shape[0]):
## ##		 xc=xc+(i+1)*cor[i,:].sum()
## ##	 for i in range(cor.shape[1]):
## ##		 yc=yc+(i+1)*cor[:,i].sum()
## ##	 x_shift=xc/mass-cor.shape[0]/2.
## ##	 y_shift=yc/mass-cor.shape[1]/2.
##
		## x_shift=xc/mass
		## y_shift=yc/mass
##
	## return x_shift, y_shift

def dms2d(hms, sep):
	"""
	Converts time in DEGsepMsepS format to plain degrees
	"""
	from string import split

	hms=split(hms, sep)
	if hms[0][0]=='-':
		sign=-1.
	else:
		sign=+1.
	deg=sign*(float(hms[0][-2:])+float(hms[1])/60+float(hms[2])/3600)

	return deg

def hms2d(hms, sep):
	"""
	Converts time in HsepMsepS format to degrees
	"""
	from string import split

	hms=split(hms, sep)
	if hms[0][0]=='-':
		sign=-1.
	else:
		sign=+1.
	deg=15.*sign*(float(hms[0][-2:])+float(hms[1])/60+float(hms[2])/3600)

	return deg

def hms2h(hms, sep):
	"""
	Converts time in HsepMsepS format to hours
	"""
	from string import split

	hms=split(hms, sep)
	if hms[0][0]=='-':
		sign=-1.
	else:
		sign=+1.
	h=sign*float(hms[0])+float(hms[1])/60+float(hms[2])/3600

	return h

def humanize_time(secs):
	mins, secs = divmod(secs, 60)
	hours, mins = divmod(mins, 60)
	## return '%02d:%02d:%06.4g' % (hours, mins, secs)

	return '{0:02.0f}:{1:02.0f}:{2:06.3f}'.format(hours, mins, secs)

def iprint(inter, text):
	"""
	Prints debug text prepended by rank number

	"""
	import string
	from sys import stdout

	if inter:
		sys.stdout.write(text)
		stdout.flush()
	else:
		print(text.strip('\r\n'))

	return

def inject_FP(in_frame, rhoVect_as, FluxPrimary_adu, DeltaMagVect, hdr, alpha=0., x0=None, y0=None, r_tel_prim=8.2, r_tel_sec=1.116, noise=True, pad=2):
	"""
	Inject fake companions to an image with a primary star centered. The companions are of different magnitudes (DeltaMagVect)
	and for each magnitude they are at different radial distances (rhoVect_as) from the primary stars

	input:

	image: 2d numpy array containing the astro image
	rhoVect_as: numpy array containing the separation of the companions in arcseconds
	FluxPrimary_adu: Flux of the star used to calculate the flux of the companions with the DeltaMagVect
	DeltaMagVect: numpy array containing the differences between the primary star and the companions
	hdr: header of the fits file of the image
	waveLen: Central wavelength of the filter in wich the image is taken

	Optional
	--------
	alpha: angle of rotation to add to the companions in radians
	x0,y0: translation in pixels of the central star from the center of the image (used if the star is not in the center)
	r_tel_prim: radius of the primary mirror of the telescope (optional, if not specified the radius is the VLT's one)
	r_tel_sec: radius of the secondary mirror of the telescope that hides a part of the field of view (optional, if not specified
	  the radius is the VLT's one)


	output: 2d numpy array containing the astro image with the companions added to it
	"""
	import math
	from scipy import fftpack

	pad_frame=np.zeros((in_frame.shape[0]*pad,in_frame.shape[1]*pad))
	pad_mask=pad_frame==0 # Ugly way to create a boolean mask, should be changed
	pad_mask[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=np.where(np.isnan(in_frame),True,False)
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame
	pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)
	l=pad_frame.shape[0]
	if x0==None:
		x0=l/2.
	else:
		x0=x0+((pad-1)/2.)*in_frame.shape[0]

	if y0==None:
		y0=l/2.
	else:
		y0=y0+((pad-1)/2.)*in_frame.shape[1]

	alpha=np.deg2rad(alpha)

	angle=2.*np.pi/(np.size(DeltaMagVect)) #used to distribute the different magnitudes in the image
	pix_scale_as_pix=hdr['ESO INS PIXSCALE'] # as/pixel
	wavelen_m=hdr['ESO INS CWLEN']*10**(-6) # microns*10**(-6)=meters
	## waveLen_nyquist=1.3778#micron
	## focal_scale_as_p_m=hdr['ESO TEL FOCU SCALE']*10**(3) #Focal scale (arcsec/mm)*10**(3)=(arcsec/m)
	pix_size_m=hdr['ESO DET CHIP PXSPACE']
	## waveLen_nyquist_m=(2*r_tel_prim*as_par_pixel)/focal_scale_as_p_m #meters
	wavelen_nyquist_m=(r_tel_prim*pix_scale_as_pix*pix_size_m)/1.22  # "Electronic imaging in astronomy - Detectors and Instrumentation - Ian S. Maclean - 4.3 Matching the plate scale pp74-75 "
	## wavelen_nyquist_m=(pix_scale_as_pix*r_tel_prim*2)/(1.22*206265) # "Electronic imaging in astronomy - Detectors and Instrumentation - Ian S. Maclean - 4.3 Matching the plate scale pp74-75 "
	## focal=(pix_size*180.*(60.**2))/(np.pi*as_par_pixel)

	## r_ext=(waveLen_nyquist_m/wavelen_m)*(l/4.)
	r_ext=(wavelen_m/wavelen_nyquist_m)*(l/8.)
	## r_ext=l/4.
	r_int=r_ext*(r_tel_sec/r_tel_prim)

	x=y=np.arange(-l/2.,l/2.)

	X,Y=np.meshgrid(x,y)
	R = np.sqrt(X**2 + Y**2)

	u_asm1=v_asm1=fftpack.fftshift(fftpack.fftfreq(np.size(x),d=pix_scale_as_pix)) # in arcsec^-1
	## u_asm1=v_asm1=fftpack.fftshift(fftpack.fftfreq(np.size(x),d=as_par_pixel*2.)) # in arcsec^-1
	u_pix1=v_pix1=fftpack.fftshift(fftpack.fftfreq(np.size(x))) # in pix^-1

	## FluxPrimary_adu=FluxPrimary_adu

	#PSF telescope
	telescopeFilter = np.ones((l,l))
	telescopeFilter[np.where(R<r_int,True,False)+np.where(R>r_ext,True,False)]=0
	## nbr_pix_pupil=np.size(telescopeFilter[np.where(R<r_ext,True,False)-np.where(R<r_int,True,False)])
	nbr_pix_pupil=np.sum(telescopeFilter) # Simply count the ones in telescopeFilter to have the size
	factor=l**2./nbr_pix_pupil
	FluxPrimary_adu=FluxPrimary_adu-np.median(in_frame)

	pupil = np.zeros((l,l),dtype=complex)

	for k in range(len(DeltaMagVect)): #different magn in each quadrant
		FluxSecondary_adu= FluxPrimary_adu*math.pow(10,-0.4*DeltaMagVect[k])

		Amplitude = factor*np.sqrt(FluxSecondary_adu)
		waveFront  = np.zeros((l,l),dtype=complex)

		for j in range(len(rhoVect_as)):#displaying the companions with the good distances from the star
			deltaX_as=rhoVect_as[j]*np.cos(k*angle+alpha)#x position of the companion from the center
			deltaY_as=rhoVect_as[j]*np.sin(k*angle+alpha)#y position of the companion from the center
			## phasor = ((np.ones((l,l))*np.exp(-2j*np.pi*(deltaX_as*u_asm1+x0*u_pix1))).T*np.exp(-2j*np.pi*(deltaY_as*v_asm1+y0*v_pix1))).T
			phasor = ((np.ones((l,l))*np.exp(-2j*np.pi*(deltaY_as*u_asm1+y0*u_pix1))).T*np.exp(-2j*np.pi*(deltaX_as*v_asm1+x0*v_pix1))).T
			waveFront += phasor

		pupil += Amplitude*waveFront*telescopeFilter

	## a=(fftpack.ifft2(fftpack.fftshift(pupil)))
	## a=(fftpack.fftshift(fftpack.ifft2(pupil)))*(np.conjugate(fftpack.fftshift(fftpack.ifft2(pupil))))
	a=(fftpack.ifft2(fftpack.fftshift(pupil)))*(np.conjugate(fftpack.ifft2(fftpack.fftshift(pupil))))

	if noise:
		fp_image = np.random.poisson(np.real(a))
	else:
		fp_image = np.real(a)

	pad_frame+=fp_image[:,:]
	## pad_frame=np.where(fp_image>pad_frame, fp_image, pad_frame)

	pad_frame[pad_mask]=np.NaN
	pad_frame=pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]

	return pad_frame


def inject_FP_beta(in_frame, rhoVect_as, FluxPrimary_adu, DeltaMagVect, hdr, alpha=0., x0=None, y0=None, r_tel_prim=8.2, r_tel_sec=1.116, noise=True, pad=2):
	"""
	Inject fake companions to an image with a primary star centered. The companions are of different magnitudes (DeltaMagVect)
	and for each magnitude they are at different radial distances (rhoVect_as) from the primary stars

	input:

	image: 2d numpy array containing the astro image
	rhoVect_as: numpy array containing the separation of the companions in arcseconds
	FluxPrimary_adu: Flux of the star used to calculate the flux of the companions with the DeltaMagVect
	DeltaMagVect: numpy array containing the differences between the primary star and the companions
	hdr: header of the fits file of the image
	waveLen: Central wavelength of the filter in wich the image is taken

	Optional
	--------
	alpha: angle of rotation to add to the companions in radians
	x0,y0: translation in pixels of the central star from the center of the image (used if the star is not in the center)
	r_tel_prim: radius of the primary mirror of the telescope (optional, if not specified the radius is the VLT's one)
	r_tel_sec: radius of the secondary mirror of the telescope that hides a part of the field of view (optional, if not specified
	  the radius is the VLT's one)
	noise: to add poisson noise
	pad: how much the frame needs to be padded in order to prevent FP wrapping due do dithering

	output: 2d numpy array containing the astro image with the companions added to it
	"""
	import math
	from scipy import fftpack

	pad_frame=np.zeros((in_frame.shape[0]*pad,in_frame.shape[1]*pad))
	pad_mask=pad_frame==0 # Ugly way to create a boolean mask, should be changed
	pad_mask[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=np.where(np.isnan(in_frame),True,False)
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame
	pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)
	l=pad_frame.shape[0]
	if x0==None:
		x0=l/2.
	else:
		x0=x0+((pad-1)/2.)*in_frame.shape[0]

	if y0==None:
		y0=l/2.
	else:
		y0=y0+((pad-1)/2.)*in_frame.shape[1]

	alpha=np.deg2rad(alpha)

	angle=2.*np.pi/(np.size(DeltaMagVect)) #used to distribute the different magnitudes in the image
	pix_scale_as_pix=hdr['ESO INS PIXSCALE'] # as/pixel
	wavelen_m=hdr['ESO INS CWLEN']*10**(-6) # microns*10**(-6)=meters
	## waveLen_nyquist=1.3778#micron
	## focal_scale_as_p_m=hdr['ESO TEL FOCU SCALE']*10**(3) #Focal scale (arcsec/mm)*10**(3)=(arcsec/m)
	pix_size_m=hdr['ESO DET CHIP PXSPACE']
	## waveLen_nyquist_m=(2*r_tel_prim*as_par_pixel)/focal_scale_as_p_m #meters
	## wavelen_nyquist_m=(r_tel_prim*pix_scale_as_pix*pix_size_m)/1.22  # "Electronic imaging in astronomy - Detectors and Instrumentation - Ian S. Maclean - 4.3 Matching the plate scale pp74-75 "
	wavelen_nyquist_m=(pix_scale_as_pix*r_tel_prim*2)/(1.22*206265) # "Electronic imaging in astronomy - Detectors and Instrumentation - Ian S. Maclean - 4.3 Matching the plate scale pp74-75 "
	## focal=(pix_size*180.*(60.**2))/(np.pi*as_par_pixel)

	## r_ext=(waveLen_nyquist_m/wavelen_m)*(l/4.)
	r_ext=(wavelen_m/wavelen_nyquist_m)*(l/8.)
	## r_ext=l/4.
	r_int=r_ext*(r_tel_sec/r_tel_prim)

	x=y=np.arange(-l/2.,l/2.)

	X,Y=np.meshgrid(x,y)
	R = np.sqrt(X**2 + Y**2)

	u_asm1=v_asm1=fftpack.fftshift(fftpack.fftfreq(np.size(x),d=pix_scale_as_pix)) # in arcsec^-1
	## u_asm1=v_asm1=fftpack.fftshift(fftpack.fftfreq(np.size(x),d=as_par_pixel*2.)) # in arcsec^-1
	u_pix1=v_pix1=fftpack.fftshift(fftpack.fftfreq(np.size(x))) # in pix^-1

	## FluxPrimary_adu=FluxPrimary_adu

	#PSF telescope
	telescopeFilter = np.ones((l,l))
	telescopeFilter[np.where(R<r_int,True,False)+np.where(R>r_ext,True,False)]=0
	## nbr_pix_pupil=np.size(telescopeFilter[np.where(R<r_ext,True,False)-np.where(R<r_int,True,False)])
	nbr_pix_pupil=np.sum(telescopeFilter) # Simply count the ones in telescopeFilter to have the size
	factor=l**2./nbr_pix_pupil
	FluxPrimary_adu=FluxPrimary_adu-np.median(in_frame)

	pupil = np.zeros((l,l),dtype=complex)

	for k in range(len(DeltaMagVect)): #different magn in each quadrant
		FluxSecondary_adu= FluxPrimary_adu*math.pow(10,-0.4*DeltaMagVect[k])

		Amplitude = factor*np.sqrt(FluxSecondary_adu)
		waveFront  = np.zeros((l,l),dtype=complex)

		for j in range(len(rhoVect_as)):#displaying the companions with the good distances from the star
			deltaX_as=rhoVect_as[j]*np.cos(k*angle+alpha)#x position of the companion from the center
			deltaY_as=rhoVect_as[j]*np.sin(k*angle+alpha)#y position of the companion from the center
			## phasor = ((np.ones((l,l))*np.exp(-2j*np.pi*(deltaX_as*u_asm1+x0*u_pix1))).T*np.exp(-2j*np.pi*(deltaY_as*v_asm1+y0*v_pix1))).T
			phasor = ((np.ones((l,l))*np.exp(-2j*np.pi*(deltaY_as*u_asm1+y0*u_pix1))).T*np.exp(-2j*np.pi*(deltaX_as*v_asm1+x0*v_pix1))).T
			waveFront += phasor

		pupil += Amplitude*waveFront*telescopeFilter

	## a=(fftpack.ifft2(fftpack.fftshift(pupil)))
	## a=(fftpack.fftshift(fftpack.ifft2(pupil)))*(np.conjugate(fftpack.fftshift(fftpack.ifft2(pupil))))
	a=(fftpack.ifft2(fftpack.fftshift(pupil)))*(np.conjugate(fftpack.ifft2(fftpack.fftshift(pupil))))

	if noise:
		fp_image = np.random.poisson(np.real(a))
	else:
		fp_image = np.real(a)

	pad_frame+=fp_image[:,:]

	pad_frame[pad_mask]=np.NaN
	pad_frame=pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]

	return pad_frame


def inject_FP_nici(in_frame, rhoVect_as, FluxPrimary_adu, DeltaMagVect, hdr, alpha=0., x0=None, y0=None, r_tel_prim=8.1, r_tel_sec=1.0, noise=True, pad=2, pix_scale_as_pix=0.018):
	"""
	Inject fake companions to an image with a primary star centered. The companions are of different magnitudes (DeltaMagVect)
	and for each magnitude they are at different radial distances (rhoVect_as) from the primary stars

	input:

	image: 2d numpy array containing the astro image
	rhoVect_as: numpy array containing the separation of the companions in arcseconds
	FluxPrimary_adu: Flux of the star used to calculate the flux of the companions with the DeltaMagVect
	DeltaMagVect: numpy array containing the differences between the primary star and the companions
	hdr: header of the fits file of the image
	waveLen: Central wavelength of the filter in wich the image is taken

	Optional
	--------
	alpha: angle of rotation to add to the companions in radians
	x0,y0: translation in pixels of the central star from the center of the image (used if the star is not in the center)
	r_tel_prim: radius of the primary mirror of the telescope (optional, if not specified the radius is the VLT's one)
	r_tel_sec: radius of the secondary mirror of the telescope that hides a part of the field of view (optional, if not specified
	  the radius is the VLT's one)
	noise: to add poisson noise
	pad: how much the frame needs to be padded in order to prevent FP wrapping due do dithering
	pix_scale_as_pix: pixel scale in arcsecond per pixel NICI=0.018


	output: 2d numpy array containing the astro image with the companions added to it
	"""
	import math, string
	from scipy import fftpack

	## CH4 H 1% S 1.587 	0.0150 (0.94%) 	G0724 G0722
	## CH4 H 1% Sp 	1.603 	0.0162 (1.01%) 	G0728 G0726
	## CH4 H 1% L 	1.628 	0.0174 (1.07%) 	G0720 G0732
	## CH4 H 1% L_2 1.628 	0.0174 (1.07%) 	G0735
	## CH4 H 4% S	1.578 	0.062 (4.00%) 	G0742 G0743
	## CH4 H 4% L	1.652 	0.066 (3.95%) 	G0740 G0737
	## CH4 K 5% S	2.080 	0.105 (5.06%)  	G0746
	## CH4 K 5% L	2.241 	0.107 (4.84%) 	G0748
	## H20 Ice L	3.09 	3.02 - 3.15 	G0715
	## J	1.25 	1.15-1.33 	G0702
	## H 	1.65 	1.49-1.78 	G0703
	## K 	2.20 	2.03-2.36 	G0704
	## Ks 	2.15 	1.99-2.30 	G0705
	## Kprime 	2.12 	1.95-2.30 	G0706
	## Lprime 	3.78 	3.43-4.13 	G0707 	  	Must be used w/ H50/50 to reduce bgd.
	## Mprime	4.68 	4.55-4.79 	G0708 	  	Installed, but not available due to saturation.
	## [Fe II] 	1.644 	1.5% 	  	G0712
	## H2 1-0 S(1) 	2.1239 	1.23% 	G0709
	## Br-gamma 	2.1686 	1.36%	G0711
	## Kcont 	2.2718 	1.55% 	G0710
	## CH4 H 6.5% S	1.596 	0.1175 (7.3%) 	G0713
	## CH4 H 6.5% L 1.701 	0.0972 (5.7%) 	G0714

	filter_wavelength={'CH4-H1%S': 1.587, 'CH4-H1%Sp': 1.603, 'CH4-H1%L':1.628, 'CH4-H1%L_2': 1.628, 'CH4-H4%S' : 1.578, 'CH4-H4%L': 1.652, 'CH4-K5%S': 2.080, 'CH4-K5%L': 2.241}

	if 'CHANNEL' in hdr.keys():
		if hdr['CHANNEL']=='BLUE':
			band=string.split(hdr['FILTER_B'],'_')[0]
			if band in filter_wavelength.keys():
				wavelen_m=filter_wavelength[band]*10**(-6)
			else:
				print('Unknow filter: '+band)
				wavelen_m=None
		elif hdr['CHANNEL']=='RED':
			band=string.split(hdr['FILTER_R'],'_')[0]
			if band in filter_wavelength.keys():
				wavelen_m=filter_wavelength[band]*10**(-6)
			else:
				print('Unknow filter: '+band)
				wavelen_m=None
	if wavelen_m==None:
		print('No wavelength information found in fits header!')
		return None

	pad_frame=np.zeros((in_frame.shape[0]*pad,in_frame.shape[1]*pad))
	pad_mask=pad_frame==0 # Ugly way to create a boolean mask, should be changed
	pad_mask[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=np.where(np.isnan(in_frame),True,False)
	pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]=in_frame
	pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)
	l=pad_frame.shape[0]
	if x0==None:
		x0=l/2.
	else:
		x0=x0+((pad-1)/2.)*in_frame.shape[0]

	if y0==None:
		y0=l/2.
	else:
		y0=y0+((pad-1)/2.)*in_frame.shape[1]

	alpha=np.deg2rad(alpha)

	angle=2.*np.pi/(np.size(DeltaMagVect)) #used to distribute the different magnitudes in the image
	## as_par_pixel=hdr['ESO INS PIXSCALE'] # as/pixel
	## wavelen_m=hdr['ESO INS CWLEN']*10**(-6) # microns*10**(-6)=meters
	## waveLen_nyquist=1.3778#micron
	## focal_scale_as_p_m=hdr['ESO TEL FOCU SCALE']*10**(3) #Focal scale (arcsec/mm)*10**(3)=(arcsec/m)
	## pix_size_m=hdr['ESO DET CHIP PXSPACE']

	wavelen_nyquist_m=(pix_scale_as_pix*r_tel_prim*2)/(1.22*206265) # "Electronic imaging in astronomy - Detectors and Instrumentation - Ian S. Maclean - 4.3 Matching the plate scale pp74-75 "

	r_ext=(wavelen_m/wavelen_nyquist_m)*(l/8.)
	r_int=r_ext*(r_tel_sec/r_tel_prim)

	x=y=np.arange(-l/2.,l/2.)

	X,Y=np.meshgrid(x,y)
	R = np.sqrt(X**2 + Y**2)

	u_asm1=v_asm1=fftpack.fftshift(fftpack.fftfreq(np.size(x),d=pix_scale_as_pix)) # en arcsec^-1
	## u_asm1=v_asm1=fftpack.fftshift(fftpack.fftfreq(np.size(x),d=as_par_pixel*2.)) # en arcsec^-1
	u_pix1=v_pix1=fftpack.fftshift(fftpack.fftfreq(np.size(x))) # en pix^-1

	## FluxPrimary_adu=FluxPrimary_adu

	#PSF telescope
	telescopeFilter = np.ones((l,l))
	telescopeFilter[np.where(R<r_int,True,False)+np.where(R>r_ext,True,False)]=0
	## nbr_pix_pupil=np.size(telescopeFilter[np.where(R<r_ext,True,False)-np.where(R<r_int,True,False)])
	nbr_pix_pupil=np.sum(telescopeFilter) # Simply count the ones in telescopeFilter to have the size
	factor=l**2./nbr_pix_pupil
	FluxPrimary_adu=FluxPrimary_adu-np.median(in_frame)

	pupil = np.zeros((l,l),dtype=complex)

	for k in range(len(DeltaMagVect)): #different magn in each quadrant
		FluxSecondary_adu= FluxPrimary_adu*math.pow(10,-0.4*DeltaMagVect[k])

		Amplitude = factor*np.sqrt(FluxSecondary_adu)
		waveFront  = np.zeros((l,l),dtype=complex)

		for j in range(len(rhoVect_as)):#displaying the companions with the good distances from the star
			deltaX_as=rhoVect_as[j]*np.cos(k*angle+alpha)#x position of the companion from the center
			deltaY_as=rhoVect_as[j]*np.sin(k*angle+alpha)#y position of the companion from the center
			## phasor = ((np.ones((l,l))*np.exp(-2j*np.pi*(deltaX_as*u_asm1+x0*u_pix1))).T*np.exp(-2j*np.pi*(deltaY_as*v_asm1+y0*v_pix1))).T
			phasor = ((np.ones((l,l))*np.exp(-2j*np.pi*(deltaY_as*u_asm1+y0*u_pix1))).T*np.exp(-2j*np.pi*(deltaX_as*v_asm1+x0*v_pix1))).T
			waveFront += phasor

		pupil += Amplitude*waveFront*telescopeFilter

	## a=(fftpack.ifft2(fftpack.fftshift(pupil)))
	## a=(fftpack.fftshift(fftpack.ifft2(pupil)))*(np.conjugate(fftpack.fftshift(fftpack.ifft2(pupil))))
	a=(fftpack.ifft2(fftpack.fftshift(pupil)))*(np.conjugate(fftpack.ifft2(fftpack.fftshift(pupil))))

	if noise:
		fp_image = np.random.poisson(np.real(a))
	else:
		fp_image = np.real(a)

	pad_frame+=fp_image[:,:]

	pad_frame[pad_mask]=np.NaN
	pad_frame=pad_frame[
		((pad-1)/2.)*in_frame.shape[0]:((pad+1)/2.)*in_frame.shape[0],
		((pad-1)/2.)*in_frame.shape[1]:((pad+1)/2.)*in_frame.shape[1]]

	return pad_frame


def inject_FP_nopad(image, rhoVect_as, FluxPrimary_adu, DeltaMagVect, hdr, alpha=0., x0=None, y0=None, r_tel_prim=8.2, r_tel_sec=1.116, noise=True):
		"""
		Inject fake companions to an image with a primary star centered. The companions are of different magnitudes (DeltaMagVect)
		and for each magnitude they are at different radial distances (rhoVect_as) from the primary stars

		input:

		image: 2d numpy array containing the astro image
		rhoVect_as: numpy array containing the separation of the companions in arcseconds
		FluxPrimary_adu: Flux of the star used to calculate the flux of the companions with the DeltaMagVect
		DeltaMagVect: numpy array containing the differences between the primary star and the companions
		hdr: header of the fits file of the image
		waveLen: Central wavelength of the filter in wich the image is taken

		Optional
		--------
		alpha: angle of rotation to add to the companions in radians
		x0,y0: translation in pixels of the central star from the center of the image (used if the star is not in the center)
		r_tel_prim: radius of the primary mirror of the telescope (optional, if not specified the radius is the VLT's one)
		r_tel_sec: radius of the secondary mirror of the telescope that hides a part of the field of view (optional, if not specified
		  the radius is the VLT's one)


		output: 2d numpy array containing the astro image with the companions added to it
		"""
		import math
		from scipy import fftpack

		l=image.shape[0]
		if x0==None:
				x0=l/2.

		if y0==None:
				y0=l/2.

		alpha=np.deg2rad(alpha)

		angle=2.*np.pi/(np.size(DeltaMagVect)) #used to distribute the different magnitudes in the image
		as_par_pixel=hdr['ESO INS PIXSCALE'] # as/pixel
		wavelen_m=hdr['ESO INS CWLEN']*10**(-6) # microns*10**(-6)=meters
		## waveLen_nyquist=1.3778#micron
		## focal_scale_as_p_m=hdr['ESO TEL FOCU SCALE']*10**(3) #Focal scale (arcsec/mm)*10**(3)=(arcsec/m)
		pix_size_m=hdr['ESO DET CHIP PXSPACE']
		## waveLen_nyquist_m=(2*r_tel_prim*as_par_pixel)/focal_scale_as_p_m #meters
		wavelen_nyquist_m=(r_tel_prim*as_par_pixel*pix_size_m)/1.22  # "Electronic imaging in astronomy - Detectors and Instrumentation - Ian S. Maclean - 4.3 Matching the plate scale pp74-75 "
		## focal=(pix_size*180.*(60.**2))/(np.pi*as_par_pixel)

		## r_ext=(waveLen_nyquist_m/wavelen_m)*(l/4.)
		r_ext=(wavelen_m/wavelen_nyquist_m)*(l/8.)
		## r_ext=l/4.
		r_int=r_ext*(r_tel_sec/r_tel_prim)

		x=y=np.arange(-l/2.,l/2.)

		X,Y=np.meshgrid(x,y)
		R = np.sqrt(X**2 + Y**2)

		u_asm1=v_asm1=fftpack.fftshift(fftpack.fftfreq(np.size(x),d=as_par_pixel)) # en arcsec^-1
		## u_asm1=v_asm1=fftpack.fftshift(fftpack.fftfreq(np.size(x),d=as_par_pixel*2.)) # en arcsec^-1
		u_pix1=v_pix1=fftpack.fftshift(fftpack.fftfreq(np.size(x))) # en pix^-1

		## FluxPrimary_adu=FluxPrimary_adu

		#PSF telescope
		telescopeFilter = np.ones((l,l))
		telescopeFilter[np.where(R<r_int,True,False)+np.where(R>r_ext,True,False)]=0
		## nbr_pix_pupil=np.size(telescopeFilter[np.where(R<r_ext,True,False)-np.where(R<r_int,True,False)])
		nbr_pix_pupil=np.sum(telescopeFilter) # Simply count the ones in telescopeFilter to have the size
		factor=l**2./nbr_pix_pupil
		FluxPrimary_adu=FluxPrimary_adu-np.median(image)

		pupil = np.zeros((l,l),dtype=complex)

		for k in range(len(DeltaMagVect)): #different magn in each quadrant
				FluxSecondary_adu= FluxPrimary_adu*math.pow(10,-0.4*DeltaMagVect[k])

				Amplitude = factor*np.sqrt(FluxSecondary_adu)
				waveFront  = np.zeros((l,l),dtype=complex)


				for j in range(len(rhoVect_as)):#displaying the companions with the good distances from the star
						deltaX_as=rhoVect_as[j]*np.cos(k*angle+alpha)#x position of the companion from the center
						deltaY_as=rhoVect_as[j]*np.sin(k*angle+alpha)#y position of the companion from the center
						## phasor = ((np.ones((l,l))*np.exp(-2j*np.pi*(deltaX_as*u_asm1+x0*u_pix1))).T*np.exp(-2j*np.pi*(deltaY_as*v_asm1+y0*v_pix1))).T
						phasor = ((np.ones((l,l))*np.exp(-2j*np.pi*(deltaY_as*u_asm1+y0*u_pix1))).T*np.exp(-2j*np.pi*(deltaX_as*v_asm1+x0*v_pix1))).T
						waveFront += phasor

				pupil += Amplitude*waveFront*telescopeFilter

		## a=(fftpack.ifft2(fftpack.fftshift(pupil)))
		## a=(fftpack.fftshift(fftpack.ifft2(pupil)))*(np.conjugate(fftpack.fftshift(fftpack.ifft2(pupil))))
		a=(fftpack.ifft2(fftpack.fftshift(pupil)))*(np.conjugate(fftpack.ifft2(fftpack.fftshift(pupil))))

		if noise:
				fp_image = np.random.poisson(np.real(a))
		else:
				fp_image = np.real(a)

		image+=fp_image[:,:]

		return image


def mask_center(frame, R, x0, y0):
	"""
	Mask out the saturated center of the frame. And draw a cross with frame median value at the center (x0, y0).


	"""
	sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
	import bottleneck

	x,y = numpy.indices(frame.shape)
	r = numpy.sqrt((x-x0)**2+(y-y0)**2)
	frame=numpy.where(r > R, frame, numpy.NaN)
	frame_med=bottleneck.nanmedian(frame)
	#frame[numpy.round(x0),numpy.round(y0)]=frame_med
	frame[x0,:]=numpy.where(numpy.isnan(frame[x0,:]),frame_med , frame[x0,:])
	frame[:,y0]=numpy.where(numpy.isnan(frame[:,y0]),frame_med , frame[:,y0])

	return frame

def mask_companion(frame, Radius, companion_positions):
	"""
	Masks out companions in the frame.


	The companion positions are give by the companion_positions array.
	The size of the companions is given by R.
	"""
	## sys.path.append("/home/spectro/hagelber/Astro/lib64/python/")
	## import bottleneck
	import numpy

	x,y = numpy.indices(frame.shape)

	if len(companion_positions.shape)==1:
		x0,y0=companion_positions
		r = numpy.sqrt((x-x0)**2+(y-y0)**2)
		frame=numpy.where(r > Radius, frame, numpy.NaN)
	elif companion_positions.shape[1]==2:
		for i in range(companion_positions.shape[0]):
			x0,y0=companion_positions[i]
			r = numpy.sqrt((x-x0)**2+(y-y0)**2)
			frame=numpy.where(r > Radius, frame, numpy.NaN)
	else:
		print("Wrong companion_positions array format. Expecting (N,2) shape.")
	return frame

def mask_cube(x0, y0,cube_in, R, d):

	# Cut window in first frame and broadcast
	# Defined edges
	xs=numpy.ceil(x0-R)
	xe=numpy.floor(x0+R)
	ys=numpy.ceil(y0-R)
	ye=numpy.floor(y0+R)

	# Check if window limits not out of bounds
	if x0-R < 0:
		xs=0
	if x0+R > cube_in[0].shape[0]:
		xe= cube_in[0].shape[0]
	if y0-R < 0:
		ys=0
	if y0+R > cube_in[0].shape[1]:
		ye= cube_in[0].shape[1]

	if d>0:
		print("xs: "+str(xs)+" xe: "+str(xe)+" ys: "+str(ys)+" ye: "+str(ye)+" x0: "+str(x0)+" y0: "+str(y0))

	# Mask window containing the star
	mask=numpy.empty(cube_in.shape,dtype=bool)
	mask[:]=False
	mask[:,xs:xe,ys:ye]=True
	masked_cube=numpy.ma.array(cube_in,mask=mask,fill_value=numpy.NaN)

	return masked_cube

def parang(dec, ha, geolat):
	"""
	Read a header and calculates the paralactic angle, using method derived by Arthur Vigan
	"""
	from numpy import sin, cos, tan, arctan, pi

	r2d = 180/pi
	d2r = pi/180

	#ra_deg = float(ra)
	dec_rad = float(dec)*d2r
	ha_rad = float(ha)*d2r
	geolat_rad=float(geolat)*d2r

	## ha_deg=(float(hdr['LST'])*15./3600)-ra_deg

	# VLT TCS formula
	f1 = cos(geolat_rad) * sin(ha_rad)
	f2 = sin(geolat_rad) * cos(dec_rad) - cos(geolat_rad) * sin(dec_rad) * cos(ha_rad)

	parang_deg=r2d*arctan(f1/f2)

	return parang_deg

def nanmask_cube(x0, y0,cube_in, R, d):

	# Cut window in first frame and broadcast
	# Defined edges
	xs=numpy.ceil(x0-R)
	xe=numpy.floor(x0+R)
	ys=numpy.ceil(y0-R)
	ye=numpy.floor(y0+R)

	# Check if window limits not out of bounds
	if x0-R < 0:
		xs=0
	if x0+R > cube_in.shape[1]:
		xe= cube_in.shape[1]
	if y0-R < 0:
		ys=0
	if y0+R > cube_in.shape[2]:
		ye= cube_in.shape[2]

	if d>0:
		print("xs: "+str(xs)+" xe: "+str(xe)+" ys: "+str(ys)+" ye: "+str(ye)+" x0: "+str(x0)+" y0: "+str(y0))

	# Mask window containing the star
	cube_in[:,xs:xe,ys:ye]=numpy.NaN

	return cube_in

def nanmask_cube_nici(x0, y0,cube_in, R, d):

	# Cut window in first frame and broadcast
	# Defined edges
	xs=numpy.ceil(x0-R)
	xe=numpy.floor(x0+R)
	ys=numpy.ceil(y0-R)
	ye=numpy.floor(y0+R)

	# Check if window limits not out of bounds
	if x0-R < 0:
		xs=0
	if x0+R > cube_in[0].shape[1]:
		xe= cube_in[0].shape[1]
	if y0-R < 0:
		ys=0
	if y0+R > cube_in[0].shape[2]:
		ye= cube_in[0].shape[2]

	if d>0:
		print("xs: "+str(xs)+" xe: "+str(xe)+" ys: "+str(ys)+" ye: "+str(ye)+" x0: "+str(x0)+" y0: "+str(y0))

	# Mask window containing the star
	cube_in[:,xs:xe,ys:ye]=numpy.NaN

	return cube_in

def nanmask_frame(x0, y0,frame, R, d):

	# Cut window in first frame and broadcast
	# Defined edges
	xs=numpy.ceil(x0-R)
	xe=numpy.floor(x0+R)
	ys=numpy.ceil(y0-R)
	ye=numpy.floor(y0+R)

	# Check if window limits not out of bounds
	if x0-R < 0:
		xs=0
	if x0+R > frame.shape[0]:
		xe= frame.shape[0]
	if y0-R < 0:
		ys=0
	if y0+R > frame.shape[1]:
		ye= frame.shape[1]

	if d>0:
		print("xs: "+str(xs)+" xe: "+str(xe)+" ys: "+str(ys)+" ye: "+str(ye)+" x0: "+str(x0)+" y0: "+str(y0))

	# Mask window containing the star
	frame[xs:xe,ys:ye]=numpy.NaN


	return frame

def read_rdb(file, h=0, comment=None):
	"""
	Reads an rdb file

	Input:
	 file: rdb_filename
	 h: line number of header
	 comment: comment char, for lines to be ignored

	Output: content of the file in form of a list
	"""

	import string, os
	# Check if file exists
	if not os.access(file, os.R_OK):
		return None

	f = open(file,'r');
	data = f.readlines()
	f.close()

	# take the second line to define the list keys.
	key = string.split(data[h][:-1],'\t')
	data_list = {}
	for i in range(len(key)): data_list[key[i]] = []

	for line in data[h+2:]:
		if not line[0]==comment or line[0:2]=='--':
			qq = string.split(line[:-1],'\t')
			for i in range(len(key)):
				try: value = float(qq[i])
				except ValueError: value = qq[i]
				data_list[key[i]].append(value)

	return data_list


def read_rdb_rows(filename,refcol):

	f = open(filename, 'r')
	data = f.readlines()
	f.close()

	key = string.split(data[0][:-1],'\t')
	iref = key.index(refcol)
	output = {}

	for line in data[2:]:
		qq1 = string.split(line[:-1],'\t')
		qq2 = {}
		for i in range(len(key)): qq2[key[i]] = qq1[i]
		output[qq1[iref]] = qq2

	return output

def rebin(a, new_shape):
	"""
	Resizes a 2d array by averaging or repeating elements,
	new dimensions must be integral factors of original dimensions

	Parameters
	----------
	a : array_like
	Input array.
	new_shape : tuple of int
	Shape of the output array

	Returns
	-------
	rebinned_array : ndarray
	If the new shape is smaller of the input array, the data are averaged,
	if the new shape is bigger array elements are repeated

	See Also
	--------
	resize : Return a new array with the specified shape.

	Examples
	--------
	>>> a = np.array([[0, 1], [2, 3]])
	>>> b = rebin(a, (4, 6)) #upsize
	>>> b
	array([[0, 0, 0, 1, 1, 1],
	[0, 0, 0, 1, 1, 1],
	[2, 2, 2, 3, 3, 3],
	[2, 2, 2, 3, 3, 3]])

	>>> c = rebin(b, (2, 3)) #downsize
	>>> c
	array([[ 0. , 0.5, 1. ],
	[ 2. , 2.5, 3. ]])

	Code by Andrea Zonca
	"""

	M, N = a.shape
	m, n = new_shape
	if m<M:
		return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
	else:
		return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)

def save_fits(filename, img, **keywords):
	## def save_fits(filename, img, hdr=None, backup_dir='prev', target_dir='.',backend='astropy'):
	"""
	Saves the cube and its header.

	Saves the cube and its header. If the file already exists a backup is copied in the backup directory.
	Input:
	-filename: the filename to save to
	-img: the image array
	-hdr: the header
	"""

	if 'hdr' in keywords.keys():
		hdr=keywords['hdr']
	else:
		hdr=None

	if 'backup_dir' in keywords.keys():
		backup_dir=keywords['backup_dir']
	else:
		backup_dir='prev'

	if 'target_dir' in keywords.keys():
		target_dir=keywords['target_dir']
	else:
		target_dir='.'

	if not 'verify' in keywords.keys():
		verify='silentfix'
	elif keywords['verify']=='fix' or  keywords['verify']=='silentfix' or  keywords['verify']=='ignore' or  keywords['verify']=='warn' or keywords['verify']=='exception':
		verify=keywords['verify']
	else:
		verify='silentfix'

	if not os.path.isdir(target_dir): # Check if target dir exists
			os.mkdir(target_dir)

	if os.access(target_dir + os.sep +filename, os.F_OK ): # Check if file already exists
		if backup_dir==None:
			os.remove(target_dir + os.sep +filename)
		elif not os.path.isdir(target_dir + os.sep + backup_dir): # Check if backup dir exists
			os.mkdir(target_dir + os.sep + backup_dir)
		shutil.move(target_dir + os.sep +filename,target_dir + os.sep +backup_dir +  os.sep +filename) # move old file into backup dir

	# Save new file
	if 'backend' in keywords.keys() and keywords['backend']=='pyfits':
		## import pyfits
		from astropy.io import fits as pyfits

		if 'hdr' in keywords.keys():
			pyfits.writeto(target_dir + os.sep +filename, img, header=hdr, output_verify=verify)
		else:
			pyfits.writeto(target_dir + os.sep +filename, img, output_verify=verify)
	else:
		## img.verify(option='silentfix')
		img.writeto(target_dir + os.sep +filename,output_verify=verify)



def send_chunks(cub_in,d):
	"""
	Dispatches the _cubes_ to parallel processes.

	-cub_in the data_cube to send
	-d debug flag
	"""

	if cub_in.shape[1]/2.<(nprocs-1): # Send only two or three columns to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[1]:
				 comm.send(cub_in.shape[1], dest=n+1)
				 comm.send(None, dest=n+1)
			elif 2*n+3==cub.shape[1]: # Reaching the end of the cube, send all the remaining frames.
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[:,2*n:,:], dest = n+1 )
			else:
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[:,2*n:2*n+2,:], dest = n+1 )
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			if n+1 == nprocs-1:
				end=cub_in.shape[1]
			comm.send(start, dest = n+1 )
			comm.send(cub_in[:,start:end,:], dest = n+1)

			if d>3:
				print("Chunk "+str(n)+" sent, shape: "+str(cub_in[:,start:end,:].shape))
	del cub_in

def send_chunks_2D(cub_in,d):
	"""
	Dispatches the _frame_ to parallel processes.

	-cub_in the frame to send
	-d debug flag
	"""

	if cub_in.shape[1]/2.<(nprocs-1): # Send only two or three columns to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[1]:
				 comm.send(cube_in.shape[1], dest=n+1)
				 comm.send(None, dest=n+1)
			elif 2*n+3==cube.shape[1]: # Reaching the end of the cube, send all the remaining frames.
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[:,2*n:], dest = n+1 )
			else:
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[:,2*n:2*n+2], dest = n+1 )

			# check=comm.recv(source = n+1)
			# print("Chunk "+str(n)+" reception confirmed")
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			if n+1 == nprocs-1:
				end=cub_in.shape[1]
			comm.send(start, dest = n+1 )
			comm.send(cub_in[:,start:end], dest = n+1)

			if d>3:
				print("Chunk "+str(n)+" sent, shape: "+str(cub_in[:,start:end].shape))

	del cub_in


def send_dirlist(dirlist):
	"""
	Dispatches the dirlist to slaves, telling them what position in the dirlist they are starting at.
	It keeps the last part for the master.
	"""
	if len(dirlist) < nprocs:
		for n in range(nprocs):
			if n+1 == len(dirlist):
				start=n
				dirlist=dirlist[n:] # take the list to the end
				if n+1 < nprocs:
					for k in range(n+1,nprocs): # kill useless slaves
						comm.send(None, dest=k)
				break
			else:
				comm.send(dirlist[n:n+1], dest = n+1 )
				comm.send(n, dest=n+1)
	else:
		for n in range(nprocs):
			start=int(n*numpy.floor(float(len(dirlist))/nprocs))
			end=int((n+1)*numpy.floor(float(len(dirlist))/nprocs))
			#if end >= cub_in.shape[0]:
			if n == nprocs-1:
				dirlist=dirlist[start:] # take the list to the end
				break
			comm.send(dirlist[start:end], dest = n+1 )
			comm.send(start, dest=n+1)

	return start,dirlist

def send_dirlist_slaves(dirlist):
	"""
	Dispatches the dirlist to slaves, telling them what position in the dirlist they are starting at.

	TODO: change behaviour in if len(dirlist) < nprocs-1. Instead of giving last proc more files, send last proc less files (with min=2).
	"""
	## if len(dirlist) < nprocs-1:
		## for n in range(nprocs-1):
			## if n+1 == len(dirlist):
				## # take the list to the end
				## start=n
				## comm.send(dirlist[n:], dest = n+1 )
				## comm.send(n, dest=n+1)
				## if n+1 < nprocs:
					## for k in range(n+1,nprocs-1): # kill useless slaves
						## comm.send(None, dest=k+1)
				## break
			## else:
				## comm.send(dirlist[n:n+1], dest = n+1 )
				## comm.send(n, dest=n+1)
	if len(dirlist) < nprocs-1:
		for n in range(nprocs-1):
			if n+1 == len(dirlist):
				# take the list to the end
				start=n
				comm.send(dirlist[n:], dest = n+1 )
				comm.send(n, dest=n+1)
				if n+1 < nprocs:
					for k in range(n+1,nprocs-1): # kill useless slaves
						comm.send(None, dest=k+1)
				break
			else:
				comm.send(dirlist[n:n+1], dest = n+1 )
				comm.send(n, dest=n+1)
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.ceil(float(len(dirlist))/nprocs))
			end=int((n+1)*numpy.ceil(float(len(dirlist))/nprocs))
			#if end >= cub_in.shape[0]:
			if n+1 == nprocs-1:
				comm.send(dirlist[start:], dest = n+1 )
				## dirlist=dirlist[start:] # take the list to the end
				## break
			else:
				comm.send(dirlist[start:end], dest = n+1 )
			comm.send(start, dest=n+1)

	# return start,dirlist

def send_dirlist_slaves_prev(dirlist):
	"""
	Dispatches the dirlist to slaves, telling them what position in the dirlist they are starting at.

	Previous version. Well tested but not optimised.
	"""
	if len(dirlist) < nprocs-1:
		for n in range(nprocs-1):
			if n+1 == len(dirlist):
				# take the list to the end
				start=n
				comm.send(dirlist[n:], dest = n+1 )
				comm.send(n, dest=n+1)
				if n+1 < nprocs:
					for k in range(n+1,nprocs-1): # kill useless slaves
						comm.send(None, dest=k+1)
				break
			else:
				comm.send(dirlist[n:n+1], dest = n+1 )
				comm.send(n, dest=n+1)
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(len(dirlist))/nprocs))
			end=int((n+1)*numpy.floor(float(len(dirlist))/nprocs))
			#if end >= cub_in.shape[0]:
			if n+1 == nprocs-1:
				comm.send(dirlist[start:], dest = n+1 )
				## dirlist=dirlist[start:] # take the list to the end
				## break
			else:
				comm.send(dirlist[start:end], dest = n+1 )
			comm.send(start, dest=n+1)


def send_frames(cub_in):
	"""
	Dispatches the cubes to parallel processes, dividing the cube in cubes with less frames.

	If the number of process is higher than the number of frames, send only one frame to each process
	and None to the remainig ones.
	-cub_in the data_cube to send
	"""

	if cub_in.shape[0]/2.<(nprocs-1): # Send only two or three frames to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[0]:
				comm.send(cub_in.shape[0], dest=n+1)
				comm.send(None, dest=n+1)
			elif 2*n+3==cub_in.shape[0]: # Reaching the end of the cube, send all the remaining frames.
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[2*n:,:,:], dest = n+1 )
			else:
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in[2*n:2*n+2,:,:], dest = n+1 )
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[0])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[0])/(nprocs-1)))
			#if end >= cub_in.shape[0]:
			if n+1 == nprocs-1:
				end=cub_in.shape[0]
			comm.send(start, dest = n+1 )
			comm.send(cub_in[start:end,:,:], dest = n+1 )

	del cub_in

def send_frames_async(cub_in):
	"""
	Dispatches the cubes to parallel processes, dividing the cube in cubes with less frames.

	If the number of process is higher than the number of frames, send only one frame to each process
	and None to the remainig ones.
	-cub_in the data_cube to send
	"""

	if cub_in.shape[0]/2.<(nprocs-1): # Send only two or three frames to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[0]:
				r1=comm.isend(cub_in.shape[0], dest=n+1)
				r2=comm.isend(None, dest=n+1)
			elif 2*n+3==cub_in.shape[0]: # Reaching the end of the cube, send all the remaining frames.
				r1=comm.isend(2*n, dest = n+1 )
				r2=comm.isend(cub_in[2*n:,:,:], dest = n+1 )
			else:
				r1=comm.isend(2*n, dest = n+1 )
				r2=comm.isend(cub_in[2*n:2*n+2,:,:], dest = n+1 )
			if n==0:
				request=[r1,r2]
			else:
				request.append(r1)
				request.append(r2)
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[0])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[0])/(nprocs-1)))
			#if end >= cub_in.shape[0]:
			if n+1 == nprocs-1:
				end=cub_in.shape[0]
			r1=comm.isend(start, dest = n+1 )
			r2=comm.isend(cub_in[start:end,:,:], dest = n+1 )
			if n==0:
				request=[r1,r2]
			else:
				request.append(r1)
				request.append(r2)
	# Wait for all transfers to be finished
	MPI.Request.Waitall(request)
	del cub_in

def send_masked_chunks(cub_in,d):
	"""
	Dispatches the cubes to parallel processes.

	-cub_in the data_cube to send
	-d debug flag
	"""

	if cub_in.shape[1]/2.<(nprocs-1): # Send only two or three columns to each proc
		for n in range(nprocs-1):
			if 2*n+2 > cub_in.shape[1]:
				 comm.send(cube_in.shape[1], dest=n+1)
				 comm.send(None, dest=n+1)
				 comm.send(None, dest=n+1)

			elif 2*n+3==cube.shape[1]: # Reaching the end of the cube, send all the remaining frames.
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in.data[:,2*n:,:], dest = n+1 )
				comm.send(cub_in.mask[:,2*n:,:], dest = n+1 )
			else:
				comm.send(2*n, dest = n+1 )
				comm.send(cub_in.data[:,2*n:2*n+2,:], dest = n+1 )
				comm.send(cub_in.mask[:,2*n:2*n+2,:], dest = n+1 )

			# check=comm.recv(source = n+1)
			# print("Chunk "+str(n)+" reception confirmed")
	else:
		for n in range(nprocs-1):
			start=int(n*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			end=int((n+1)*numpy.floor(float(cub_in.shape[1])/(nprocs-1)))
			if n+1 == nprocs-1:
				end=cub_in.shape[1]
			comm.send(start, dest = n+1 )
			print(str(rank)+" start: "+str(start))
			print(str(rank)+" end: "+str(end))
			print(str(rank)+" cub_in: "+str(cub_in))
			print(str(rank)+" cub_in.mask: "+str(cub_in.mask))

			comm.send(cub_in.data[:,start:end,:], dest = n+1)
			comm.send(cub_in.mask[:,start:end,:], dest = n+1)

			if d>1:
				print("Chunk "+str(n)+" sent, shape: "+str(cub_in[:,start:end,:].shape))

	del cub_in


def shift_diff(shift, rw, shift_im ,x_start, x_end, y_start, y_end,R):
	"""
	Calculate the difference between the reference window (rw) and the shifted image (shift_im)

	-shift tuple containing x and y shift to apply
	-xs,ys x position of the reference window start
	-xe,ye y position of the reference window end
	-R the window size
	"""

	# Cut out the window
	sw = shift_im[x_start:x_end,y_start:y_end]
	# Set to zero saturated and background pixels
	# sw = numpy.where(sw>0.28,0,sw) #desaturate
	# sw = numpy.where(sw<0.02,0,sw) #set background to 0
	# rw = numpy.where(rw>0.28,0,rw) #desaturate
	# rw = numpy.where(rw<0.02,0,rw) #set background to 0

	fsw=numpy.fft.fft2(sw)
	#	print(fsw.shape)
	n=numpy.fft.fftfreq(fsw.shape[0])

	u=numpy.exp(-2j*numpy.pi*shift[0]*n).reshape(1,fsw.shape[0])
	v=numpy.exp(-2j*numpy.pi*shift[1]*n).reshape(fsw.shape[1],1)

	fsw=fsw*u*v

	diff_im=numpy.real(numpy.fft.ifft2(numpy.fft.fft2(rw)-fsw))

	## cor=correlate2d(rw,sw,mode='same')
	## mass=cor.sum()
	## xc=0
	## yc=0
	## for i in range(cor.shape[0]):
	##	 xc=xc+(i+1-cor.shape[0]/2.)*cor[i,:].sum()
	## for i in range(cor.shape[1]):
	##	 yc=yc+(i+1-cor.shape[1]/2.)*cor[:,i].sum()

	##	 x_shift=xc/mass
	##	 y_shift=yc/mass


	return diff_im.flatten()


def shift_diff_interpol(shift, rw, shift_im ,x_start, x_end, y_start, y_end,R):
	"""
	Calculate the difference between the reference window (rw) and the shifted image (shift_im) using interpolation shift

	-shift tuple containing x and y shift to apply
	-xs,ys x position of the reference window start
	-xe,ye y position of the reference window end
	-R the window size
	"""

	# Cut out the window
	sw = shift_im[x_start:x_end,y_start:y_end]
	# Set to zero saturated and background pixels
	# sw = numpy.where(sw>0.28,0,sw) #desaturate
	# sw = numpy.where(sw<0.02,0,sw) #set background to 0
	# rw = numpy.where(rw>0.28,0,rw) #desaturate
	# rw = numpy.where(rw<0.02,0,rw) #set background to 0

	## fsw=numpy.fft.fft2(sw)
	## #	print(fsw.shape)
	## n=numpy.fft.fftfreq(fsw.shape[0])

	## u=numpy.exp(-2j*numpy.pi*shift[0]*n).reshape(1,fsw.shape[0])
	## v=numpy.exp(-2j*numpy.pi*shift[1]*n).reshape(fsw.shape[1],1)

	## fsw=fsw*u*v

	fsw=ndimage.interpolation.shift(sw, (shift[0],shift[1]), order=3, mode='constant', cval=0.0, prefilter=False)

	ratio=shift[2]

	diff_im=rw-ratio*fsw

	## cor=correlate2d(rw,sw,mode='same')
	## mass=cor.sum()
	## xc=0
	## yc=0
	## for i in range(cor.shape[0]):
	##	 xc=xc+(i+1-cor.shape[0]/2.)*cor[i,:].sum()
	## for i in range(cor.shape[1]):
	##	 yc=yc+(i+1-cor.shape[1]/2.)*cor[:,i].sum()

	##	 x_shift=xc/mass
	##	 y_shift=yc/mass


	return diff_im.flatten()

def shift_diff_interpol_nowin(shift, rw, sw ):
	"""
	Calculate the difference between the reference window (rw) and the shifted image (shift_im) using interpolation shift

	-shift tuple containing x and y shift to apply
	"""

	# Cut out the window
	fsw=ndimage.interpolation.shift(sw, (shift[0],shift[1]), order=3, mode='constant', cval=0.0, prefilter=False)

	# Disabelign ratio optimisation
	shift[2]=1

	ratio=shift[2]

	## if numpy.ma.isMaskedArray(rw): # remove the mask befor substracting
	##	 diff_im=rw.data-ratio*fsw
	## else:
	diff_im=rw-ratio*fsw

	return diff_im.flatten()

def trim_overscan(cube):
	"""
	Trims the overscan of a cube to make it square
	"""
	cube_shape=cube.shape

	if cube.shape[1]-cube.shape[2]==2:
		overscan_limit=cube.shape[2]
		cube=cube[:,:overscan_limit,:]
	elif cube.shape[2]-cube.shape[1]==2:
		overscan_limit=cube.shape[1]
		cube=cube[:,:overscan_limit,:]
	else:
		print('Error! No overscan detected for: '+str(dirlist[i]))

	return cube

def write_log(runtime, log_file, comments=None):
	"""
	Adds the executed command to the logfile
	from subprocess import Popen, PIPE
	"""
	from datetime import datetime
	import string
	from subprocess import Popen, PIPE
	import sys

	if rank==0:
		p=Popen(["ps","-o", "cmd=","-p",str(os.getpid())], stdout=PIPE)
		out, err = p.communicate()
		f = open(log_file, 'aw')
		f.write(string.replace(string.zfill('0',80),'0','-')+'\n')
		f.write('mpirun -n '+str(nprocs)+' '+str(out))
		if not comments==None:
			if isinstance(comments,list):
				for c in comments:
					f.write(c+'\n')
			elif isinstance(comments,str):
				f.write(comments)
			else:
				print('Unknown '+type(comments)+': '+str(comments))
		f.write(string.join(sys.argv)+'\n')
		f.write('Job finished on: '+datetime.isoformat(datetime.today())[:-7]+'. Total time: '+humanize_time(runtime)+'\n')
		f.close()


def write_array2rdb(filename,data,keys,s=''):
	import string

	if s=='':
		for k in range(len(keys)):
			s=s+'%F\t'

		s=s+'\n'

	f = open(filename, 'w')

	head1 = string.join(keys,'\t')
	head2 = ''
	for i in head1:
		if i=='\t': head2 = head2+'\t'
		else: head2 = head2+'-'

	f.write(head1+'\n')
	f.write(head2+'\n')

	#try:
	if True:
		if len(data.shape)==2:
			for i in range(data.shape[0]):
				line = []
				for j in range(len(keys)): line.append(data[i][j])
				f.write(s % tuple(line))
	## except:
		## print("Error, expected 2d array to write rdb file.")

	f.close()

def write_rdb(filename,data,keys,format):
	import string

	f = open(filename, 'w')

	head1 = string.join(keys,'\t')
	head2 = ''
	for i in head1:
		if i=='\t': head2 = head2+'\t'
		else: head2 = head2+'-'

	f.write(head1+'\n')
	f.write(head2+'\n')

	if len(data.values()) > 0:
		for i in range(len(data.values()[0])):
			line = []
			for j in keys: line.append(data[j][i])
			f.write(format % tuple(line))

	f.close()
