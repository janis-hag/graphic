import numpy as np
import pyfits as pf
import sys, glob
from Neutral_density import *

def primary_flux(wdir,pattern):
	for i,allfiles in enumerate(glob.iglob(pattern+"*")):
		if i==0:
			image_filename=allfiles
		else:
			print "Warning more than one file found with this pattern name. Used the first one:",image_filename
	
	if ("left" in image_filename) or ("sdi" in image_filename):
		flux_filename='psf_left.fits'
	elif ("right" in image_filename):
		flux_filename='psf_right.fits'
	else:
		print "Error could not find if left or right or sdi file for psf filename"
	image_file=wdir+image_filename
	flux_file=wdir+flux_filename
	# Load the image and the flux image
	image,hdr=pf.getdata(image_file,header=True)
	flux_image,hdr_flux=pf.getdata(flux_file,header=True)

	primary_flux=np.nanmax(flux_image)

	DIT_flux=hdr_flux['HIERARCH ESO DET SEQ1 DIT']
	DIT_image=hdr['HIERARCH ESO DET SEQ1 DIT']

	band_filter_image=hdr['HIERARCH ESO INS COMB IFLT']
	band_filter_flux=hdr_flux['HIERARCH ESO INS COMB IFLT']


	lambda_image=band_filter(band_filter_image,image_filename)
	lambda_flux=band_filter(band_filter_flux,flux_filename)

	Neutral_density_image=1./Neutral_density(hdr,lambda_image)
	Neutral_density_flux=1./Neutral_density(hdr_flux,lambda_flux)

	primary_flux=(primary_flux*Neutral_density_flux*DIT_image)/(Neutral_density_image*DIT_flux)
	f=open(wdir+"primary_flux.txt","a")
	if ("left" in image_filename) or ("sdi" in image_filename):
		f.write("left_image:\t"+str(primary_flux)+"\n")
	elif ("right" in image_filename):
		f.write("right_image:\t"+str(primary_flux)+"\n")
	f.close()
