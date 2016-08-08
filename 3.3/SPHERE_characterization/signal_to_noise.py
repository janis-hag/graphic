import pyfits,glob
import numpy as np
#import pylab as py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as py


def signal_to_noise(path,path_results,pattern,plate_scale,fwhm_pl):
	for i,allfiles in enumerate(glob.iglob(pattern+"*")):
		if i==0:
			image_filename=allfiles
			print "image filename:",image_filename
		else:
			print "Error more than one file found with this pattern. Used the first one:",image_filename
	im,hdr=pyfits.getdata(path+image_filename,header=True)

	target_name=hdr["OBJECT"]
	"""Radius_to_hide=40
	x_t=np.arange(-np.shape(im)[0]/2,np.shape(im)[0]/2)
	y_t=x_t
	X_t,Y_t=np.meshgrid(x_t,y_t)
	R_t=np.sqrt(X_t**2+Y_t**2)
	mask_t=np.where(R_t<Radius_to_hide,np.nan,1)"""

	import os.path
	if os.path.isfile(path_results+"companion_extraction.txt"):
		f=open(path_results+"companion_extraction.txt",'r')
		lines=f.readlines()
		f.close()
		for line in lines:
			if line.strip().split()[0]!='filter' and line.strip().split()[0][0]!='-':
				rho_comp=float(line.strip().split()[1])/plate_scale
				PA_comp=float(line.strip().split()[3])*np.pi/180.
				xpl=-np.sin(PA_comp)*rho_comp+np.shape(im)[1]/2.
				ypl=np.cos(PA_comp)*rho_comp+np.shape(im)[0]/2.
		
		xpl=int(xpl-np.shape(im)[1]/2)
		ypl=int(ypl-np.shape(im)[0]/2)

		angle_mask=20 #in degree
		angle_adi_subtraction=15

		#quadrant determination
		if xpl>=0:
			if ypl>=0:
				quad=1
			else:
				quad=3
		else:
			if ypl>=0:
				quad=2
			else:
				quad=4

		tangente_pl=(1.*ypl)/xpl
		index_to_center=[xpl,ypl]


		r1=np.sqrt(xpl**2+ypl**2)-3*fwhm_pl
		r2=np.sqrt(xpl**2+ypl**2)+3*fwhm_pl
		x=np.arange(-np.shape(im)[0]/2,np.shape(im)[0]/2)
		y=x
		X,Y=np.meshgrid(x,y)
		R=np.sqrt(X**2+Y**2)

		donut=np.where(R<r1,np.nan,1)
		donut=donut*np.where(R>r2,np.nan,1)
		tangente=(1.*Y)/(1.*X)
		Theta=(5*np.pi/2.-np.arctan2(X,Y)) %(2.*np.pi)
		theta=-((5*np.pi/2.-np.arctan2(xpl,ypl)) %(2.*np.pi))
		if (-theta)*180/np.pi-angle_mask<0:
			mask1=np.where(Theta<2*np.pi-theta-angle_mask*np.pi/180.,0,1)
			mask2=np.where(Theta>-theta+angle_mask*np.pi/180.,0,1)
			mask=np.where(mask1+mask2==0,np.nan,1)
		elif (-theta)*180/np.pi+angle_mask>360:
			mask1=np.where(Theta<-theta-angle_mask*np.pi/180.,0,1)
			mask2=np.where(Theta>-theta+angle_mask*np.pi/180.-2*np.pi,0,1)
			mask=np.where(mask1+mask2==0,np.nan,1)
		else:
			mask=np.where(Theta<-theta-angle_mask*np.pi/180.,np.nan,1)
			mask=mask*np.where(Theta>-theta+angle_mask*np.pi/180.,np.nan,1)
		mask=donut*mask

		r1_adi_sub=np.sqrt(xpl**2+ypl**2)-0.5*fwhm_pl
		r2_adi_sub=np.sqrt(xpl**2+ypl**2)+0.5*fwhm_pl
		donut_adi_subtraction=np.where(R<r1_adi_sub,np.nan,1)
		donut_adi_subtraction=donut_adi_subtraction*np.where(R>r2_adi_sub,np.nan,1)
		if (-theta)*180/np.pi-angle_adi_subtraction<0:
			mask1_adi_subtraction=np.where(Theta<2*np.pi-theta-angle_adi_subtraction*np.pi/180.,0,1)
			mask2_adi_subtraction=np.where(Theta>-theta+angle_adi_subtraction*np.pi/180.,0,1)
			mask_adi_subtraction=np.where(mask1_adi_subtraction+mask2_adi_subtraction==0,np.nan,1)
		elif (-theta)*180/np.pi+angle_adi_subtraction>360:
			mask1_adi_subtraction=np.where(Theta<-theta-angle_adi_subtraction*np.pi/180.,0,1)
			mask2_adi_subtraction=np.where(Theta>-theta+angle_adi_subtraction*np.pi/180.-2*np.pi,0,1)
			mask_adi_subtraction=np.where(mask1_adi_subtraction+mask2_adi_subtraction==0,np.nan,1)
		else:
			mask_adi_subtraction=np.where(Theta<-theta-angle_adi_subtraction*np.pi/180.,np.nan,1)
			mask_adi_subtraction=mask_adi_subtraction*np.where(Theta>-theta+angle_adi_subtraction*np.pi/180.,np.nan,1)
		mask_adi_subtraction=donut_adi_subtraction*mask_adi_subtraction
		mask_adi_subtraction=np.where(np.isnan(mask_adi_subtraction),1,np.nan)


		x2=x-xpl
		y2=y-ypl
		X2,Y2=np.meshgrid(x2,y2)
		R2=np.sqrt(X2**2+Y2**2)
		mask_pl=np.where(R2<=fwhm_pl,np.nan,1)

		py.close("all")
		py.figure(1)
		py.imshow(mask*im,origin='lower')
		py.savefig(path_results+"companion_S_N.png",dpi=300)
		py.figure(2)
		py.imshow(im*mask_adi_subtraction*mask_pl*mask,origin='lower')
		py.savefig(path_results+"companion_masked_S_N.png", dpi=300)


		Signal=np.nanmax(im*mask)-np.nanmedian(im*mask_pl*mask_adi_subtraction*mask)
		Noise=np.nanstd(im*mask_pl*mask_adi_subtraction*mask)
		print "Signal: ", Signal
		print "Noise: ", Noise
		print "S/N= ",Signal/Noise
		
		f=open(path_results+"companion_extraction.txt","r")
		lines=f.readlines()
		f.close()
		
		f=open(path_results+"companion_extraction.txt","w")
		for i,line in enumerate(lines):
			if line!="\n":
				if line.strip().split()[0]=="filter":
					if line.strip().split()[-1]!="Signal/Noise":
						header=line.strip().split()
						header.append("Signal/Noise")
						line="\t".join(header)+"\n"
				elif line.strip().split()[0][0]!="-" and i==np.size(lines)-1:
					comp_info=line.strip().split()
					comp_info.append(str(round(Signal/Noise,3)))
					line="\t".join(comp_info)+"\n"
			f.write(line)
		f.close()
	else:
		print "No companion detected!"





