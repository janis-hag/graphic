import pyfits
import numpy as np
import pylab as py
from scipy import fftpack
from scipy import signal



def low_pass(image, r, order, cut_off):
	fft=fftpack.fftshift(fftpack.fft2(np.nan_to_num(np.abs(image))))
	l=np.shape(image)[1]
	x=np.arange(-l/2,l/2)
	y=np.arange(-l/2,l/2)
	X,Y=np.meshgrid(x,y)
	R = np.sqrt(X**2 + Y**2)
	R_ent=np.round(R) #partie entiere

	B,A=signal.butter(order,cut_off/(l/2.)-r/(l/2.))
	z=np.zeros(l)
	z[0]=1.
	zf=signal.lfilter(B,A,z)
	fft_zf=fftpack.fftshift(fftpack.fft(zf))
	fft_zf = fft_zf[l/2.:l]

	F_bas=np.zeros((l,l))+0j
	for i in np.arange(l):
		for j in np.arange(l):
			if R_ent[i,j] < l/2.:
				F_bas[i,j] = np.abs(fft_zf[R_ent[i,j]])

	#~ py.figure(1)
	#~ py.imshow(np.real(F_bas))
	#~ py.figure(2)
	#~ py.plot(np.abs(F_bas[l/2.,l/2.:]))
	#~ py.plot(np.arange(l/2.),np.sqrt(0.5)*np.ones(l/2.))
	#~ py.show()

	f_bas=fftpack.ifftshift(fft*F_bas)
	im_bis=np.real(fftpack.ifft2(f_bas))
	return im_bis



def high_pass(image, r, order, cut_off):
	fft=fftpack.fftshift(fftpack.fft2(image))
	l=np.shape(image)[1]
	x=np.arange(-l/2,l/2)
	y=np.arange(-l/2,l/2)
	X,Y=np.meshgrid(x,y)
	R = np.sqrt(X**2 + Y**2)
	R_ent=np.round(R) #partie entiere

	B,A=signal.butter(order,cut_off/(l/2.)-r/(l/2.),btype='high')
	z=np.zeros(l)
	z[0]=1.
	zf=signal.lfilter(B,A,z)
	fft_zf=fftpack.fftshift(fftpack.fft(zf))
	fft_zf = fft_zf[l/2.:l]

	F_haut=np.ones((l,l))+0j
	for i in np.arange(l):
		for j in np.arange(l):
			if R_ent[i,j] < l/2.:
				F_haut[i,j] = np.abs(fft_zf[R_ent[i,j]])


	#~ py.figure(3)
	#~ py.imshow(F_haut.real)
	#~ py.figure(4)
	#~ py.plot(np.abs(F_haut[l/2.:,l/2.]))
	#~ py.plot(np.arange(l/2.),np.sqrt(0.5)*np.ones(l/2.))
#~
	#~ py.show()

	f_haut=fftpack.ifftshift(fft*F_haut)
	im_bis=np.real(fftpack.ifft2(f_haut))
	return im_bis
