# -*- coding: utf-8 -*-
import pyfits
import numpy as np
import pylab as py
import scipy
from scipy import ndimage
from scipy import fftpack

def Calib(xmag,ymag,xrotation,yrotation,exmag,eymag,exrotation,eyrotation,dxpix,dypix,edxpix,edypix,f1,f2,ef1,ef2):
    
    b =  xmag * np.cos(xrotation)
    c =  ymag * np.sin(yrotation)
    e = -xmag * np.sin(xrotation)
    f =  ymag * np.cos(yrotation)
    eb=np.sqrt((exmag*np.cos(xrotation))**2 + (xmag*np.sin(xrotation)*exrotation)**2 )   
    ec=np.sqrt((eymag*np.sin(yrotation))**2 + (ymag*np.cos(yrotation)*eyrotation)**2 )   
    ee=np.sqrt((exmag*np.sin(xrotation))**2 + (xmag*np.cos(xrotation)*exrotation)**2 )   
    ef=np.sqrt((eymag*np.cos(yrotation))**2 + (ymag*np.sin(yrotation)*eyrotation)**2 )
    
    
    xi_arcsec_mod  =   b * dxpix + c * dypix
    eta_arcsec_mod =   e * dxpix + f * dypix
    
    exi_arcsec_mod  =  np.sqrt( (eb * dxpix)**2 + (ec * dypix)**2  +  (b * edxpix)**2 + (c * edypix)**2)
    eeta_arcsec_mod =  np.sqrt( (ee * dxpix)**2 + (ef * dypix)**2  +  (e * edxpix)**2 + (f * edypix)**2)
    
    rho=np.sqrt(xi_arcsec_mod**2+eta_arcsec_mod**2)
    
    PA=(2*np.pi-np.arctan2(xi_arcsec_mod,eta_arcsec_mod)) %(2.*np.pi);
    
    #montecarlo
    
    s=np.random.randn(2000)
    dxpix_monte=(dxpix+s*edxpix)
    dypix_monte=(dypix+s*edypix)
    
    xi_arcsec_mod_monte  =   b * dxpix_monte + c * dypix_monte
    eta_arcsec_mod_monte =   e * dxpix_monte + f * dypix_monte
    
    rho_monte=np.sqrt(xi_arcsec_mod_monte**2+eta_arcsec_mod_monte**2)
    
    PA_monte=(2.0*np.pi-np.arctan2(xi_arcsec_mod_monte,eta_arcsec_mod_monte)) %(2.0*np.pi);
    #atan(xi_arcsec_mod,eta_arcsec_mod)/pi*180.
    #atan(eta_arcsec_mod,xi_arcsec_mod)/pi*180.
    #PROBLEME DU CALCUL D ERREUR LORSQU ON EST PROCHE DE 90 DONC CALCUL DIFFERENT VERIFIER AVEC DAMIEN
    #PA_test=span(0,360,361)*pi/180.
    
    erho=1./rho* (np.sqrt( (xi_arcsec_mod *exi_arcsec_mod)**2) + np.sqrt((eta_arcsec_mod *eeta_arcsec_mod)**2))
    ePA = 1./np.sqrt(1.0+(xi_arcsec_mod/eta_arcsec_mod)**2) *  np.sqrt( (exi_arcsec_mod/eta_arcsec_mod)**2 + (eeta_arcsec_mod*xi_arcsec_mod/eta_arcsec_mod**2)**2)
    ePA1 = np.arctan(1./rho* (np.sqrt((eeta_arcsec_mod*np.sin(PA))**2)+np.sqrt((exi_arcsec_mod*np.cos(PA))**2)))
    ePA2 = 1./rho* (np.sqrt((eeta_arcsec_mod*np.sin(PA))**2)+np.sqrt((exi_arcsec_mod*np.cos(PA))**2))
    
    dmag  = abs(-2.5*np.log(f2/f1)/np.log(10))
    edmag =  2.5/np.log(10)*(f1/f2)*np.sqrt((ef1*f2/f1**2)**2+(ef2/f1)**2)
    
    """print "rho",rho
    print "rho_monte",np.median(rho_monte)
    print "erho",erho
    print "rms_rho",np.std(rho_monte)
    print "PA",PA*180.0/np.pi
    print "PA_monte",np.median(PA_monte)*180/np.pi
    print "ePA",ePA2*180/np.pi
    print "rms_PA",np.std(PA_monte)*180./np.pi
    print "dmag",dmag
    print "edmag",edmag"""
    

    return rho,erho,PA,ePA2,dmag,edmag
    
    