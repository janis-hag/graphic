from SPHERE_psf_combine import *
from primary_flux import *
from mpi4py import MPI
import argparse

rank   = MPI.COMM_WORLD.Get_rank()

target_dir = "."
wdir='./'

parser = argparse.ArgumentParser(description='Preparation for the injection of fake companion (combine psf and compute primary flux)')
parser.add_argument('--pattern_left', action="store", dest="pattern_left",  default="nomask*left", help='pattern for final image on which to compute the primary flux')
parser.add_argument('--pattern_right', action="store", dest="pattern_right",  default="nomask*right", help='pattern for final image on which to compute the primary flux')
parser.add_argument('--pattern_psf_l', action="store", dest="pattern_psf_l",  default="left*FLUX", help='pattern for flux frames to median combine')
parser.add_argument('--pattern_psf_r', action="store", dest="pattern_psf_r",  default="right*FLUX", help='pattern for flux frames to median combine')

args = parser.parse_args()
pattern_left=args.pattern_left
pattern_right=args.pattern_right
pattern_psf_l=args.pattern_psf_l
pattern_psf_r=args.pattern_psf_r

if rank==0:
	print "\nStart of psf combine\n"
	psf_combine(wdir,pattern_psf_l,pattern_psf_r)

	print "\nStart of psf primary flux determination\n"
	primary_flux(wdir,pattern_left)
	primary_flux(wdir,pattern_right)

	print "finished, psf_left.fits and psf_right.fits are produced and primary fluxes for left and right images are stored in primary_flux.txt"
	os._exit(1)
else:
	sys.exit(1)
