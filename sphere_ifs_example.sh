#!/bin/bash
#SBATCH   --mail-user=sebastien.peretti@unige.ch --mail-type=ALL --error=error_output_%j

#module add openmpi/gcc/1.6
#module add python/2.6.6

# Change these values to easily update all of the calls to the following code
GRAPHIC_VERSION="3.3"
GRAPHIC_DIR="/Users/cheetham/code/graphic_git/"$GRAPHIC_VERSION"/"
GRAPHIC_N_CORES="" #for running on a cluster
# GRAPHIC_N_CORES="-n 6" # for running on a laptop. Make sure n_cores is less than or equal to the number of files!


for ix in {0..38}
do
	cd "wavelength"$ix"/"

	# Print a message so we know where we're up to
	echo
	echo "########"
	echo "Wavelength Channel "$ix
	echo "########"
	echo

	#####################
	# Make the images have an even number of pixels. This step is mandatory or the final astrometry will be wrong!
	#####################
	mpirun $GRAPHIC_N_CORES $GRAPHIC_OPTIONS python $GRAPHIC_DIR"GRAPHIC_rm_overscan_"$GRAPHIC_VERSION".py" --pattern wav*SPHER --l_max 290 --centre_offset 145 145

	#####################
	#register
	#####################
	mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern *SPHER*SCIENCE -nofit -no_psf
	mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern *SPHER*STAR_CENTER -nofit -no_psf


	# #####################
	# #Star center
	# #####################
	# The manual values below are here because the star is always at the same position for IFS, so there is no point calculating it
	mpirun -mca btl ^openib python $GRAPHIC_DIR"star_center_sphere_science_waffle.py" --pattern o*SPHER*STAR_CENTER -ifs --lowpass_r 30 --manual_rough_centre 144 142

	# #####################
	# #Cut and center frames
	# #####################
	mpirun -mca btl ^openib python $GRAPHIC_DIR"cut_center_cube_sphere_science_waffle.py" --pattern o_wav*SPHER*SCIENCE -ifs

	# #####################
	# #Register the cubes if not done before
	# #####################
	mpirun -mca btl ^openib python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern cen_*SCIENCE -nofit -no_psf

	# #####################
	# #frame selection (detection of bad frames and writing of file frame_selection.txt)
	# #####################
	mpirun -mca btl ^openib python $GRAPHIC_DIR"frame_selection_auto.py" --pattern cen*SCIENCE

	# #####################
	# #frame selection (removing the bad frames)
	# #####################
	mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"frame_selection.py" --pattern cen_*SCIENCE --info_dir cube-info --info_pattern all_info_20_5_300_

	# #####################
	# # Centre the flux frames, run frame selection and bin them
	# #####################
	GRAPHIC_N_FLUX="$(ls -l wav*FLUX*.fits | wc -l)" # Number of flux cubes to bin together.
	mpirun -mca btl ^openib python $GRAPHIC_DIR"GRAPHIC_naco_register_3.3.py" -sphere --pattern o_wav*FLUX
	mpirun python $GRAPHIC_DIR"GRAPHIC_frame_selection_"$GRAPHIC_VERSION".py" --pattern o_wav*FLUX --info_pattern all_info_*FLUX --centering_nsigma 1e5 --flux_nsigma 5 --psf_width_nsigma 5 --debug 3
	mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_recenter_cubes_"$GRAPHIC_VERSION".py" --pattern o_wav*FLUX --info_pattern all_info_framesel*FLUX --naxis3 $GRAPHIC_N_FLUX --lmax 40 -combine_frames
	mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_convert_for_pca.py" --pattern rc_o*FLUX --output_dir "./ADI/" -skip_parang --output_file 'flux.fits'


	# #####################
	# # Convert for pca and run PCA algo
	# #####################
	mpirun python $GRAPHIC_DIR"GRAPHIC_convert_for_pca.py" --pattern cen_o*SPHERE --output_dir "./ADI/"
	# # # # The PCA algorithm should be run separately because we need to know what the FWHM is first!
	cd ADI
		python $GRAPHIC_DIR"GRAPHIC_pca.py" --output_dir "./GRAPHIC_PCA/" --threads 6 --arc_length 1e4 --n_annuli 6 -median_combine --n_fwhm 0.75 --n_modes 10 --min_reference_frames 10 --fwhm 4. --r_min 3. --r_max 100 >> pca_output.txt
	cd ..

	# #####################
	# # Calculate the contrast curve (work in progress)
	# #####################

    cd ..
done