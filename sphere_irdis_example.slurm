#!/bin/bash
#SBATCH   --mail-user=sebastien.peretti@unige.ch --mail-type=ALL --error=error_output_%j

#module add openmpi/gcc/1.6
#module add python/2.6.6

# Change these values to easily update all of the calls to the following code
GRAPHIC_VERSION="3.3"
GRAPHIC_DIR="/Users/cheetham/code/graphic_git/"$GRAPHIC_VERSION"/"
GRAPHIC_N_CORES="" #for running on a cluster
GRAPHIC_N_CORES="-n 2" # for running on a laptop. Make sure n_cores is less than or equal to the number of files!

#####################
#register
#####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern SPHER*SKY -nofit -no_psf
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern SPHER*FLUX -nofit -no_psf
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern SPHER*SCIENCE -nofit -no_psf
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern SPHER*STAR_CENTER -nofit -no_psf

# #####################
# #Create the master dark
# #####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_n_NaNmedian_3.3.py" --pattern SPHER*SKY --win 0 --info_pattern all_info_ -nofit

# #####################
# #Create the master flat
# #####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"master_flat_sphere.py" --pattern SPHER*FLAT

# #####################
# #Subtract the sky from each valid frame and divide the flats
# #####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_sky_sub_3.3.py" -sphere --pattern SPHER*FLUX --sky_pattern med --sky_dir sky-num --info_dir cube-info --info_pattern all_info_ -nofit --flat_filename master_flat.fits
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_sky_sub_3.3.py" -sphere --pattern SPHER*STAR_CENTER --sky_pattern med --sky_dir sky-num --info_dir cube-info --info_pattern all_info_ -nofit --flat_filename master_flat.fits
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_sky_sub_3.3.py" -sphere --pattern SPHER*SCIENCE --sky_pattern med --sky_dir sky-num --info_dir cube-info --info_pattern all_info_ -nofit --flat_filename master_flat.fits

# #####################
# # Clean bad pixels using the sky frames as reference
#####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_bad_pix_3.3.py" -sphere --flat_filename master_flat.fits -bottleneck --pattern nomed --dark_pattern med --dark_dir sky-num --coef 10

# #####################
# #Star center
# #####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"star_center_sphere_science_waffle.py" --pattern cl*SPHER*STAR_

# #####################
# #Cut and center frames (duplicate the rdb files into left and right too)
# #####################
# # # # mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"cut_center_cube_sphere_science_waffle.py" --pattern cl_nomed_SPHER*STAR_CENTER # no reason for this
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"cut_center_cube_sphere_science_waffle.py" --pattern cl_nomed_SPHER*FLUX
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"cut_center_cube_sphere_science_waffle.py" --pattern cl_nomed_SPHER*SCIENCE

# #####################
# # Centre the flux frames, run frame selection and bin them
# #####################
GRAPHIC_N_FLUX="$(ls -l left*FLUX*.fits | wc -l)" # Number of flux cubes to bin together.
mpirun -mca btl ^openib python $GRAPHIC_DIR"GRAPHIC_naco_register_3.3.py" -sphere --pattern left*FLUX
mpirun -mca btl ^openib python $GRAPHIC_DIR"GRAPHIC_naco_register_3.3.py" -sphere --pattern right*FLUX
mpirun $GRAPHIC_OPTIONS python $GRAPHIC_DIR"GRAPHIC_frame_selection_"$GRAPHIC_VERSION".py" --pattern left*FLUX --info_pattern all_info_*left*FLUX --centering_nsigma 1e5 --flux_nsigma 5 --psf_width_nsigma 5 --debug 3
mpirun $GRAPHIC_OPTIONS python $GRAPHIC_DIR"GRAPHIC_frame_selection_"$GRAPHIC_VERSION".py" --pattern right*FLUX --info_pattern all_info_*right*FLUX --centering_nsigma 1e5 --flux_nsigma 5 --psf_width_nsigma 5 --debug 3
mpirun $GRAPHIC_N_CORES $GRAPHIC_OPTIONS python $GRAPHIC_DIR"GRAPHIC_recenter_cubes_"$GRAPHIC_VERSION".py" --pattern left*FLUX --info_pattern all_info_framesel*left*FLUX --naxis3 $GRAPHIC_N_FLUX  --lmax 40 -combine_frames
mpirun $GRAPHIC_N_CORES $GRAPHIC_OPTIONS python $GRAPHIC_DIR"GRAPHIC_recenter_cubes_"$GRAPHIC_VERSION".py" --pattern right*FLUX --info_pattern all_info_*framesel*right*FLUX --naxis3 $GRAPHIC_N_FLUX --lmax 40 -combine_frames

# #####################
# # Convert the flux frames so they are ready for PCA (i.e. in the SPECAL/PCA/PHISHI format)
# #####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_convert_for_pca.py" --pattern rc_left --output_dir "left/" -skip_parang --output_file 'flux.fits'
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_convert_for_pca.py" --pattern rc_right --output_dir "right/" -skip_parang --output_file 'flux.fits'

# #####################
# #Register the cubes if not done before
# #####################
rm cube-info/left*.fits
rm cube-info/right*.fits
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern left_*SCIENCE -nofit -no_psf
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern right_*SCIENCE -nofit -no_psf

# #####################
# #frame selection (detection of bad frames and writing of file frame_selection.txt)
# #####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"frame_selection_auto.py" --pattern left*SCIENCE

# #####################
# #Sdi -> make a new cube with the subtraction -> have to change the rdb files
# #####################
mpirun $GRAPHIC_N_CORES python  $GRAPHIC_DIR"sdi_func.py" --pattern left*SCIENCE --info_dir cube-info --info_pattern all_info_*left*SCIENCE

# #####################
# #frame selection (removing the bad frames)
# #####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"frame_selection.py" --pattern sdi_*SCIENCE --info_dir cube-info --info_pattern all_info_20_5_300_
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"frame_selection.py" --pattern left_*SCIENCE --info_dir cube-info --info_pattern all_info_20_5_300_
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"frame_selection.py" --pattern right_*SCIENCE --info_dir cube-info --info_pattern all_info_20_5_300_

# #####################
# #Perform ADI subtraction
# #####################
# cd sdi
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_psf_sub_3.3.py" -sphere -centred --pattern frame_sel_sdi --tmax 1200 --n_fwhm 0.3 --rmin 30 --info_dir cube-info --info_pattern all_info_20_5_300_frame_sel_sdi --fmax 200 --fwhm 3.5 -nofit # --debug 2
# cd ..
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_psf_sub_3.3.py" -sphere -centred --pattern left_cl --tmax 1200 --n_fwhm 0.3 --rmin 30 --info_dir cube-info --info_pattern all_info_20_5_300_left --fmax 200 --fwhm 3.5 -nofit # --debug 2
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_psf_sub_3.3.py" -sphere -centred --pattern right_cl --tmax 1200 --n_fwhm 0.3 --rmin 30 --info_dir cube-info --info_pattern all_info_20_5_300_right --fmax 200 --fwhm 3.5 -nofit # --debug 2


# #####################
# # Derotate and median combine
# #####################
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_deromed_3.3.py" -sphere -derotate_not_on_first_frame -nofit --pa_offset -1.84 --steps 10 --pattern ps_3.3*frame_sel_sdi --info_pattern all_info_ps_3.3*frame_sel_sdi
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_deromed_3.3.py" -sphere -derotate_not_on_first_frame -nofit --pa_offset -1.84 --steps 10 --pattern ps_3.3*left --info_pattern all_info_ps_3.3*left
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_deromed_3.3.py" -sphere -derotate_not_on_first_frame -nofit --pa_offset -1.84 --steps 10 --pattern ps_3.3*right --info_pattern all_info_ps_3.3*right

# #####################
# # Convert for pca and run PCA algo
# #####################
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_convert_for_pca.py" --pattern frame_sel_sdi_*SCIENCE --output_dir sdi
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_convert_for_pca.py" --pattern frame_sel_left_*SCIENCE --output_dir left
mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_convert_for_pca.py" --pattern frame_sel_right_*SCIENCE --output_dir right
# # # For a 3.5pix fwhm, 2 fwhm width annulus, with r_min = 7, r_max = 217, n_annuli =30
# # # For a 4.5pix fwhm, 2 fwhm width annulus, with r_min = 7, r_max = 196, n_annuli = 21
# # # For a 4.5pix fwhm, 2 fwhm width annulus, and 198pix radius, use 22 annuli
cd sdi
	python $GRAPHIC_DIR"GRAPHIC_pca.py" --output_dir "./GRAPHIC_PCA/" --threads 6 --arc_length 1e4 --n_annuli 30 -median_combine --n_fwhm 0.75 --n_modes 12 --min_reference_frames 15 --fwhm 3.5 --r_min 7. --r_max 217. -save_derot_cube
cd ..
cd left
	python $GRAPHIC_DIR"GRAPHIC_pca.py" --output_dir "./GRAPHIC_PCA/" --threads 6 --arc_length 1e4 --n_annuli 30 -median_combine --n_fwhm 0.75 --n_modes 12 --min_reference_frames 15 --fwhm 3.5 --r_min 7. --r_max 217. -save_derot_cube
cd ..
cd right
	python $GRAPHIC_DIR"GRAPHIC_pca.py" --output_dir "./GRAPHIC_PCA/" --threads 6 --arc_length 1e4 --n_annuli 30 -median_combine --n_fwhm 0.75 --n_modes 12 --min_reference_frames 15 --fwhm 3.5 --r_min 7. --r_max 217. -save_derot_cube
cd ..

# #####################
# # PCA contrast curve code
# #####################
# Now run the detection limits and self-subtraction code
# # # # plate scale (from Maire et al 2016):
# # # H2: 0.012255, H3: 0.012250, K1: 0.012267, K2: 0.012263
cd sdi
python $GRAPHIC_DIR"GRAPHIC_contrast_pca_3.3.py" --output_dir "./GRAPHIC_PCA/" --threads 6 --plate_scale 0.012255 --median_filter_length 20 --smooth_image_length 1.25 --r_max 217. --n_throughput 10
cd ..
cd left
python $GRAPHIC_DIR"GRAPHIC_contrast_pca_3.3.py" --output_dir "./GRAPHIC_PCA/" --threads 6 --plate_scale 0.012255 --median_filter_length 20 --smooth_image_length 1.25 --r_max 217. --n_throughput 10
cd ..
cd right
python $GRAPHIC_DIR"GRAPHIC_contrast_pca_3.3.py" --output_dir "./GRAPHIC_PCA/" --threads 6 --plate_scale 0.012250 --median_filter_length 20 --smooth_image_length 1.25 --r_max 217. --n_throughput 10
cd ..

# #####################
# # Preparation for injection of fake companions (psf combine and computation of primary flux
# ####################
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"SPHERE_characterization/SPHERE_prep_for_FP_injection.py"

# #####################
# # injection of fake companions
# ####################
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_inject_fake_3.3.py" -sphere --pattern left_cl*SCIENCE --info_dir cube-info --info_pattern all_info_20_5_300_left -nofit --deltamag 10 12 14 16 --sepvect 0.3 0.5 0.7 1. 1.5 2. 3.0 --wavelen 1.5888
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_register_3.3.py" -sphere --pattern FP*left -nofit -no_psf
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_psf_sub_3.3.py" -sphere -centred --pattern FP*left --tmax 1200 --n_fwhm 0.3 --rmin 30 --info_dir cube-info --info_pattern all_info_*FP_*left --fmax 200 --fwhm 3.5 -nofit # --debug 2
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"GRAPHIC_deromed_3.3.py" -sphere -derotate_not_on_first_frame -nofit --pa_offset -1.84 --steps 10 --pattern ps_3.3*FP*left --info_pattern all_info_ps_3.3*FP*left

# #####################
# # SPHERE Characterization
# ####################
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"SPHERE_characterization/contrast_curve_SPHERE/Companion_detection_and_characterization.py" --pattern_image nomask*sdi --pattern_psf psf_left --pattern_image_FP nomask*FP*left --pixel_scale 0.01227 --sep_FP 0.3 0.5 0.7 1. 1.5 2. 3.0 --dmag_given 10 --fwhm 4
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"SPHERE_characterization/contrast_curve_SPHERE/Contrast_curves.py" --pattern_image nomask*sdi --pattern_psf psf_left --age 5.000 --pixel_scale 0.01227 

# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"SPHERE_characterization/contrast_curve_SPHERE/Companion_detection_and_characterization.py" --pattern_image nomask*left --pattern_psf psf_left --pattern_image_FP nomask*FP*left --pixel_scale 0.01227 --sep_FP 0.3 0.5 0.7 1. 1.5 2. 3.0 --dmag_given 10 --fwhm 4 -no_comp_detection
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"SPHERE_characterization/contrast_curve_SPHERE/Contrast_curves.py" --pattern_image nomask*left --pattern_psf psf_left --age 5.000 --pixel_scale 0.01227 

# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"SPHERE_characterization/contrast_curve_SPHERE/Companion_detection_and_characterization.py" --pattern_image nomask*right --pattern_psf psf_right --pattern_image_FP nomask*FP*left --pixel_scale 0.01227 --sep_FP 0.3 0.5 0.7 1. 1.5 2. 3.0 --dmag_given 10 --fwhm 4 -no_comp_detection
# mpirun $GRAPHIC_N_CORES python $GRAPHIC_DIR"SPHERE_characterization/contrast_curve_SPHERE/Contrast_curves.py" --pattern_image nomask*right --pattern_psf psf_right --age 5.000 --pixel_scale 0.01227
