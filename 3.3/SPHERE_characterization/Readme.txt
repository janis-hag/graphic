Instruction to use characterization for SPHERE:

1) First put your final files in a directory with:
	- final image
	- flux_left files
	- flux_right files

2) Use SPHERE_psf_combine.py to produce a median image file of your psf files
Don’t forget to change the path (and the pattern of the left and right flux files if needed)

3) Use signal_to_noise.py to compute the S/N of a potential companion that you may have detected (optional for next parts).
Don’t forget to change the path and filename of your final image
Don’t forget to change the Radius_to_hide (line 10) in order to hide the central part of the image to see better the companion.
Values are not registered (it’s just an information)

4) Use primary_flux.py to compute the primary flux for the input of the GRAPHIC_inject_fake_3.3.py routine
Don’t forget to change wdir
Don’t forget to change image_filename
Don’t forget to change flux_filename
	the routine gives the primary_flux that can be used for the injection of fake planet in order to compute the adi self_subtraction

4) Use companion_extraction.py in the photometry_SPHERE directory
Don’t forget to change path
Don’t forget to change date (date of the observation in case of multiple observation for 1 target)
Don’t forget to change path_psf
Don’t forget to change image_filename
Don’t forget to change target_name
Don’t forget to change filter_name
Don’t forget to change psf_filename
Don’t forget to change pix_scale (line 41)
Don’t forget to change adi_self_subtraction (line 38) (to do this you should use the GRAPHIC_inject_fake_3.3.py to add a planet at the good separation and use companion_extraction.py a first time with the switch find_adi_subtraction to True and the dmag_given with the dmag you implemented in the GRAPHIC_inject_fake_3.3.py routine)
Don’t forget to change xmag, ymag, exmag, eymag, exrotation, eyrotation (line 42-49) (rotation and yrotation should be corrected with the key word pa_offset in the routine GRAPHIC_deromed_3.3.py on the GRAPHIC pipeline)
	this routine gives in return:
	- one file "companion_photo_astro_"+target_name+".rdb" in the path directory with the result of astrometry and photometry fit of the companion.
	- one file "../contrast_curve_SPHERE/companion_extraction.txt" to be used for contrast curve later.

5) Use graphic_contrast_adi.py in the contrast_curve_SPHERE directory to produce an adi contrast curve.
Don’t forget to change wdir with the directory of your data results (images)
Don’t forget to change image_filename
Don’t forget to change flux_filename (with the filename produced at stage 2))
Don’t forget to change name that will be used for the files at the end
Don’t forget to change the SETTINGS (line 43-56)
Don’t forget to change the SWITCHES (line 59-64)
Don’t forget to change the plate_scale if needed (line 68) you can find it at http://wiki.oamp.fr/sphere/AstrometricCalibration
	the routine gives in return:
	- a directory Contrast_curve_resulting_files in the wdir directory
	- plots of the contrast curves in this directory (contrast and mag, if the SWITCHES is turned on)
	- a file contrast_curve.rdb in this directory which will be used at the next steps

6) Use contrast_curve_sdi.py in the contrast_curve_SPHERE directory (after running the graphic_contrast_adi.py) to produce a adi+sdi detection limit in mass
Don’t forget to change the target
Don’t forget to change wdir
Don’t forget to change path_psf (if different than wdir)
Don’t forget to change psf_left_filename and psf_right_filename
Don’t forget to change path_results
Don’t forget to change SETTINGS (line 24-32)
Don’t forget to change Switches (line 35)