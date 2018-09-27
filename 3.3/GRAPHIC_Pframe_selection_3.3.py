#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

Its purpose is to create a list of frames that should be kept.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""
import numpy as np
import astropy.io.fits as pyfits
import sys
import graphic_nompi_lib_330 as graphic_nompi_lib
import argparse
import time
import pandas as pd

target_dir = "."
parser = argparse.ArgumentParser(
        description='Supress the frame in the cubes and rdb files from the \
        selection_frame file')
parser.add_argument('--pattern', action="store", dest="pattern", default="sdi",
                    help='cubes to apply the frame selections')
parser.add_argument('--info_pattern', action="store", dest="info_pattern",
                    required=True, help='Info filename pattern')
parser.add_argument('--info_dir', action="store", dest="info_dir",
                    default='cube-info', help='Info directory')
args = parser.parse_args()
pattern = args.pattern
info_pattern = args.info_pattern
info_dir = args.info_dir


def frame_selec(frame_sel_filename, pattern):
    f = open(frame_sel_filename,'r')
    lines = f.readlines()
    f.close()

    #path_cube_info="cube-info/"
    #all_info_pattern="all_info_20_5_300_"
    path_cube_info = info_dir+"/"
    all_info_pattern = info_pattern

    for line in lines:
        print(line.strip().split()[0])

        if line.strip().split()[0]!='filename' and line.strip().split()[0]!='--------':
            filename=line.strip().split()[0]
            frame_to_del=line.strip().split("\t")[1]
            print ('frame_to_del:',frame_to_del)
            frame_to_del=np.array(eval(frame_to_del))

            print(frame_to_del)

            if "sdi" in pattern:
                if "left" in filename:
                    filename = filename.replace("left","sdi")
                    all_info_pattern = all_info_pattern.split('sdi')[0]
                elif "right" in filename:
                    filename = filename.replace("right","sdi")
                    all_info_pattern = all_info_pattern.split('sdi')[0]
            elif "left" in pattern:
                if "sdi" in filename:
                    filename = filename.replace("sdi","left")
                    all_info_pattern = all_info_pattern.split('left')[0]
                elif "right" in filename:
                    filename = filename.replace("right","left")
                    all_info_pattern = all_info_pattern.split('left')[0]
            elif "right" in pattern:
                if "sdi" in filename:
                    filename = filename.replace("sdi","right")
                    all_info_pattern = all_info_pattern.split('right')[0]
                elif "left" in filename:
                    filename = filename.replace("left","right")
                    all_info_pattern = all_info_pattern.split('right')[0]

            print("filename", filename)
            cube, hdr = pyfits.getdata(filename, header=True)

            # Read all_info into a pandas DataFrame
            pd_table = pd.read_table(path_cube_info + all_info_pattern
                                     + filename.replace(".fits", ".rdb"),
                                     sep='\t', header=0, skiprows=[1],
                                     index_col=False)

            f = open(path_cube_info + all_info_pattern
                     + filename.replace(".fits",".rdb"))
            lines = f.readlines()
            f.close()

            index_keep=np.arange(np.shape(cube)[0])
#            counter1=0
#            counter2=0
            if np.size(frame_to_del)!=0:
                index_del=index_keep[frame_to_del]
                index_keep=np.delete(index_keep,frame_to_del)
            else:
                index_del=np.array([])
#                counter1=99999
#            table=[]
#            for line in lines:
#                if np.size(index_del)==0:
#                    table=np.append(table,line)
#                elif counter1!=index_del[counter2]+2:
#                    table=np.append(table,line)
#                elif counter2<np.size(index_del)-1:
#                    counter2+=1
#                else:
#                    counter1=99999
#                    counter1+=1
            # print 'Keeping:',index_keep

            cube = cube[index_keep, :, :]

            pd_table = pd_table.drop(index_del)

            hdr['NAXIS3'] = np.shape(cube)[0]

            pyfits.writeto("frame_sel_"+filename,cube, header=hdr, overwrite=True)
            filename3 = path_cube_info + all_info_pattern + "frame_sel_" + filename.replace(".fits",".rdb")

            graphic_nompi_lib.write_array2rdb(filename3, pd_table.values,
                                              pd_table.keys().values)
#            f = open(filename3,'w')
#            for i in range(np.size(table)):
#                f.write(table[i])
#            f.close()

frame_sel_filename="frame_selection.txt"
#print "ERROR: No files for frame selection detected!"

t0=time.time()
print("beginning of frame selection")
frame_selec(frame_sel_filename,pattern)


print("Total time: "+graphic_nompi_lib.humanize_time((time.time()-t0)))
print("frame selection finished")
sys.exit(0)
