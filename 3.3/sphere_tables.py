"""
Janis Hagelberg <janis.hagelberg@unige.ch>

This program is part of GRAPHIC, the
Geneva Reduction and Analysis Pipeline for High-contrast Imaging of Companions.

This step is to perform SDI subtraction on each single frame.

If you find any bugs or have any suggestions email: janis.hagelberg@unige.ch
"""


def sdi_wavelength(hdr):
    #HIERARCH ESO INS1 OPTI2 ID
    if not 'ESO INS1 OPTI2 ID' in hdr.keys():
        print('No DBI keyword found')
        return None
    elif hdr['ESO INS1 OPTI2 ID'] == 'FILT_DBF_Y23':
        lambdas = {'left': 1025.6, 'right': 1080.2}
    elif hdr['ESO INS1 OPTI2 ID'] == 'FILT_DBF_J23':
        lambdas = {'left': 1189.5, 'right': 1269.8}
    elif hdr['ESO INS1 OPTI2 ID'] == 'FILT_DBF_H23':
        lambdas = {'left': 1588.8, 'right': 1667.1}
    elif hdr['ESO INS1 OPTI2 ID'] == 'FILT_DBF_H32':
        lambdas = {'left': 1589.0, 'right': 1665.3}
    elif hdr['ESO INS1 OPTI2 ID'] == 'FILT_DBF_H34':
        lambdas = {'left': 1665.0, 'right': 1736.4}
    elif hdr['ESO INS1 OPTI2 ID'] == 'FILT_DBF_K12':
        lambdas = {'left': 2102.5, 'right': 2255.0}
    else:
        print('Unknown filter:' + str(hdr['ESO INS1 OPTI2 ID']))
        return None
    return lambdas
