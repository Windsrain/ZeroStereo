import re
import os
import cv2
import numpy as np
from PIL import Image

def pfm_reader(filename):
    file = open(filename, 'rb')
    color, width, height, scale, endian = None, None, None, None, None
    header = file.readline().rstrip()

    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())

    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())

    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data

def kitti_disp_reader(filename, mask):
    disp = None
    if mask == 'all':
        disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    elif mask == 'noc':
        disp = cv2.imread(filename.replace('disp_occ', 'disp_noc'), cv2.IMREAD_ANYDEPTH) / 256.0
    else:
        raise Exception(f'Invalid mask name: {mask}.')

    valid = disp > 0

    return disp, valid

def middlebury_disp_reader(filename, mask):
    disp = pfm_reader(filename).astype(np.float32)
    nocc = np.array(Image.open(filename.replace('disp0GT.pfm', 'mask0nocc.png')))
    valid = None

    if mask == 'all':
        valid = nocc > 0
    elif mask == 'noc':
        valid = nocc == 255
    else:
        raise Exception(f'Invalid mask name: {mask}.')

    return disp, valid

def eth3d_disp_reader(filename, mask):
    disp = pfm_reader(filename).astype(np.float32)
    nocc = np.array(Image.open(filename.replace('disp0GT.pfm', 'mask0nocc.png')))
    valid = None

    if mask == 'all':
        valid = nocc > 0
    elif mask == 'noc':
        valid = nocc == 255
    else:
        raise Exception(f'Invalid mask name: {mask}.')

    return disp, valid