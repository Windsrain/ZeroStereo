import re
import os
import cv2
import json
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

def tartanair_disp_reader(filename):
    depth = np.load(filename)
    disp = 80.0 / depth
    valid = disp > 0

    return disp, valid

def crestereo_disp_reader(filename):
    disp = np.array(Image.open(filename))
    valid = disp > 0.0

    return disp.astype(np.float32) / 32., valid

def fallingthings_disp_reader(filename):
    a = np.array(Image.open(filename))
    with open('/'.join(filename.split('/')[:-1] + ['_camera_settings.json']), 'r') as f:
        intrinsics = json.load(f)
    fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    disp = (fx * 6.0 * 100) / a.astype(np.float32)
    valid = disp > 0

    return disp, valid

def vkitti2_disp_reader(filename):
    depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = (depth / 100).astype(np.float32)
    valid = (depth > 0) & (depth < 655)
    
    focal_length = 725.0087
    baseline = 0.532725
    disp = baseline * focal_length / depth
    disp[~valid] = 0.0

    return disp, valid

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