import numpy as np
import scipy.io as scio

## Following functions are used to read annot.mat

def read(annot_file):
    mat = scio.loadmat(annot_file)
    rawdata = mat['annot']
    return rawdata[0]

def get_subject(rawdata):
    subject = rawdata['subject'][0]
    return subject[0,0]

def get_action(rawdata):
    action = rawdata['action'][0]
    return action[0]

def get_subaction(rawdata):
    subaction = rawdata['subaction'][0]
    return subaction[0,0]

def get_camera(rawdata):
    camera = rawdata['camera'][0]
    return camera[0,0]

def get_path(rawdata):
    path = rawdata['path'][0]
    return path[0]

def get_frames(rawdata):
    data = rawdata['skeleton'][0]
    return data[0,:]

def get_frameIndex(frame):
    return frame['frame'][0,0]

def get_skel2D(frame):
    return frame['skel2D']

def get_skel3D(frame):
    return frame['skel3D']

def nomalize_skel3D(skel3d):
    for i in range(0,3):
        skel3d[i,:] -= skel3d[i,0]
    root = skel3d[:, 0]
    spline = skel3d[:, 7]
    scale = np.linalg.norm(spline-root)
    skel3d /= scale
    return skel3d

def nomalize_skel2D(skel2d):
    for i in range(0,2):
        skel2d[i,:] -= skel2d[i,0]
    root = skel2d[:, 0]
    spline = skel2d[:, 7]
    scale = np.linalg.norm(spline-root)
    skel2d /= scale
    return skel2d

#Following functions are used to get information from h5 file

def h5get_filename(id,h5file):
    str_arr = h5file.get('imgname')[id]
    str_arr = str_arr[np.nonzero(str_arr)].astype(int)
    return ''.join(map(chr, str_arr))

def h5get_3Dto2D(id, h5file):
    f = h5file.get('f')[id]
    c = h5file.get('c')[id]
    return np.array([[f[0], 0, c[0]],
                     [0, f[1], c[1]],
                     [0, 0, 1]])

def h5get_3d(id, h5file):
    skel3d = h5file.get('part_3Dmono')[id]
    return skel3d.transpose()

def h5get_o3d(id, h5file):
    skel3d = h5file.get('part_3D')[id]
    return skel3d.transpose()

def h5get_2d(id, h5file):
    skel2d = h5file.get('part_2D')[id]
    return skel2d.transpose()

# Following functions are used to get useful information

def get_boundingbox2D(coords):
    #x.max is max column, y.max is max row
    x = coords[0,:]
    y = coords[1,:]
    boundingbox = np.array([[y.min(),y.max()],[x.min(),x.max()]])
    return boundingbox

def enlarge_bbox(bbox, r_scale, c_scale):
    row_mean = (bbox[0,0]+bbox[0,1])/2.
    col_mean = (bbox[1,0]+bbox[1,1])/2.
    r0 = (bbox[0, 0]-row_mean)*r_scale + row_mean
    c0 = (bbox[1, 0]-col_mean)*c_scale + col_mean
    r1 = (bbox[0, 1]-row_mean)*r_scale + row_mean
    c1 = (bbox[1, 1]-col_mean)*c_scale + col_mean

    bbox_poly = np.array([(r0, c0),
                          (r1, c0),
                          (r1, c1),
                          (r0, c1)])
    return bbox_poly

def get_square_bbox(image, coords, scale):
    bbox = get_boundingbox2D(coords)
    row_mean = (bbox[0,0]+bbox[0,1])/2.
    col_mean = (bbox[1,0]+bbox[1,1])/2.
    half_edge = np.max([row_mean-bbox[0,0], col_mean-bbox[1,0]])
    if (2 * scale * half_edge > np.min(image.shape[0:2])):
        half_edge = 0.5 * np.min(image.shape[0:2]) /scale
    crop_box = np.array([[int(np.ceil(row_mean-half_edge*scale)), int(np.floor(row_mean+half_edge*scale))],
                         [int(np.ceil(col_mean-half_edge*scale)), int(np.floor(col_mean+half_edge*scale))]])
    if (crop_box[0,0]<0) :
        crop_box[0,:] -= crop_box[0,0]
    if (crop_box[0,1]>=image.shape[0]) :
        crop_box[0,:] += image.shape[0]-crop_box[0,1]-1
    if (crop_box[1,0]<0) :
        crop_box[1,:] -= crop_box[1,0]
    if (crop_box[1,1]>=image.shape[1]) :
        crop_box[1,:] += image.shape[1]-crop_box[1,1]-1
    return crop_box

def get_affine_matrix(skel3d, skel3d_mono):
    a = np.hstack((skel3d.transpose()[0:4], np.ones((4,1))))
    b = np.hstack((skel3d_mono.transpose()[0:4], np.ones((4,1))))
    x = np.linalg.solve(a,b)
    return x.transpose()

