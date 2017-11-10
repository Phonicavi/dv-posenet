import cv2
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np
import math
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
from visualize import generate_arrow


# Following graphic model is based on Human3.6M
def get_poseKinematic():
    torso = np.array([(0,7),(7,8),(0,1),(0,4)])
    head  = np.array([(8,9),(9,10)])
    r_leg = np.array([(1,2),(2,3)])
    l_leg = np.array([(4,5),(5,6)])
    r_arm = np.array([(8,14),(14,15),(15,16)])
    l_arm = np.array([(8,11),(11,12),(12,13)])
    kinematic = np.vstack((torso, head, r_leg, l_leg, r_arm, l_arm))
    return kinematic


def get_poseKinematic16():
    r_leg = np.array([(0,1),(1,2),(2,6)])
    l_leg = np.array([(6,3),(3,4),(4,5)])
    torso = np.array([(6,7),(7,8)])
    head  = np.array([(8,9)])
    r_arm = np.array([(8,12),(12,11),(11,10)])
    l_arm = np.array([(8,13),(13,14),(14,15)])
    kinematic = np.vstack((r_leg, l_leg, torso, head, r_arm, l_arm))
    return kinematic


def get_poseConnect():
    connect = np.array([[1,4,7],[0,2],[1,3],[2],[0,5],[4,6],[5],[0,8],
                        [7,9,11,14],[8,10],[9],[8,12],[11,13],[12],
                        [8,15],[14,16],[15]])
    return connect


def mpii_id():
    convert = np.array([3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 13, 12, 11, 14, 15, 16])
    return convert


def draw_gt_skel2D(skel2, image, color, thick):
    kinematic = get_poseKinematic()
    for limb in kinematic:
        x0, y0 = skel2[:,limb[0]]
        x1, y1 = skel2[:,limb[1]]
        cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), color, thick)
    return image


def draw_skel2D_16(skel2, image, color, thick):
    kinematic = get_poseKinematic16()
    for limb in kinematic:
        x0, y0 = skel2[limb[0],:]
        x1, y1 = skel2[limb[1],:]
        cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), color, thick)
    return image


def draw_vec_skel2D_16(skel2, rankmat, image, thick):
    kinematic = get_poseKinematic16()
    for limb in kinematic:
        i = limb[0]
        j = limb[1]
        x0, y0 = skel2[i,:]
        x1, y1 = skel2[j,:]
        vert_i = (int(x0), int(y0))
        vert_j = (int(x1), int(y1))
        prob_i = rankmat[i][j]
        prob_j = rankmat[j][i]
        # decide pStart & pEnd & color
        pStart, pEnd, color = generate_arrow(i, j, vert_i, vert_j, prob_i, prob_j)
        cv2.arrowedLine(image, pStart, pEnd, color, thick)
    return image
"""
def draw_skel3D(skel3, ax):
    kinematic = get_poseKinematic()
    for limb in kinematic:
        x0, y0, z0 = skel3[:, limb[0]]
        x1, y1, z1 = skel3[:, limb[1]]
        line = axes3d.art3d.Line3D((x0,x1),(y0,y1),(z0,z1))
        ax.add_line(line)
    return ax

def draw_bones(bones, ax):
    for bone in bones.transpose():
        line = axes3d.art3d.Line3D((0,bone[0]),(0,bone[1]),(0,bone[2]))
        ax.add_line(line)
    return ax

"""
def rotate_skel3D(skel3d, roll, pitch, yaw):
    angle = np.array([roll, pitch, yaw]) # for x, y, z-axis
    angle *= np.pi / 180.
    rotate_x = np.array([[1, 0, 0],
                         [0, np.cos(angle[0]), -np.sin(angle[0])],
                         [0, np.sin(angle[0]), np.cos(angle[0])]])
    rotate_y = np.array([[ np.cos(angle[1]), 0, np.sin(angle[1])],
                         [0, 1, 0],
                         [-np.sin(angle[1]), 0, np.cos(angle[1])]])
    rotate_z = np.array([[np.cos(angle[2]), -np.sin(angle[2]), 0],
                         [np.sin(angle[2]),  np.cos(angle[2]), 0],
                         [0, 0, 1]])
    rotation = np.dot(np.dot(rotate_x, rotate_y), rotate_z)
    for i in range(0, 17):
        k = skel3d[:, i].transpose()
        skel3d[:,i] = np.dot(rotation, k)
    return skel3d


def skel2bones(skel3d):
    bones = np.zeros((3,16))
    kinematic = get_poseKinematic()
    for i in range(0, 16):
        parent = skel3d[:, kinematic[i,0]]
        joint = skel3d[:, kinematic[i,1]]
        bones[:, i] = parent - joint
    return bones


def bones2skel(bones):
    skel3d = np.zeros((3,17))
    kinematic = get_poseKinematic()
    for i in range(0, 16):
        skel3d[:, kinematic[i, 1]] = skel3d[:, kinematic[i,0]] - bones[:, i]
    return skel3d


def path_on_skel(s, t):
    kinematic = get_poseKinematic()
    connect = np.zeros((17,17), dtype=bool)
    # # use connect matrix
    # for bone in kinematic:
    #     connect[bone[0], bone[1]] = True
    #     connect[bone[1], bone[0]] = True
    previous = -np.ones(17, dtype=np.int)
    queue = -np.ones(17, dtype=np.int)
    pointer = 0
    queue_size = 1
    previous[t] = -2
    queue[0] = t
    flag = False
    conn_table = get_poseConnect()
    while True:
        current = queue[pointer]
        for i in conn_table[current]:
            if (previous[i]==-1):
                previous[i] = current
                queue[queue_size] = i
                queue_size += 1
            if (i==s):
                flag=True
                break
        if flag:
            break
        pointer += 1
    size = int(1)
    path = np.zeros(17, dtype=np.int)
    path[0] = s
    while (path[size-1] != t):
        path[size] = previous[path[size-1]]
        size += 1
    return path[:size]


# Following functions are design for c2f annotations
def draw_skel3D(pose3d):
    lcolor = "#3498db"
    rcolor = "#e74c3c"
    kinematic = get_poseKinematic()
    I = np.array(kinematic[:,0])
    J = np.array(kinematic[:,1])
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    lines = []
    for i in np.arange( len(I) ):
        x, y, z = [np.array((pose3d[I[i],j],pose3d[J[i],j])) for j in range(3)]
        lines.append(ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)[0])
    for i in range(17):
        ax.text(pose3d[i, 0], pose3d[i, 1], pose3d[i, 2], i)


    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = pose3d[0,0], pose3d[0,1], pose3d[0,2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=-90, elev=-64)
    # Get rid of the ticks and tick labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])
    # ax.set_zticklabels([])
    # ax.set_aspect('equal')
    plt.show()
    return


# Following functions are design for c2f annotations
def draw_skel3D_16(pose3d):
    lcolor = "#3498db"
    rcolor = "#e74c3c"
    kinematic = get_poseKinematic16()
    I = np.array(kinematic[:,0])
    J = np.array(kinematic[:,1])
    LR = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    lines = []
    for i in np.arange( len(I) ):
        x, y, z = [np.array((pose3d[I[i],j],pose3d[J[i],j])) for j in range(3)]
        lines.append(ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)[0])
    for i in range(16):
        ax.text(pose3d[i, 0], pose3d[i, 1], pose3d[i, 2], i)


    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = pose3d[6,0], pose3d[6,1], pose3d[6,2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=-89, elev=-62)
    # Get rid of the ticks and tick labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])
    # ax.set_zticklabels([])
    # ax.set_aspect('equal')
    plt.show()
    return


# Following functions are design for 16 point mpii annotations
def get_limbLength(pose3d):
    kinematic = get_poseKinematic16()
    length = []
    for i in range(kinematic.shape[0]):
        s,t = kinematic[i]
        X = (pose3d[s][0] - pose3d[t][0]) ** 2
        Y = (pose3d[s][1] - pose3d[t][1]) ** 2
        Z = (pose3d[s][2] - pose3d[t][2]) ** 2
        length.append(math.sqrt(X + Y + Z))
        #print(s,t,math.sqrt(X + Y + Z))
    return length
