#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import glob

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

from io import StringIO
from numpy.linalg import inv,norm
def camera_parameters(camera_template):
  ''' Returns two dicts containing undistorted camera intrinsics (3x3) and extrinsics (4x4),
      respectively, for a given scan. Viewpoint IDs are used as dict keys. '''
  intrinsics = {}
  extrinsics = {}
  with open(camera_template) as f:
    pos = -1
    for line in f.readlines():
      if 'intrinsics_matrix' in line:
        intr = line.split()
        C = np.zeros((3, 3), np.double)
        C[0,0] = intr[1] # fx
        C[1,1] = intr[5] # fy
        C[0,2] = intr[3] # cx
        C[1,2] = intr[6] # cy
        C[2,2] = 1.0
        pos = 0
      elif pos >= 0 and pos < 6:
        q = line.find('.jpg')
        camera = line[q-37:q]
        if pos == 0:
          intrinsics[camera[:-2]] = C
        T = np.loadtxt(StringIO(line.split('jpg ')[1])).reshape((4,4))
        # T is camera-to-world transform, invert for world-to-camera
        extrinsics[camera] = (T,inv(T))
        pos += 1
  return intrinsics,extrinsics


def readCamerasMP3D(path, transformsfile, dataset_type= "mp3d"):
    

    cam_infos = []

    intrinsics, extrinsics = camera_parameters(os.path.join(path, "undistorted_camera_parameters/17DRP5sb8fy/undistorted_camera_parameters", transformsfile))


    for idx, frame in enumerate(extrinsics.keys()):
        c2w = extrinsics[frame][0]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, "undistorted_color_images/17DRP5sb8fy/undistorted_color_images", frame+'.jpg')

        intrinic = intrinsics[list(intrinsics.keys())[0]]
        print("intrinic", intrinic)
        focal = intrinic[0, 0]
        
        image_name = Path(image_path).stem
        image = Image.open(image_path)

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", dataset_type= "blender"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        if dataset_type == "blender":
            fovx = contents["camera_angle_x"]
        # else:
        #     fovx = contents["camera"]["angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            if dataset_type == "ABO":
                fovx = frame["camera"]["angle_x"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        focal = float(lines[0].split()[0])
        H, W = lines[-1].split()
        H, W = int(H), int(W)
    return focal, H, W

def make_poses(base_dir):
    intrinsic_path = os.path.join(base_dir, 'intrinsics.txt')
    focal, H, W = load_intrinsic(intrinsic_path)

    pose_dir = os.path.join(base_dir, 'pose')
    txtfiles = np.sort([os.path.join(pose_dir, f.name) for f in os.scandir(pose_dir)])
    posefiles = np.array(txtfiles)
    srn_coords_trans = np.diag(np.array([1, -1, -1, 1])) # SRN dataset
    poses = []
    all_c2w = []
    for posefile in posefiles:
        pose = np.loadtxt(posefile).reshape(4,4)
        all_c2w.append(pose@srn_coords_trans)
    return all_c2w, focal, H, W

def readCamerasSRN(path, white_background, extension=".png"):
    cam_infos = []

    all_c2w, focal, H, W = make_poses(path)

    # with open(os.path.join(path, transformsfile)) as json_file:
    #     contents = json.load(json_file)
    #     fovx = contents["camera_angle_x"]

    # hfov = np.rad2deg(np.arctan(W / 2. / focal) * 2.)
    # vfov = np.rad2deg(np.arctan(H / 2. / focal) * 2.)

    # frames = contents["frames"]

    frames = sorted(glob.glob(path+'/rgb/*.png'))
    for idx, img_name in enumerate(frames):
        # cam_name = os.path.join(path, frame["file_path"] + extension)

        # # NeRF 'transform_matrix' is a camera-to-world transform
        # c2w = np.array(frame["transform_matrix"])

        c2w = all_c2w[idx]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # image_path = os.path.join(path, cam_name)
        # image_name = Path(cam_name).stem
        # image = Image.open(image_path)

        image = Image.open(img_name)
        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        # fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        fovx = focal2fov(focal, image.size[0])
        fovy = focal2fov(focal, image.size[1])

        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", dataset_type = "blender"):
    
    if dataset_type == "blender":       
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, dataset_type)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, dataset_type)
        
        if not eval:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []
    elif dataset_type == "ABO":
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension, dataset_type)
        # print("Reading Test Transforms")
        # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
        
        # if not eval:
        #     train_cam_infos.extend(test_cam_infos)
        #     test_cam_infos = []

        test_cam_infos = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readMP3DInfo(path, white_background, eval, extension=".png", dataset_type = "blender"):
    
    # if dataset_type == "blender":       
    #     print("Reading Training Transforms")
    #     train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, dataset_type)
    #     print("Reading Test Transforms")
    #     test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, dataset_type)
        
    #     if not eval:
    #         train_cam_infos.extend(test_cam_infos)
    #         test_cam_infos = []
    # elif dataset_type == "ABO":
    #     print("Reading Training Transforms")
    #     train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension, dataset_type)
    #     # print("Reading Test Transforms")
    #     # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
        
    #     # if not eval:
    #     #     train_cam_infos.extend(test_cam_infos)
    #     #     test_cam_infos = []

    #     test_cam_infos = []

    #folder = '/home/zubairirshad/Downloads/mp3d'

    train_cam_infos = readCamerasMP3D(path, "17DRP5sb8fy.conf")
    test_cam_infos = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    print("nerf_normalization", nerf_normalization["radius"], nerf_normalization["translate"])

    ply_path = os.path.join(path, "poisson_meshes/17DRP5sb8fy/poisson_meshes", '17DRP5sb8fy_10.ply')
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")
        
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readSRNSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "SRN": readSRNSyntheticInfo,
    "ABO": readNerfSyntheticInfo,
    "MP3D": readMP3DInfo
}