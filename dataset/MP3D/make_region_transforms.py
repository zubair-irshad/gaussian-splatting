
import sys
sys.path.append('/home/zubairirshad/gaussian-splatting')
from dataset.MP3D.mp3d_utils import camera_parameters, load_region_boundaries, filter_poses_by_region, fetchPlyForRegion, storePly
from scene.gaussian_model import BasicPointCloud
import os
import numpy as np
import shutil
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

folder = '/home/zubairirshad/Downloads/mp3d_single_scene/undistorted_camera_parameters/17DRP5sb8fy/undistorted_camera_parameters/17DRP5sb8fy.conf'
ply_path = '/home/zubairirshad/Downloads/poisson_meshes/17DRP5sb8fy/poisson_meshes/17DRP5sb8fy_11.ply'
house_path = '/home/zubairirshad/Downloads/house_segmentations/17DRP5sb8fy/house_segmentations/17DRP5sb8fy.house'

intrinsics, extrinsics =  camera_parameters(folder)
region_boundaries = load_region_boundaries(house_file = house_path, region_index=0)
positions, colors, normals = fetchPlyForRegion(ply_file=ply_path, house_file=house_path, region_index=0, mask=False)

# basic_point_cloud = BasicPointCloud(points=positions, colors=colors, normals=normals)



#Now let's save the basic pointcloud to the region 0 folder under pointcloud folder

#Let's create a folder for the region 0
region_folder = '/home/zubairirshad/Downloads/mp3d_single_scene/regions/17DRP5sb8fy/region_0/pointcloud'
os.makedirs(region_folder, exist_ok=True)

#Now let's save the basic point cloud

ply_save_path = os.path.join(region_folder, 'pointcloud.ply')
storePly(ply_save_path, positions, colors * 255)
# np.save(, basic_point_cloud.points)


filtered_extrinsics = filter_poses_by_region(extrinsics, region_boundaries)

# Now let's save the extrinsics and corresponding images to the region 0 folder under poses and images folders, only save the images which are in the region 0

original_images_folder = '/home/zubairirshad/Downloads/mp3d_single_scene/undistorted_color_images/17DRP5sb8fy/undistorted_color_images'

#Let's create a folder for the region 0 images
region_images_folder = '/home/zubairirshad/Downloads/mp3d_single_scene/regions/17DRP5sb8fy/region_0/images'

os.makedirs(region_images_folder, exist_ok=True)

#Now let's copy the images found in filtered extrinsics keys to the region images folder
for key in filtered_extrinsics.keys():
    image_name = key + '.jpg'
    image_path = os.path.join(original_images_folder, image_name)
    shutil.copy(image_path, region_images_folder)


#Now let's save the extrinsics as transforms.json file with the same keys as the images

pose_folder = '/home/zubairirshad/Downloads/mp3d_single_scene/regions/17DRP5sb8fy/region_0/poses'
import json

os.makedirs(pose_folder, exist_ok=True)


#Also save focal as a key in the json file

intrinic = intrinsics[list(intrinsics.keys())[0]]
print("intrinic", intrinic)
focal = intrinic[0, 0]

#load one image and check its width and height in pillow
from PIL import Image

#pick one image randomly from the extrinsics keys
image_name = list(filtered_extrinsics.keys())[0] + '.jpg'

image = Image.open(os.path.join(region_images_folder, image_name))
width, height = image.size

# image = Image.open(os.path.join(region_images_folder, '000000.jpg'))

fovx = focal2fov(focal, width)


#save in this format

# {
#     "camera_angle_x": 0.6911112070083618,
#     "frames": [
#         {
#             "file_path": "./train/r_0",
#             "rotation": 0.012566370614359171,
#             "transform_matrix": [
#                 [
#                     -0.9938939213752747,
#                     -0.10829982906579971,
#                     0.021122142672538757,
#                     0.08514608442783356
#                 ],
#                 [
#                     0.11034037917852402,
#                     -0.9755136370658875,
#                     0.19025827944278717,
#                     0.7669557332992554
#                 ],
#                 [
#                     0.0,
#                     0.19142703711986542,
#                     0.9815067052841187,
#                     3.956580400466919
#                 ],
#                 [
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0
#                 ]
#             ]
#         },

#Let's save transforms.json in the above format

#Let's create a list of dictionaries for each image
frames = []

for key, value in filtered_extrinsics.items():
    image_name = key + '.jpg'
    image_path = os.path.join(region_images_folder, image_name)
    frames.append({"file_path": image_name, "rotation": fovx, "transform_matrix": value})

#Let's save the frames list in the json file
with open(os.path.join(pose_folder, 'transforms.json'), 'w') as fp:
    json.dump({"camera_angle_x": fovx, "frames": frames}, fp)

# with open(os.path.join(pose_folder, 'transforms.json'), 'w') as fp:
#     json.dump(filtered_extrinsics, fp)
    


