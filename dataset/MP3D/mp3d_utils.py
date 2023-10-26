
from io import StringIO
from numpy.linalg import inv,norm
import numpy as np
from plyfile import PlyData, PlyElement

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

def load_region_boundaries(house_file, region_index):
    boundaries = {}
    print("region_index", region_index)
    with open(house_file, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] == "R":
                print("parts", parts)

                print("parts[1]", parts[1], region_index, int(parts[1]) == int(region_index))
            if parts and parts[0] == "R" and int(parts[1]) == int(region_index):
                xlo, ylo, zlo, xhi, yhi, zhi = map(float, parts[9:15])
                print("xlo, ylo, zlo, xhi, yhi, zhi", xlo, ylo, zlo, xhi, yhi, zhi)
                boundaries['xlo'] = xlo
                boundaries['xhi'] = xhi
                boundaries['ylo'] = ylo
                boundaries['yhi'] = yhi
                boundaries['zlo'] = zlo
                boundaries['zhi'] = zhi
                return boundaries
    return None

def filter_poses_by_region(pose_dict, region_boundaries):
    xlo, xhi, ylo, yhi, zlo, zhi = (region_boundaries['xlo'], region_boundaries['xhi'],
                                   region_boundaries['ylo'], region_boundaries['yhi'],
                                   region_boundaries['zlo'], region_boundaries['zhi'])

    filtered_poses = {}
    for pose_id, pose_matrix in pose_dict.items():
        # Extract the translation part (fourth column) of the pose matrix
        translation = pose_matrix[0][:3, 3]

        # Check if the translation falls within the region's boundaries
        if (xlo <= translation[0] <= xhi) and (ylo <= translation[1] <= yhi) and (zlo <= translation[2] <= zhi):
            filtered_poses[pose_id] = pose_matrix[0].tolist()

    return filtered_poses

def fetchPlyForRegion(ply_file, house_file, region_index, mask=False):
    region_boundaries = load_region_boundaries(house_file, region_index)

    if not region_boundaries:
        print(f"Region {region_index} not found in the house file.")
        return None, None, None

    plydata = PlyData.read(ply_file)
    vertices = plydata["vertex"]
    print("plydata", plydata)

    xlo, xhi, ylo, yhi, zlo, zhi = (region_boundaries['xlo'], region_boundaries['xhi'],
                                   region_boundaries['ylo'], region_boundaries['yhi'],
                                   region_boundaries['zlo'], region_boundaries['zhi'])

    # Filter vertices for the specific region
    region_mask = (vertices["x"] >= xlo) & (vertices["x"] <= xhi) & \
                  (vertices["y"] >= ylo) & (vertices["y"] <= yhi) & \
                  (vertices["z"] >= zlo) & (vertices["z"] <= zhi)
    region_vertices = vertices[region_mask]

    positions = np.vstack([region_vertices["x"], region_vertices["y"], region_vertices["z"]]).T
    if mask:
        lb, ub = [-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]
        positions = positions[
            np.prod(np.logical_and((positions > lb), (positions < ub)), axis=-1)
        ]

    # print("positions min max", positions.min(), positions.max())
    # print("vertices keys", region_vertices.data.dtype.names)
    colors = np.vstack([region_vertices["red"], region_vertices["green"], region_vertices["blue"]]).T / 255.0
    normals = np.vstack([region_vertices["nx"], region_vertices["ny"], region_vertices["nz"]]).T

    #save the ply file in open3d format

    return positions, colors, normals


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