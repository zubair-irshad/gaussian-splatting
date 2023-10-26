import os

# Replace with the path to your .house file
house_file_path = "/home/zubairirshad/Downloads/house_segmentations/17DRP5sb8fy/house_segmentations/17DRP5sb8fy.house"

# Replace with the path to your image directory
image_dir = "/home/zubairirshad/Downloads/mp3d_single_scene/undistorted_color_images/17DRP5sb8fy/undistorted_color_images"

# Create a dictionary to store the mapping of regions to image files
region_to_images = {}

# Parse the .house file to extract region information
with open(house_file_path, "r") as house_file:
    lines = house_file.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            if parts[0] == "R":
                region_index = int(parts[1])
                region_name = parts[4]
                region_to_images[region_index] = region_name

# Create subfolders for each region
for region_index, region_name in region_to_images.items():
    region_folder = os.path.join(image_dir, region_name)
    os.makedirs(region_folder, exist_ok=True)

# List all image files in the image directory
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Copy images to their respective subfolders
for image_file in image_files:
    image_name = image_file.split(".")[0]
    image_region_index = int(image_name.split("_")[0])
    if image_region_index in region_to_images:
        region_name = region_to_images[image_region_index]
        destination_folder = os.path.join(image_dir, region_name)
        image_path = os.path.join(image_dir, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        os.rename(image_path, destination_path)

print("Images copied to their respective subfolders.")
