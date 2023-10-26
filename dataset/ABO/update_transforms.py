# import os
# import shutil
# import json

# # Define the source and destination directories
# source_dir = '/home/zubairirshad/Downloads/ABO_RELEASE/B00EUL2B16'
# destination_dir = os.path.join(source_dir, 'frames')

# # Create the destination directory if it doesn't exist
# os.makedirs(destination_dir, exist_ok=True)

# # Move all .png files to the frames folder
# for filename in os.listdir(source_dir):
#     if filename.endswith('.png'):
#         src_file = os.path.join(source_dir, filename)
#         dst_file = os.path.join(destination_dir, filename)
#         shutil.move(src_file, dst_file)

# # Update the file paths in transforms.json to be relative to the frames folder
# json_file_path = os.path.join(source_dir, 'transforms.json')
# with open(json_file_path, 'r') as json_file:
#     data = json.load(json_file)

# for frame in data['frames']:
#     frame['file_path'] = f'./frames/{os.path.basename(frame["file_path"])}'

# # Write the updated data back to transforms.json
# with open(json_file_path, 'w') as json_file:
#     json.dump(data, json_file, indent=4)

# print("PNG files moved to 'frames' folder and file paths updated in transforms.json.")


# import os
# import json

# # Define the source directory
# source_dir = '/home/zubairirshad/Downloads/ABO_RELEASE/B00EUL2B16'

# # Read the transforms.json file
# json_file_path = os.path.join(source_dir, 'transforms.json')

# with open(json_file_path, 'r') as json_file:
#     data = json.load(json_file)

# # Update the file paths in transforms.json to remove the .png extension
# for frame in data['frames']:
#     if 'file_path' in frame:
#         frame['file_path'] = os.path.splitext(frame['file_path'])[0]

# # Write the updated data back to transforms.json
# with open(json_file_path, 'w') as json_file:
#     json.dump(data, json_file, indent=4)

# print("File paths in transforms.json updated to remove .png extension.")

