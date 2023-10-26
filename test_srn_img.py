
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
# img_name = '/home/zubairirshad/Downloads/srn_cars/cars_train/1a1dcd236a1e6133860800e6696b8284/rgb/000001.png'
img_name = '/home/zubairirshad/Downloads/srn_cars/cars_train/1b85c850cb4b93a6e9415adaaf77fdbf/rgb/000000.png'
#img_name = '/home/zubairirshad/Downloads/nerf_synthetic/nerf_synthetic/chair/train/r_1.png'

white_background = False
image = Image.open(img_name)
image = np.array(image)

# image = imageio.imread(img_name)[..., :3]
# image_mask = np.array(image)[:,:,:3]

image_mask = image[:,:,:3]
print("image_mask", image.max(), image.min())

#the below code is for pytorch we want to do the same in numpy
mask = (image_mask != 255).all(axis=-1)[..., None]
# mask = ~mask
mask = mask.astype(np.uint8) * 255
plt.imshow(mask)
plt.show()

print("mask", mask.max(), mask.min())

alpha = np.array(image)[:, :, 3:4]
plt.imshow(alpha)
plt.show()

print("alpha", alpha.max(), alpha.min())

#now let's replace alpha with mask
image = np.array(image)
image[:, :, 3:4] = mask
#concatentate rgb with mask (alpha)
# image = np.concatenate((image, mask), axis=2)
# size = image.size

# image = imageio.imread(img_name)

# print("image", image.shape)
# image = np.array(image)
# H,W = image.shape[:2]
# image = image.reshape(-1, 4) # (h*w, 4) RGBA 
# image = image.astype(np.float32)/255.0
# image = image[:, :3]*image[:, -1:] + (1-image[:, -1:]) # blend A to RGB
# # print("image", image.shape)
# image = image.reshape(H, W, 3)
# im_data = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB").convert("RGBA")

# print("image.size", size)
# background = Image.new('RGBA', size, (0, 0, 0))

# print("background shape", np.array(background).shape)
# print("im_data shape", np.array(image).shape)

# im_data = Image.alpha_composite(background, image)

plt.imshow(image)
plt.show()


im_data = np.array(image)


print("im_data.shape", im_data.shape)
print("image", im_data.max(), im_data.min())

bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

norm_data = im_data / 255.0

# print("norm_data[:,:,3:4]", np.max(norm_data[:,:,3:4]), np.min(norm_data[:,:,3:4]))
# print("norm_data", np.max(norm_data), np.min(norm_data))
arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
# arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

print("image", np.array(image).shape)
plt.imshow(np.array(image))
plt.show()