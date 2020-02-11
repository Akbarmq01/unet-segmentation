import time
import numpy as np 
import torch
from torchvision import transforms
import cv2
import os
from model.Unet_model import UNET 
import matplotlib.pyplot as plt 


data_dir = '/home/akbar/data/unet_dataset/data/'
imglist = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

image = cv2.imread(imglist[np.random.randint(0, len(imglist))], 0)
image = cv2.resize(image, (512,512))

input_tensor = torch.tensor(image, dtype=torch.float).view(1,1,image.shape[0], image.shape[1])/255.0


with torch.set_grad_enabled(False):
	model = UNET(1, 1).cuda()
	model.load_state_dict(torch.load('./weights/u_net_wts.pth'))
	since = time.time()
	out_tnsr = model(input_tensor.cuda())
	print('total inference time : {0:4f}'.format((time.time())))

# convert output tensor to 2D ndarray
out = out_tnsr.view(512, 512).cpu().detach().numpy()

# final thresholded score map
out[np.where(out > 0.5)]  = 255
out[np.where(out <= 0.5)] = 0

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title("Predicted Score Map")
plt.imshow(out)

plt.savefig('./result/inference_test.png')
