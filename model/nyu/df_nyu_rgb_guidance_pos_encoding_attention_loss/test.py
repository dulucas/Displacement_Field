import cv2
import torch
import numpy as np
import pickle as pkl
from df import Displacement_Field

image = cv2.imread('/home/duy/phd/Displacement_Field/dataset/image.png', cv2.IMREAD_GRAYSCALE)
depth = pkl.load(open('/home/duy/phd/largedisk/duy/depth_predictions/depth.pkl', 'rb'))
image = image[12:-12, 12:-12]
image = cv2.resize(image, (640, 480))
depth = cv2.resize(depth, (640, 480))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Displacement_Field()
pth_path = './log/snapshot/epoch-last.pth'
model.load_state_dict(torch.load(pth_path)['model'])
model.to(device)
model.eval()

image = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
depth = torch.from_numpy(depth.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
depth = (depth - depth.min()) / (depth.max() - depth.min())

pred = model(image, depth)

pred = pred.detach().cpu().numpy()[0, 0]

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(pred)
plt.figure()
plt.imshow(depth.cpu().numpy().squeeze())
plt.show()
