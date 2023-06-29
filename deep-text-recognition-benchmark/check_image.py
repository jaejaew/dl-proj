#%%
import matplotlib.pyplot as plt
import cv2
import json

data = json.load(open('./data/labeled/label.json'))
images = data['images']
img = cv2.imread('./data/raw/'+images[1]['file_name'])
plt.imshow(img)