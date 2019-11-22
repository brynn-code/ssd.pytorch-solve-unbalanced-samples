from data import *
from matplotlib import pyplot as plt
import cv2
from SIXray import SIXrayDetection
from train import args
from utils import SSDAugmentation

TRAIN_SET_PATH=HOME+"/test_data/"
test_set = SIXrayDetection(TRAIN_SET_PATH, image_sets=HOME + "train.id", transform=SSDAugmentation(coco['min_dim'],MEANS))
img_id = 0
image = test_set.pull_image(img_id)
annos = test_set.pull_annotation(img_id)

for anno in annos:
    cv2.rectangle(image, anno[1], anno[2], (0, 0, 255), 2)
    name = anno[0]
    if name == '带电芯充电宝':
        name = 'Core'
    else:
        name = 'Core_Less'
    cv2.putText(image, name, (anno[1][0], anno[1][1] - 5), 0,
                0.6, (0, 0, 255), 2)

plt.figure(figsize=(10, 10))
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.show()
