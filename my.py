import  os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from itertools import  islice
from util.visualizer import save_images
from util import html
import matplotlib.pyplot as plt
import numpy as np
import pylab


opt = TrainOptions().parse()
dataset = create_dataset(opt)
dataset_size = len(dataset)
print('The number of training images = %d' % dataset_size)

print(dataset)


for i, data in enumerate(dataset):

    if i == 1:
        print(data['A'].shape)
        print(data['A'][0][1])
        image = data['A'][0][1].numpy()
        #print(image)

        #break


image = np.mat(image)


fig = plt.figure(figsize=(100, 100))
plt.subplot(111)
plt.title('123')
plt.imshow(image,cmap=plt.cm.gray)
plt.imsave('./image.png',image)
plt.show()
print(image)

plt.imsave('./image.png',image)