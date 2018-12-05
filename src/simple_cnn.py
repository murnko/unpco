import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from src.utils import create_raw_data_df_list, create_csv_data
from src.customdataset import CustomDatasetFromImages


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

path_bi_data = '../data/raw_data'

# get some random training images

df_bi_labeled = create_csv_data(path_bi_data, [2])
msk = np.random.rand(len(df_bi_labeled)) < 0.8
train = df_bi_labeled[msk]
test = df_bi_labeled[~msk]
# print(df_bi_labeled.depth_2.value_counts())
ds_train = CustomDatasetFromImages(train,transform)
ds_test = CustomDatasetFromImages(test,transform)
# crop_top(df_bi_labeled)

train_loader = torch.utils.data.DataLoader(dataset=ds_train,
                                                batch_size=5,
                                                shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=ds_test,
                                                batch_size=5,
                                                shuffle=True)
classes = ['Good', 'Bad']
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
