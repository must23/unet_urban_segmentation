# import the necessary packages
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision as tv
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from skimage import io
from torch.autograd import Variable
from UNet_PyTorch import UNet
import warnings



warnings.filterwarnings("ignore")



# Let's define the standard ISPRS color palette
palette = {0: (255, 255, 255), # Impervious surfaces (white)
           1: (0, 0, 255),     # Buildings (blue)
           2: (0, 255, 255),   # Low vegetation (cyan)
           3: (0, 255, 0),     # Trees (green)
           4: (255, 255, 0),   # Cars (yellow)
           5: (255, 0, 0),     # Clutter (red)
           6: (0, 0, 0)}       # Undefined (black)
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """ '(From 0 to 6)'
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


class Load_dataset(torch.utils.data.Dataset):
    def __init__(self, ids):
        super(Load_dataset, self).__init__()
        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.label_files = [LABELS_FOLDER.format(id) for id in ids]
        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

    def __len__(self):
        return len(self.data_files)  # the length of the used data

    def __getitem__(self, idx):
        #         Pre-processing steps
        #     # Data is normalized in [0, 1]
        self.data = 1/255 * np.asarray(io.imread(self.data_files[idx]).transpose(
                (2, 0, 1)), dtype='float32')
        self.label = np.asarray(convert_from_color(
            io.imread(self.label_files[idx])), dtype='int64')
        data_p, label_p = self.data,  self.label
        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))




# BATCH_SIZE -> how many images you can train at once, determined by how much memory you have (play around with it)
# EPOCH -> how many times we should train model across entire dataset (~8-10 EPOCH is sufficient)
BATCH_SIZE = 10
EPOCH = 10

# Parameters
# Number of input channels (e.g. RGB)
IN_CHANNELS = 3
MAIN_FOLDER = "C:/Users/Desktop/ECCE 633/Assignment#2/Assignment#2/patches/ISPRS Dataset/"
DATA_FOLDER = MAIN_FOLDER + 'Images/Image_{}.tif'
LABELS_FOLDER = MAIN_FOLDER + 'Labels/Label_{}.tif'


classes = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
N_CLASSES = len(classes)                   # Number of classes
WEIGHTS = torch.ones(N_CLASSES)           # Weights for class balancing

train_ids =list(range(0, 2000))
trainset = Load_dataset(train_ids)


###############################################################
trainload = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)



def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size



def train(net, train_loader, epochs, scheduler=None, weights=WEIGHTS):

    optimizer = optim.SGD(model.parameters(), lr=0.0015, momentum=0.9, weight_decay=5e-4)

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()
    print('Training Started...')

    since = time.time()
    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()



        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_-100):iter_])

            if iter_ % 40 == 39:
                
                rgb = np.asarray(
                    255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)

                gt = target.data.cpu().numpy()[0]
                
                # Display performance after each mini-batch (40)
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(e, epochs, batch_idx, len(
                    train_loader), 100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))

               
            iter_ += 1
            del(data, target, loss)

    # Display Loss after each Epoch
    plt.title('Training loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(mean_losses[:iter_]) and plt.show()  
       
    time_elapsed = time.time() - since
    print('-----------------------------------------')
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('-----------------------------------------')




print(trainset.__len__(), "images loaded for training!")

# load our the network weights from disk, flash it to the current
# device, and set it to evaluation mode
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("CUDA FOUND! Training model on the GPU...\n")
else:
    DEVICE = torch.device("cpu")
    print("CUDA NOT FOUND! Training model on the CPU...\n")


# PREPARE NETWORK FOR LEARNING
networkName = 'UNet'
model = UNet(n_channels=IN_CHANNELS, n_classes=N_CLASSES)

model.to(DEVICE)
print('Network loading routine completed...\n')


# TRAIN NETWORK
train(model, train_loader=trainload, epochs=EPOCH, weights=WEIGHTS)

# SAVE TRAINED MODEL
save_path = 'C:/Users/Desktop/ECCE 633/Assignment#2/Trained Model/' + networkName + '_trained_segmentation.pth'
torch.save(model.state_dict(), save_path)
print('Trained model saved in: ' + save_path + '\n')
