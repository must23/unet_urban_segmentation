# import the necessary packages
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
import warnings
import itertools
from sklearn.metrics import confusion_matrix
from UNet_PyTorch import UNet
import os

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
classes = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
BATCH_SIZE = 10

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



def sliding_window(top, step=10, window_size=(20,20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]




def count_sliding_window(top, step=10, window_size=(20,20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c



def groupIterElements(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk



def metrics(predictions, gts, label_values=classes):
    cm = confusion_matrix(
            gts,
            predictions,)
    
    print("Confusion matrix :")
    print(cm)   
    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    print("---")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")
        
    return accuracy


WINDOW_SIZE = (300, 300) # Patch size
def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABELS_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(LABELS_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    
    # Switch the network to inference mode
    net.eval()

    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(tqdm(groupIterElements(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
            if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                    _pred = np.argmax(pred, axis=-1)
                    # fig = plt.figure()
                    # fig.add_subplot(1,3,1)
                    # plt.title('Original')
                    # plt.imshow(np.asarray(255 * img, dtype='uint8'))
                    
                    # fig.add_subplot(1,3,2)
                    # plt.title('Ground truth') 
                    # plt.imshow(gt)
                    
                    # fig.add_subplot(1,3,3)
                    # plt.title('Prediction')
                    # plt.imshow(convert_to_color(_pred))
                    # plt.show()

            # Build the tensor
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
            
            # Do the inference
            outs = net(image_patches)
            outs = outs.data.cpu().numpy()
            
            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1,2,0))
                pred[x:x+w, y:y+h] += out
            del(outs)

        pred = np.argmax(pred, axis=-1)

        # fig = plt.figure()
        # fig.add_subplot(1, 3, 1)
        # plt.title('Original')
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
                
        # fig.add_subplot(1, 3, 2)
        # plt.title('Ground truth')
        # plt.imshow(gt)
        
        # fig.add_subplot(1, 3, 3)
        # plt.title('Prediction')
        # plt.imshow(convert_to_color(pred))
        # plt.show()

        all_preds.append(pred)
        all_gts.append(gt_e)

        # Compute some metrics
        metrics(pred.ravel(), gt_e.ravel())
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy









# Parameters
# Number of input channels (e.g. RGB)
IN_CHANNELS = 3
MAIN_FOLDER = "C:/Users/Desktop/ECCE 633/Assignment#2/patches/ISPRS Dataset/"
DATA_FOLDER = MAIN_FOLDER + 'Images/Image_{}.tif'
LABELS_FOLDER = MAIN_FOLDER + 'Labels/Label_{}.tif'


N_CLASSES = len(classes)                   # Number of classes
WEIGHTS = torch.ones(N_CLASSES)           # Weights for class balancing

test_ids =  list(range(2000,2400))
testset = Load_dataset(test_ids)

testload = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)



# #######################################
# # main()
# #######################################

print(testset.__len__(), "images loaded for testing!")

# load our the network weights from disk, flash it to the current
# device, and set it to evaluation mode
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("CUDA FOUND! Testing model on the GPU...\n")
else:
    DEVICE = torch.device("cpu")
    print("CUDA NOT FOUND! Testing model on the CPU...\n")


# PREPARE NETWORK FOR LEARNING
networkName = 'UNet'
model = UNet(n_channels=IN_CHANNELS, n_classes=N_CLASSES)

model.to(DEVICE)
print('Network loading routine completed...\n')

model.load_state_dict(torch.load('C:/Users/Desktop/ECCE 633/Assignment#2/Trained Model/UNet_trained_segmentation.pth'))


_, all_preds, all_gts = test(model, test_ids, all=True, stride=150)

for p, id_ in zip(all_preds, test_ids):
    img = convert_to_color(p)
    plt.imshow(img) and plt.show()
    io.imsave('C:/Users/Desktop/ECCE 633/Assignment#2/Results/Testing Images/Epoch_{}.png'.format(id_), img)
