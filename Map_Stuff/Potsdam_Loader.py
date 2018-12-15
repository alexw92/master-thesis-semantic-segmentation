import glob
import os
import gdal
import matplotlib.pyplot as plt
import numpy as np
import random

# get env
prefix = 'G:/'
with open('../ANN_DATA/ENV') as env_file:
    env = env_file.readline()
if env == 'PC2':
    prefix = 'G:/ANN_Data/Potsdam'
elif env == 'Main':
    prefix = 'G:/ANN_Data/Potsdam'

image_path_rgb = prefix+'/2_Ortho_RGB'  # RGB (tif)
image_path_rgbir = prefix+'/Potsdam_Ortho_RGBIR'  # RGBIR (tif)
image_path_labels = prefix+'/5_Labels_for_participants' # labels/groundtruth (tif)

initialized = False
tif_files_rgb = []
tif_files_rgbir = []
img_index = 0
cursor_index = 0


def initialize(img_path_rgb=image_path_rgb, img_path_rgbir=image_path_rgbir):
    """
    Initializes data, sets up paths, etc.
    See maploader.py for more details

    :param img_path_rgb:
    :param img_path_rgbir
    :return:
    """
    os.chdir(img_path_rgb)
    for file in glob.glob("*.tif"):
        tif_files_rgb.append(file)
    if len(tif_files_rgb) == 0:
        print('No RGB tif files found')
    os.chdir(img_path_rgbir)
    for file in glob.glob("*.tif"):
        tif_files_rgbir.append(file)
    if len(tif_files_rgbir) == 0:
        print('No RGBIR tif files found')
    global initialized
    initialized = True


def get_sample(rgbir=False, img_width=224, img_height=224, big_img_width=6000, big_img_height=6000):
    if not initialized:
        initialize()
    file_list = tif_files_rgb
    file_path = image_path_rgb
    label_path = image_path_labels
    mode = "RGB"
    if rgbir:
        file_list = tif_files_rgbir
        file_path = image_path_rgbir
        mode = "RGBIR"

    # check cursor_index
    max_cursor_index = (big_img_height//img_height) * (big_img_width//img_width) - 1

    if cursor_index > max_cursor_index:
        global cursor_index
        cursor_index = 0
        global img_index
        img_index = (img_index + 1) % len(file_list)
        if img_index == 0:
            print("Epoch finished! Restarting from beginning!")

    # get image
    file = file_list[img_index]
    row_index = cursor_index // (big_img_width//img_width)
    col_index = cursor_index % (big_img_width//img_width)
    row_start = col_index * img_width
    row_end = row_start + img_width
    col_start = row_index * img_height
    col_end = col_start + img_height

    img_rgb = gdal.Open(file_path + "/" + file).ReadAsArray().transpose([1, 2, 0])
    x = img_rgb[col_start:col_end, row_start:row_end]

    label_file = file.replace(mode, "label")
    img_rgb = gdal.Open(label_path + "/" + label_file).ReadAsArray().transpose([1, 2, 0])
    y = img_rgb[col_start:col_end, row_start:row_end]

    # update cursor index
    global cursor_index
    cursor_index = cursor_index + 1

    # return image
    return x, y


def plot_sample(img_mask, train_mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('image')
    ax[0].imshow(img_mask)
    ax[1].imshow(train_mask, vmin=0, vmax=10, cmap='tab10')
    ax[1].set_title('label')
    plt.show()


def unique_map_pixels_vectorized(imgs):
    N,H,W = len(imgs), imgs.shape[2], imgs.shape[3]
    img2D = imgs.transpose(0, 2, 3, 1).reshape(-1,3)
    ID = np.ravel_multi_index(img2D.T,img2D.max(0)+1)
    _, firstidx, tags = np.unique(ID,return_index=True,return_inverse=True)
    return tags.reshape(N,H,W), img2D[firstidx]

class PotsdamDataGenerator:
    @staticmethod
    def to_one_hot(img, n_cls):
        convert_dict = {
            (255, 255, 255): 0,  # weiß, straße
            (255, 0, 0): 1,  # rot, baustelle/müll
            (0, 0, 255): 2,  # blau, gebäude
            (255, 255, 0): 3,  # gelb, auto
            (0, 255, 0): 4,  # grün, baum
            (0, 255, 255): 5  # hellblau, ground/wiese
        }
        # convert colors to single channel, todo
       # img = np.vectorize(convert_dict.get)(img)
       # [ for i in ]
        w, h, _ = img.shape
        converted = np.zeros([w, h], dtype=int)
        for wi in range(0, w):
            for hi in range(0, h):
                converted[wi][hi] = convert_dict.get(tuple(img[wi][hi]))
        return (np.arange(n_cls) == converted[:, :, None]).astype(int)

    def __init__(self, width, height, big_width, big_height, n_cls, batch_size):
        self.lines = []
        self.width = width
        self.big_width = big_width
        self.height = height
        self.big_height = big_height
        self.n_cls = n_cls
        self.batch_size = batch_size
        self.i = 0

    def get_sample(self):
        if self.i == 0:
            random.shuffle(self.lines)
        orig, gt = get_sample()
        # n_cls+1 weil obergrenze bei np.arrange nicht drin ist
        gt = gt = PotsdamDataGenerator.to_one_hot(gt, self.n_cls + 1)
        return orig, gt

    def get_batch(self):
        while True:
            orig_batch = np.zeros((self.batch_size, self.width, self.height, 3))
            gt_batch = np.zeros((self.batch_size, self.width, self.height, self.n_cls + 1))
            for i in range(self.batch_size):
                orig, gt = self.get_sample()
                orig_batch[i] = orig
                gt_batch[i] = gt
            yield orig_batch, gt_batch

    def get_size(self):
        return (self.big_height//self.height) * (self.big_width//self.width) - 1


if __name__ == '__main__':
    # x, y = get_sample()
    # for i in range(1, 40):
    #     x, y = get_sample(img_height= 500, img_width=500)
    #     plot_sample(x, y)
    # print(x.shape)
    initialize()
    gen = PotsdamDataGenerator(224, 224, 6000, 6000, 6, 1)
    for i in gen.get_batch():
        x, y = i