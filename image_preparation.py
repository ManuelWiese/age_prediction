import numpy as np
import cv2
import os

SIZE = 64
SRC_PATH = './cleaned_data'
DST_PATH = "{}_{}".format(SRC_PATH, SIZE)
SUBFOLDERS = ["train", "test", "validate"]

os.mkdir(DST_PATH)

for subfolder in SUBFOLDERS:
    os.mkdir("{}/{}".format(DST_PATH, subfolder))

    filenames = os.listdir("{}/{}".format(SRC_PATH, subfolder))[:]
    src_filenames = ["{}/{}/{}".format(SRC_PATH, subfolder, filename) for filename in filenames]
    dst_filenames = ["{}/{}/{}".format(DST_PATH, subfolder, filename) for filename in filenames]

    for src_filename, dst_filename in zip(src_filenames, dst_filenames):
        original_image = cv2.imread(src_filename, 1)
        resized_image = cv2.resize(original_image, (SIZE, SIZE))

        cv2.imwrite(dst_filename, resized_image)
        

