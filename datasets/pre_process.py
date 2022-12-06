"""
Perform data cleaning according to DECA paper: If the distance between FAN[1] landmarks on shifted face crops on the
face image larger than a threshold, discard the image. Also discard the image if no landmarks are found or the image is
too small. Face crops provided with the dataset and landmark coordinations are saved as a csv file.

[1]: https://github.com/1adrianb/face-alignment
"""

import cv2
import csv
import face_alignment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class pre_process:

    def __init__(self, dataset, dataset_dir, mode, device):
        self.dataset = dataset #e.g. vggface2
        self.dataset_dir = dataset_dir
        self.data_path = f"{self.dataset_dir}/data"
        self.meta_path = f"{self.dataset_dir}/meta"
        self.mode = mode

        self.fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
        #Euler does not let internet connection to download models from website. The files can be downloaded from the
        #links given in error and moved to .cache of torch.

    def read_files(self):
        if self.dataset == 'vggface2':
            meta = []

            if self.mode == "train":
                train_list_path = f"{self.data_path}/train_list.txt"
                img_meta_train_path = f"{self.meta_path}/bb_landmark/loose_bb_train.csv"

                with open(train_list_path) as f:
                    img_list = [line.rstrip() for line in f]

                img_meta_train_csv = csv.DictReader(open(img_meta_train_path, "r"))
                for row in img_meta_train_csv:
                    meta.append(dict(row))
                # e.g. {'NAME_ID': 'n000002/0001_01', 'X': '161', 'Y': '140', 'W': '224', 'H': '324'}
            else:
                test_list_path = f"{self.data_path}/test_list.txt"
                img_meta_test_path = f"{self.meta_path}/bb_landmark/loose_bb_test.csv"

                with open(test_list_path) as f:
                    img_list = [line.rstrip() for line in f]
                img_meta_test_csv = csv.DictReader(open(img_meta_test_path, "r"))

                for row in img_meta_test_csv:
                    meta.append(dict(row))

            return img_list, meta

    def data_clean(self, img_list, meta_list):
        dicts_to_csv = []
        for file_name in img_list:
            if self.mode == "train":
                img_path = f"{self.data_path}/train/{file_name}"
            else:
                img_path = f"{self.data_path}/test/{file_name}"

            img = cv2.imread(img_path)
            img_shape = img.shape
            height, width = img_shape[0], img_shape[1]

            bounding_box = next(item for item in meta_list if item['NAME_ID'] == file_name[:-4]) #remove .jpg in the end
            meta_list.remove(bounding_box)
            X = int(bounding_box['X'])
            Y = int(bounding_box['Y'])
            W = int(bounding_box['W'])
            H = int(bounding_box['H'])

            # shift bounded box by 5% to bottom right
            eps_x = round(0.05*W)
            eps_y = round(0.05*H)

            # shifted top_left corner
            X_s = int(X + eps_x)
            Y_s = int(Y + eps_y)

            # expand the bounding boxes
            X_e = int(X - round(0.2 * W))  # 20% to the left
            Y_e = int(Y - round(0.1*H)) # 10% to the top
            X_s_e = int(X_s - round(0.2 * W))  # 20% to the left
            Y_s_e = int(Y_s - round(0.1 * H))  # 10% to the top
            W_e = int(round(1.4 * W)) # 20% to the right
            H_e = int(round(1.3 * H))  # 20% to the bottom

            # check if modified bounding boxes are still in the image
            # check if input image is too small, otherwise FAN complains
            # check if the crop is too large, otherwise cuda out of memory error
            in_the_image = X_e>0 and Y_e>0 and X_s_e+W_e<width and Y_s_e+H_e<height
            too_small = 100>height or 100>width
            too_large = W_e>500 or H_e>500

            if (in_the_image) and (not too_small) and (not too_large):
                img_one = img[Y_e:Y_e+H_e, X_e:X_e+W_e]
                img_two = img[Y_s_e:Y_s_e+H_e, X_s_e:X_s_e+W_e ]

                # if landmarks are found, shape is (1, 68, 2)
                landmarks_one = self.fan.get_landmarks_from_image(img_one)
                landmarks_two = self.fan.get_landmarks_from_image(img_two)

                if np.shape(landmarks_one) == (1, 68, 2) and np.shape(landmarks_two) == (1, 68, 2):
                    landmarks_one = landmarks_one[0]
                    landmarks_two = landmarks_two[0]

                    # to visualize landmarks on face crops
                    # detection = landmarks_one
                    # x = detection[:, 0] + X_e
                    # y = detection[:, 1] + Y_e
                    # plt.figure()
                    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # plt.scatter(x, y, 2)
                    # plt.show()
                    #
                    # detection = landmarks_two
                    # x = detection[:, 0]
                    # y = detection[:, 1]
                    # plt.figure()
                    # plt.imshow(cv2.cvtColor(img_two, cv2.COLOR_BGR2RGB))
                    # plt.scatter(x, y, 2)
                    # plt.show()

                    difference = landmarks_two-landmarks_one
                    normalized_difference_x = abs(difference[:,0]-eps_x)/width
                    normalized_difference_y = abs(difference[:,1]-eps_y)/height
                    normalized_difference = np.linalg.norm([normalized_difference_x, normalized_difference_y], axis=0)

                    # discard images where the difference of landmarks between original crop and shifted crop is large
                    if np.max(normalized_difference) < 0.1:
                        landmark_dict = {'NAME_ID': file_name, 'X': X, 'Y': Y, 'W': W, 'H': H, 'height': height, 'width': width}

                        # translate landmark back to original img from cropped
                        detection = landmarks_one
                        x_lm = detection[:, 0] + X_e
                        y_lm = detection[:, 1] + Y_e

                        for i in range(len(x_lm)):
                            landmark_dict[f'x{i}']= int(x_lm[i])
                            landmark_dict[f'y{i}']= int(y_lm[i])

                        dicts_to_csv.append(landmark_dict)
                        print(f'done cleaning{file_name}')
        return dicts_to_csv

    def create_csv(self, dicts_list):
        keys = dicts_list[0].keys()
        with open(f"{self.meta_path}/fan_landmarks_{self.mode}.csv", 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(dicts_list)

    def filter_size(self, landmark_csv_path, mode):
        # divide data as larger than 224x224 or not
        landmarks_frame = pd.read_csv(open(landmark_csv_path, "r"))
        size_mask = (landmarks_frame['height'] >= 224) & (landmarks_frame['width'] >= 224)
        larger = landmarks_frame[size_mask]
        smaller = landmarks_frame[~size_mask]
        larger.to_csv(f"../data/landmarks_large_{mode}.csv", index=False)
        smaller.to_csv(f"../data/landmarks_small_{mode}.csv", index=False)


pre_proc = pre_process(dataset='vggface2', dataset_dir="path/to/VGG-Face2", device='cuda', mode="train") #device cpu or cuda
# to create csv files with landmarks
img_list, meta_list = pre_proc.read_files()
csv_file_to_write = pre_proc.data_clean(img_list, meta_list)
pre_proc.create_csv(csv_file_to_write)

# to filter based on size
# pre_proc.filter_size("path/to/fan_landmarks/train/fan_landmarks_train.csv", mode="train")
