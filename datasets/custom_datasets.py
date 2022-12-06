from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd

class vgg_face2(Dataset):
    def __init__(self, csv_file, root_dir, mode, transform):
        """
        Args:
            csv_file (string): Path to the FAN landmarks csv file.
            root_dir (string): Path to VGG-Face2
            transform (callable, optional): Rescale and normalize for ResNet.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform_image = transform
        self.root_dir = root_dir
        self.mode = mode

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if self.mode == "train": folder_name = "train"
        else: folder_name = "test"

        img_path = f"{self.root_dir}/data/{folder_name}/{self.landmarks_frame.iloc[idx, 0]}"
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        [X, Y, W, H] = self.landmarks_frame.iloc[idx, 1:5]
        # expand the bounding box
        X_e = int(X - round(0.2 * W))  # 20% to the left
        Y_e = int(Y - round(0.1 * H))  # 10% to the top
        W_e = int(round(1.4 * W))  # 20% to the right
        H_e = int(round(1.3 * H))  # 20% to the bottom
        face_crop = image[Y_e:Y_e + H_e, X_e:X_e + W_e]

        landmarks = self.landmarks_frame.iloc[idx, 7:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)  # convert to 68x2 matrix

        # move points back from whole image to face crop
        landmarks[:, 0] = landmarks[:, 0] - X_e
        landmarks[:, 1] = landmarks[:, 1] - Y_e

        # transform the face crop
        orig_height, orig_width = face_crop.shape[1], face_crop.shape[0]
        face_crop_tran = self.transform_image(face_crop)
        mod_height, mod_width = face_crop_tran.shape[1], face_crop_tran.shape[2]

        # transform the landmarks to 224x224 face
        landmarks_tran = landmarks.copy()

        # save the transforms in x and y dimensions while rescaling
        tf_x, tf_y = mod_height/orig_height, mod_width/orig_width 

        min_x = np.amin(landmarks[:, 0])
        min_y = np.amin(landmarks[:, 1])

        # move each axis to 0
        landmarks_tran[:, 0] = landmarks[:, 0] - min_x
        landmarks_tran[:, 1] = landmarks[:, 1] - min_y
        
        # scale the same ratio as image dimensions
        landmarks_tran[:,0] = landmarks_tran[:,0] *  tf_x
        landmarks_tran[:, 1] = landmarks_tran[:, 1] * tf_y

        # move landmarks back to their place on the image
        landmarks_tran[:, 0] = landmarks_tran[:, 0] + (min_x * tf_x)
        landmarks_tran[:, 1] = landmarks_tran[:, 1] + (min_y * tf_y)
    
        sample = {"image": face_crop_tran, 
                "landmarks": landmarks_tran, 
                "scale_ratio": (tf_x, tf_y)}
                
        return sample
