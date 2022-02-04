import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import mtcnn

# Load Images
def load_images_fer(data_path):
    # Transformations
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            #torchvision.transforms.Normalize(mean=[0.445, 0.445, 0.445],
                                            #std=[0.269, 0.269, 0.269]),
                                            )
    # Prepare Dataframes
    data_df = pd.read_csv(data_path)
    train_data_df = data_df.iloc[:28708]
    test_data_df = data_df.iloc[32298:]

    # Prepare Lists
    train_lst = []
    for i in range(0, len(train_data_df)):
        pixels = train_data_df.iloc[i]['pixels']
        emotion = train_data_df.iloc[i]['emotion']
        pixels_list = pixels.split()
        map_obj = map(int, pixels_list)
        pixels_list = list(map_obj)
        img_array = np.array(pixels_list, dtype=np.uint8)
        img_array = img_array.reshape([48, 48])
        img_array = torchvision.transforms.ToPILImage()(img_array)
        img_array = torchvision.transforms.functional.to_grayscale(img_array, num_output_channels=3)
        img = trans(img_array)
        train_lst.append( (img, emotion) )

    test_lst = []
    for i in range(0, len(test_data_df)):
        pixels = test_data_df.iloc[i]['pixels']
        emotion = test_data_df.iloc[i]['emotion']
        pixels_list = pixels.split()
        map_obj = map(int, pixels_list)
        pixels_list = list(map_obj)
        img_array = np.array(pixels_list, dtype=np.uint8)
        img_array = img_array.reshape([48, 48])
        img_array = torchvision.transforms.ToPILImage()(img_array)
        img_array = torchvision.transforms.functional.to_grayscale(img_array, num_output_channels=3)
        img = trans(img_array)
        test_lst.append( (img, emotion) )

    return (train_lst, test_lst)

# Detect face with MTCNN
def face_detect():
    detector = mtcnn.MTCNN()
    face = detector.detect_faces()
    return face
