#import all libraries

import os
import h5py

import numpy as np
from numpy import linalg as LA

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from scipy.spatial import distance
from PIL import Image

import pandas as pd
import streamlit as st
from PIL import Image

import io

# configures the page for UI
st.set_page_config(page_title="Clothing Finder App", layout="centered")

# title
st.title("Clothing Finder App")

# creates upload image screen
query = "Riya_learning copy/query images/pantQuery.png"
query1 = st.file_uploader(label="Upload your clothing image!", accept_multiple_files=False, type=['png', 'jpeg', 'jpg'])

# if the user uploads an image, display it
if query1 is not None:
    query = query1
    query_display = Image.open(query)
    st.markdown("## Your image:")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.write(query_display, use_container_width=True)

    # creating the VGG16 Model
    # include_top is false since you don't want to include the top layer, which classifies the object (not necessary for image similarity)
    # the input shape is (height, width, #of channels)
    model = VGG16(weights = 'imagenet', input_shape = ((224, 224, 3)), pooling = 'max', include_top = False)

    #creating a class with the variables & functions for extracting a feature vector
    class VGGNet:
        def __init__(self):
            self.input_shape = (224, 224, 3)
            # pre-trained weights
            self.weight = 'imagenet'
            self.pooling = 'max'
            self.model = VGG16(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top= False)

            # creates a NumPy array filled entirely with 0s
            # 1 image, 224x224, 3 (RGB channels all initialized to 0 so that pixed is purely black)
            # can think of it like a dummy test case to kind of "warm up" the model to prevent errors later
            # output is never used
            self.model.predict(np.zeros((1,224, 224, 3)))
        

        # output is 512 features in a vector
        def extract_feat(self, img_path):
            # loads image from disk and resizes it to 224x224 so all images are same shape (needed for CNN)
            img = image.load_img(img_path, target_size = (self.input_shape[0], self.input_shape[1]))
            
            # creates an aray of shape (224, 224, 3)
            # for each of the 224*224 pixels, the RGB value (3 sep values) are stored
            img = image.img_to_array(img)

            # axis = 0 --> add new dimension at front
            # turns image into batch with 1 image in it (requires this shape of (batch size, height, width, channels))
            # now image.shape = (1, 224, 224, 3)
            img = np.expand_dims(img, axis = 0)

            # function preprocess_input customized specifically for VGG16 (to match the data VGG16 was originally trained with)
            # 1. reorders channels from RGB --> BGR
            # 2. converts pixel from [0, 255] to float32 (essentially into decimals)
            # 3. centers colors around 0 by subtracting average color values from each pixel (usually around [-128, 128])
            img = preprocess_input(img)

            # returns a 512-dimensional feature vector (NumPy array of shape (1, 12) )
            feat = self.model.predict(img)

            # normalizing the vector by dividing the actual vector by the length/magnitude of vector to get the "unit" vector of length 1
            # doing this so you can fairly compare feature vectors using cosine similarity later
            norm_feat = feat[0]/LA.norm(feat[0])
            return norm_feat
        

    model = VGGNet()

    # reading the clothing database csv file
    clothing_database = pd.read_csv("clothing_database.csv")

    # if the h5 file doesn't already exist (basically if it was created on a previous run already), then create the file
    if not os.path.exists("CNNFeatures.h5"):

        # initializes list of features & image names
        feats = []
        names = []

        for i, row in clothing_database.iterrows():
            # for each image in the database, add the image name to the names list
            img_name = str(row["image name"]) + ".png"
            img_path = os.path.join("all images", img_name)

            # making sure file exists
            if os.path.exists(img_path):
                print("Extracting features from: ", img_name)
                # for each image in the database, create & add the feature vector to the features list
                X = model.extract_feat(img_path)
                feats.append(X)
                names.append(img_name)

        # turn feature list into an array
        feats = np.array(feats)

        # initialize the h5 file
        output = "CNNFeatures.h5"
        print(" writing feature extraction results to h5 file")

        # writing the features to the h5 file
        h5f = h5py.File(output, 'w')
        h5f.create_dataset('dataset_1', data = feats)
        h5f.create_dataset('dataset_2', data = np.bytes_(names))
        h5f.close()
        print("file created")



    with h5py.File("CNNFeatures.h5", 'r') as h5f:

        # override features into a new NumPy array containing same data as before but being read from the file
        feats = h5f['dataset_1'][:]
        imgNames = h5f['dataset_2'][:]

    # creating feature vector for query image
    queryImg = query
    query_feat = model.extract_feat(queryImg)

    # comparing query image's feature vector to one image vector at a time from feats (being looped through with iteration variable feat)
    # cosine distance is outputted from [0, 2] (lower is better)
    # similarity = 1 - distance.cosine(a,b) --> range [-1, 1] (higher is better)
    # 1 --> most similar, 0 --> no similarity, -1 --> compeltely opposite
    scores = [1 - distance.cosine(query_feat, feat) for feat in feats]
    scores = np.array(scores)

    # np.argsort(scores) returns the indices that would sort an array
    # [:: -1] reverse so in descending order
    rank_ID = np.argsort(scores)[:: -1]

    # creates new array in the order of the indices given by the rank_ID array (already sorted from most similar to  least)
    rank_score = scores[rank_ID]

    # Get top_n matches
    top_n = 3
    top_matches = rank_ID[: top_n]
    top_scores = rank_score[:top_n]

    # display top matches on the web page
    st.markdown(f"## Your top {top_n} matches!")
    for i, (image_id, score) in enumerate(zip(top_matches, top_scores)):
        # safety check to make sure filenames are stored correctly
        image_name = imgNames[image_id].decode('utf-8') if isinstance(imgNames[image_id], bytes) else imgNames[image_id]
        brand = clothing_database.iloc[image_id]["brand"]
        link = clothing_database.iloc[image_id]["link"]

        # create two columns to display image in the left column & text in the right column
        col1, col2 = st.columns([1, 2])

        # print image based on width of left column
        with col1:
            image_path = os.path.join("all images", image_name)
            st.image(Image.open(image_path), use_container_width=True)

        with col2:    
            st.markdown(f"{i+1}. Similarity Score: {score: .4f}")
            st.markdown(f" Brand: {brand}")
            st.markdown(f" Link: {link}")