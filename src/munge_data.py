# importing necessary libraries
import os 
from pathlib import Path
from glob import glob


# Path to the dataset
path = Path('dataset')
image_path = path/'indoorCVPR_09/Images' # Path to the images
image_files = glob(str(image_path/'**/*.jpg')) # List of all the images

# label encoding for the classes
class_names = [p.name for p in image_path.glob('*') if p.is_dir()]

label2index = {v:k for k,v in enumerate(class_names)}
index2label = {v:k for k,v in label2index.items()}

# Creating a dataframe with the image path and the corresponding label
import pandas as pd
df = pd.DataFrame({'image':image_files})
df['label'] = df['image'].map(lambda x: label2index[x.split('/')[-2]])

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Saving the train and test dataframe
train_df.to_csv('dataset/train.csv', index=False)
test_df.to_csv('dataset/test.csv', index=False)
