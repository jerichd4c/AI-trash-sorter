import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

# image config

IMG_SIZE= (224, 224)
BATCH_SIZE= 32
EPOCHS= 20
NUM_CLASSES= 12 
DATASET_PATH= 'dataset'