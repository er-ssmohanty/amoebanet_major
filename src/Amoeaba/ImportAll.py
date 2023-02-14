import keras
from keras import layers
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Dense, GlobalAveragePooling2D,MaxPooling2D
from keras.regularizers import l2
from keras.models import Model


from Amoeaba.Normal_cell import *
from Amoeaba.Build_amoeba_net import *
from Amoeaba.Reduction_cell import *
from Amoeaba.Stem import *


#from src.Amoeaba.Normal_cell import *
#from src.Amoeaba.Build_amoeba_net import *
#from src.Amoeaba.Reduction_cell import *
#from src.Amoeaba.Stem import *
