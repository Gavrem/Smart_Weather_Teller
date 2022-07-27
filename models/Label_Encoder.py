import numpy as np
import pandas as pd

def Label_Encoder(y):
    y=y.values
    #arranging labels
    y=np.where(y == 'Cold', 1, y)
    y=np.where(y == 'Fog', 2, y)
    y=np.where(y == 'Precipitation', 3, y)
    y=np.where(y == 'Rain', 4, y)
    y=np.where(y == 'Snow', 5, y)
    y=np.where(y == 'Storm', 6, y)
    y=np.where(y == 'Hail', 6, y)


    return y

