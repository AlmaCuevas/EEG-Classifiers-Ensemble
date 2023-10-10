import numpy as np
from sklearn.model_selection import train_test_split

def data_transform(x, subtract_mean:bool =True, subtract_axis:int =0, transpose: bool =False):
    # Transpose the second and third dimension
    if transpose:
        x = np.transpose(x, (0, 2, 1))

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(x, axis=subtract_axis)
        mean_image = np.expand_dims(mean_image, axis=subtract_axis)
        x -= mean_image
    return x


def train_test_val_split(dataX, dataY, valid_flag: bool = False):
    train_ratio = 0.75
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

    if valid_flag:
        validation_ratio = 0.15
        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=test_ratio / (test_ratio + validation_ratio))
    else:
        x_val = None
        y_val = None
    return x_train, x_test, x_val, y_train, y_test, y_val