import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern

def extract_features_from_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Feature 1: Color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256]).flatten()

    # Feature 2: LBP (texture)
    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 27),
                                 range=(0, 26))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)

    return np.concatenate([hist, hist_lbp])

def extract_single_image_features(image_path):
    return extract_features_from_image(image_path)

def load_data_labels(features_dir):
    X, y = [], []
    for file in os.listdir(features_dir):
        if file.endswith(".npz"):
            data = np.load(os.path.join(features_dir, file))
            X.append(data["features"])
            y.append(data["label"])
    return np.array(X), np.array(y)
