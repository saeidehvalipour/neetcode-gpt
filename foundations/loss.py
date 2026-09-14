import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float: 
        y_pred = np.clip(y_pred,1e-7, 1 - 1e-7)

        total_loss = 0.0

        for y,p in zip(y_true, y_pred):
            total_loss +=  -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return round(float(total_loss / len(y_pred)),4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        loss_per_sample = - np.sum((y_true * np.log(y_pred)), axis =1)
        
        return round(float(np.mean(loss_per_sample)), 4)
