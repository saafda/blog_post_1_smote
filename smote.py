from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
from sklearn.neighbors import NearestNeighbors
import cv2
from numpy.typing import NDArray

class SMOTE:
    """
    A class that implements Synthesized Minority Oversampling Technique

    Attributes:
        data (NDArray): data points to sample from.
    
    Methods:
        generate_data(): Generates data points to be used for SMOTE.
        convex_combine(): Main part of the SMOTE method.
        smote(): Uses generate_data() and convex_combine() to execute the SMOTE method.
        plot(): Help method for plotting.
    """
    
    def __init__(self, data_samples):
        self.data_samples = data_samples
        
    def generate_data(self, ratio:float, N:int) -> tuple[NDArray, NDArray]:
        """Generates N classification data points for two classes with given class ratio

        Args:
            ratio (float): Ratio of classes, of the form class_0:class_1.
                           Class 1 is the one to be classified. 
            N (int): Number of data points to be generated.

        Returns:
            tuple[NDArray, NDArray]: tuple containing class 0 and class 1 data points
        """
        class_0 = self.data_samples[self.data_samples[:, 2] == 0]
        class_1 = self.data_samples[self.data_samples[:, 2] == 1]
        N0 = int((1-ratio)*N)
        N1 = int(ratio*N)
        
        class_0_sampled = class_0[np.random.randint(class_0.shape[0], size=N0), :]
        class_1_sampled = class_1[np.random.randint(class_1.shape[0], size=N1), :]
        
        return class_0_sampled, class_1_sampled
        
    
    def convex_combine(self, minority_class: NDArray, k: int) -> NDArray:
        """Generates a convex combination between a randomly selected point in class 1
           and one of its random neighbours (also in class 1).

        Args:
            minority_class (NDArray): all data points of class 1.
            k (int): number of neighbours we would like to consider in kNN.

        Returns:
            NDArray: synthesized data point on the form (x, y, 1).
        """
        random_data_point = minority_class[rd.randint(0,len(minority_class))]
        all_neighbours = minority_class[:,:2]

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(all_neighbours)
        distances, indices = nbrs.kneighbors([random_data_point[:2]])
        kNN_indices = indices[0][1:] 
        kNN_points = minority_class[kNN_indices]

        random_neighbour = kNN_points[rd.randint(low=0, high=k)]
        
        alpha = rd.uniform()
        convex_combination = alpha * random_data_point[:2] + (1 - alpha) * random_neighbour[:2]
        convex_combination = convex_combination.astype(int)
        convex_combination = np.append(convex_combination, 1)
        return convex_combination

    def smote(self, minority_class: NDArray, M: int, k: int) -> NDArray:
        """Uses SMOTE to add M data points of class 1 to the given data points of class 1 (minority_class).

        Args:
            minority_class (NDArray): all data points of class 1.
            M (int): Number of data points to be synthesized.
            k (int): number of neighbours we would like to consider in kNN.

        Returns:
            NDArray: all class 1 data points in addition to the M new synthesized data points.
        """
        for i in range(M):
            synthesized_point = self.convex_combine(minority_class, k)
            minority_class = np.vstack([minority_class, synthesized_point])
        return minority_class
    
    def plot(self, class_0: NDArray, class_1: NDArray, image: NDArray, cmap: str, size: tuple[int, int]) -> None:
        """Plots class 0, class 1 and true class boundaries.

        Args:
            class_0 (NDArray): all data points of class 0.
            class_1 (NDArray): all data points of class 1.
            image (NDArray): true class boundaries.
            cmap (str): color map to use.
            size (int): figure size.
        """
        c00, c01 = class_0
        c10, c11 = class_1
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(size)
        ax[0].imshow(image, cmap=cmap[0])
        ax[0].scatter(c00[:,0], c00[:,1], c=c00[:,2], cmap=cmap[1])
        ax[0].scatter(c10[:,0], c10[:,1], c=c10[:,2], cmap=cmap[2])
        ax[0].axis('off')
        
        ax[1].imshow(image, cmap=cmap[0])
        ax[1].scatter(c01[:,0], c01[:,1], c=c01[:,2], cmap=cmap[1])
        ax[1].scatter(c11[:,0], c11[:,1], c=c11[:,2], cmap=cmap[2])
        ax[1].axis('off')
        plt.show()
        
class pi():
    """
    A simple class for generating data given by class boundaries from a pi symbol.

    Attributes:
        data (NDArray): data points to sample from.
        image (NDArray): true class boundaries.
    """
    def __init__(self):
        im_frame = Image.open('pi3.jpg')
        image = np.array(im_frame.getdata())
        r, g, b = np.transpose(image)
        image = r.reshape((1200, 1200))
        N = 3000
        data = np.zeros((N, 3))
        for n in range(N):
            i, j = rd.randint(0, 1200, 2)
            if image[i, j] == 255:
                data[n] = (j, i, 0)
            else:
                data[n] = (j, i, 1)
        self.data = data
        self.image = image

class circles():
    """
    A simple class for generating data given by class boundaries from two balls in a 2d plane.

    Attributes:
        data (NDArray): data points to sample from.
        image (NDArray): true class boundaries.
    """
    def __init__(self):
        size = 256
        image = np.zeros((size, size), dtype=np.uint8)

        # Draw a white circle
        center = (size // 2-80, size // 2+40)
        radius = 25
        cv2.circle(image, center, radius, color=1, thickness=-1)
        center = (size // 2+60, size // 2-40)
        radius = 50
        cv2.circle(image, center, radius, color=1, thickness=-1)
        N = 3000
        data = np.zeros((N, 3))
        for n in range(N):
            i, j = rd.randint(0, 256, 2)
            data[n] = (j, i, image[i, j])
        self.data = data
        self.image = image
