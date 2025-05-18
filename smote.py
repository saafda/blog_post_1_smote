from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
from sklearn.neighbors import NearestNeighbors
import cv2
from numpy.typing import NDArray

class SMOTE:
    
    def __init__(self, data_samples):
        self.data_samples = data_samples
        
    def generate_data(self, ratio:float, N:int) -> tuple[NDArray, NDArray]:
        class_0 = self.data_samples[self.data_samples[:, 2] == 0]
        class_1 = self.data_samples[self.data_samples[:, 2] == 1]
        N0 = int((1-ratio)*N)
        N1 = int(ratio*N)
        
        class_0_sampled = class_0[np.random.randint(class_0.shape[0], size=N0), :]
        class_1_sampled = class_1[np.random.randint(class_1.shape[0], size=N1), :]
        
        return class_0_sampled, class_1_sampled
        
    
    def convex_combine(self, minority_class, k):
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
        for i in range(M):
            synthesized_point = self.convex_combine(minority_class, k)
            minority_class = np.vstack([minority_class, synthesized_point])
        return minority_class
    
    def plot(self, class_0, class_1, image, cmap, size):
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
