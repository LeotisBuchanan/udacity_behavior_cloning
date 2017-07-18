import numpy as np
from sklearn.utils import shuffle
import matplotlib.image as mpimg

class DataGenerator:

    def __init__(self,image_height, image_width,channels):
        self.IMAGE_HEIGHT = image_height
        self.IMAGE_CHANNELS = channels
        self.IMAGE_WIDTH  = image_width
        self.CENTER = 0


    def readImage(self, path):
        image = mpimg.imread(path.strip())
        return image
    


    def validationGenerator(self, samples, batch_size=32):
        num_samples = len(samples)
        images = np.empty([batch_size, self.IMAGE_HEIGHT,
                           self.IMAGE_WIDTH, self.IMAGE_CHANNELS])
        steer_angles = np.empty(batch_size)
        while 1:  # Loop forever so the generator never terminates
            samples = shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                i = 0 
                for row in batch_samples:
                    image_path = row[self.CENTER]
                    image  = self.readImage(image_path)
                    angle =  row[3]
                    images[i] = image
                    steer_angles[i] = angle
                    i = i + 1
                    if i ==  batch_size:
                        break
                yield images, steer_angles

        
    def trainGenerator(self, samples, batch_size=32):
        num_samples = len(samples)
        images = np.empty([batch_size, self.IMAGE_HEIGHT,
                           self.IMAGE_WIDTH, self.IMAGE_CHANNELS])
        steer_angles = np.empty(batch_size)
        while 1:  # Loop forever so the generator never terminates
            samples = shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                i = 0 
                for row in batch_samples:
                    image_path = row[self.CENTER]
                    image = self.getAugmentedImage(self.CENTER,row)
                    angle = self.getAugmentedAngle(self.CENTER, row)
                    images[i] = image
                    steer_angles[i] = angle
                    i = i + 1
                    if i ==  batch_size:
                        break
                yield images, steer_angles

    def getSteeringAngle(self,row, index):
        return row[index]

                
    def getAugmentedAngle(self, camera_name, row):
        angle = self.getSteeringAngle(row, 3)
        return angle

    def _exit(self, message):
        import sys
        print(message)
        sys.exit(0)
        
    def getAugmentedImage(self,camera_name,row):

        path_prefix = "data/track1/"
        image_path = row[camera_name]
        image  = None
        path = path_prefix + image_path.strip()
        image = self.readImage(path)
        return image

                
    def getAugmentedSample(self, row):
        # always get the center image
        STEERING_IDX = 3
        steering_angle = getSteeringAngle(row, STEERING_IDX)
        image = self.getAugmentedImage(self.CENTER,
                                       row,
                                       self.augment_func)
                                       

        return image, steering_angle

    
if __name__ == '__main__':
    driving_log_file = "data/track1/driving_log_fixed.csv"
    samples = []
    with open(driving_log_file) as csv_file:
        samples = [line.split(",") for line in csv_file]

    print(len(samples))        
    d = DataGenerator(160,320,3)
    r = d.trainGenerator(samples)
    image, angle = next(r)
    print(image.shape)
    print(len(angle))
