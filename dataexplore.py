import pandas as pd
import matplotlib.pyplot as plt


class DataExploration:

    def __init__(self, data_path):
        self.data_path = data_path
        pass

    def plotSteerAngleDistribution(self):
        """
        plot the histogram of the steering angles
        """
        df = pd.read_csv(self.data_path)
        print(df.columns)
        plt.figure()
        df[["steering"]].plot.hist(alpha=0.5)
        plt.title("before data augmentation")
        plt.show()


if __name__ == "__main__":
    path = "data/driving_log.csv"
    de = DataExploration(path)
    de.plotSteerAngleDistribution()
