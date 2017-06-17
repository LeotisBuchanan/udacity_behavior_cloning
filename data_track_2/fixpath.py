import glob
import csv

data = "driving_log.csv"

search_string="""/Users/yardman/development/my_udacity_behavior_cloning_project/CarND-Behavioral-Cloning-P3/data_track_2/"""

with open(data) as csvfile:
    reader = csv.reader(csvfile)
    #next(reader)
    for line in reader:
        for idx in range(0, 3):
            line[idx] = line[idx].replace(search_string, "")
        
        print(",".join(line))

