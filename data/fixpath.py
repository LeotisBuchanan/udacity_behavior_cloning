import glob
import csv

data = "driving_log.csv"

with open(data) as csvfile:
    reader = csv.reader(csvfile)
    #next(reader)
    for line in reader:
        line[0] = line[0].replace('/Users/yardman/data/', "")
        line[1] = line[1].replace('/Users/yardman/data/', "")
        line[2] = line[2].replace('/Users/yardman/data/', "")



        line[0] = line[0].replace('/Users/yardman/development/data2/', "")
        line[1] = line[1].replace('/Users/yardman/development/data2/', "")
        line[2] = line[2].replace('/Users/yardman/development/data2/', "")

        line[0] = line[0].replace('/Users/yardman/development/data3/', "")
        line[1] = line[1].replace('/Users/yardman/development/data3/', "")
        line[2] = line[2].replace('/Users/yardman/development/data3/', "")


        line[0] = line[0].replace('/Users/yardman/development/data4/', "")
        line[1] = line[1].replace('/Users/yardman/development/data4/', "")
        line[2] = line[2].replace('/Users/yardman/development/data4/', "")

        
        print(",".join(line))

