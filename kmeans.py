#-------- import / constants --------#
from pyspark import SparkContext, SparkConf
from math import radians, cos, sin, asin, sqrt

MAX_ITER = 3

#-------- functions --------#
def myfilter(line):
    line = line.split(',')
    for i in [3,4,5,6]:
        if float(line[i]) == 0.0:
            return False
    return True

def haversine(xyA, xyB):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [xyA[0], xyA[1], xyB[0], xyB[1]])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

#def haversine(xyA, xyB):
#    return abs(xyA[0] - xyB[0]) + abs(xyA[1] - xyB[1])

# input: a tuple (x, y) (longtitude, landtitude) and a list of (id, (longtitude, langtitude)) of centers
def closestCenter(xy, centers):
    min_dist = haversine(xy, centers[0][1])
    min_center_id = centers[0][0]
    for i in range(1, len(centers)): # lengths should be 5
        dist = haversine(xy, centers[i][1])
        if dist < min_dist:
            min_dist = dist
            min_center_id = centers[i][0] # id
    return ( min_center_id, (xy[0], xy[1], 1) )

#-------- program --------#
appName = "Kmeans App"
conf = SparkConf().setAppName(appName)
sc = SparkContext(conf=conf)

HDFS = "hdfs://master:9000/"
rides = sc.textFile(HDFS + "yellow_tripdata_1m.csv") # pointer to the file
filter_rides = rides.filter(myfilter)  # remove incorrect data lines (contain 0.0 coordinate)
coords = filter_rides.map(lambda line: (float(line.split(",")[3]), float(line.split(",")[4]))) # map data to correct format
#TODO sc.cashe() for extra grade
centers = [(idx, tup) for idx, tup in enumerate(coords.take(5))] # list type, not rdd

#print("\n\n-------- {} --------\n{}\n\n".format('Beginning', centers))

for i in range(0, MAX_ITER):
    mapped = coords.map(lambda tup: closestCenter(tup, centers)) # emit (id of center,  )
    avg = mapped \
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])) \
            .mapValues(lambda v: (v[0]/v[2], v[1]/v[2])) # reduce by key (id of center) and  tranform only the value
    centers = avg.collect()
    #print("\n\n-------- ITER{} --------\n{}\n\n".format(str(i), centers))

for c in centers:
    print("center with id {} has coordinates {}".format(str(c[0]), str(c[1])))
