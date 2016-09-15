import numpy as np
from pyspark import SparkContext
from scipy.spatial.distance import cosine

sc = SparkContext("local", "pyspark")

print "Generating list of 100 random numbers..."
data = np.random.randint(-100, 100, size=100).astype("float")
print "Mean: %f" % data.mean()
print "Stv: %f" % data.std(ddof=1)

rdd = sc.parallelize(data)
mean1 = rdd.map(lambda x: (x, 1.)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
mean1 = mean1[0] / mean1[1]
print "Compute mean using rdd.map().reduce(): %f" % mean1
mean2 = rdd.aggregate((0., 0.), lambda acc, val: (acc[0] + val, acc[1] + 1), lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]))
mean2 = mean2[0] / mean2[1]
print "Compute mean using rdd.aggregate(): %f" % mean2

def combine(nums):
	sumCount = [0., 0.]
	for num in nums:
		sumCount[0] += num
		sumCount[1] += 1
	return [sumCount]
mean3 = rdd.mapPartitions(combine, preservesPartitioning=True).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
mean3 = mean3[0] / mean3[1] 
print "Compute mean using rdd.mapPartitions().reduce(): %f" % mean3

mean = mean1
result = rdd.map(lambda x: ((x - mean)**2, 1.)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
std = np.sqrt(result[0] / (result[1] - 1))
print "Compute std using rdd.map.reduce(): %f" % std

print "\nGenerating two lists of 100 random numbers..."
v1 = np.random.rand(100)
v2 = np.random.rand(100)
rdd1 = sc.parallelize(v1)
rdd2 = sc.parallelize(v2)

print "Cosine similarity: %f" % (1 - cosine(v1, v2))
result = rdd1.zip(rdd2).aggregate((0., 0., 0.), lambda acc, val: (acc[0] + val[0] * val[1], acc[1] + val[0]**2, acc[2] + val[1]**2), lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1], acc1[2] + acc2[2]))

result = result[0] / (np.sqrt(result[1] * result[2]))
print "Compute cosine similarity: %f" % result
