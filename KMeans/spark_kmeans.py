from pyspark import SparkContext
import numpy as np

sc = SparkContext('local', 'pyspark')
rawdata = sc.textFile('file').map(lambda x: x.split('\t')).map(lambda x: map(float, x))

N = 100
D = 2
MAX = 10

for c in range(MAX):

	print 'Iteration = %d' % c

	data = rawdata.collect()
	data = zip(*data)
	assnmt1 = dict(zip(data[0], data[1]))

	# Update centroid
	dataCentroidKey = rawdata.map(lambda x: [x[1], x + [1.0]])
	centroid = dataCentroidKey.reduceByKey(lambda x, y: np.array(x) + np.array(y))
	centroid = centroid.mapValues(lambda x: list(x[2 : 2 + D] / x[2 + D]))

	# Assign data
	dataIdKey = rawdata.map(lambda x: [x[0], x])
	centroid = centroid.map(lambda x: [x[0]] + x[1])
	cartesian = centroid.flatMap(lambda x: [[float(i), x] for i in range(N)])

	grouped = dataIdKey.cogroup(cartesian)
	grouped = grouped.mapValues(lambda x: [list(x[0])[0], list(x[1])])
	grouped = grouped.mapValues(lambda x: [[x[1][i][0], np.linalg.norm(np.array(x[0][2:]) - np.array(x[1][i][1:]))] + x[0][2:] for i in range(len(x[1]))])

	grouped = grouped.mapValues(lambda x: min(x, key = lambda x: x[1]))

	rawdata = grouped.map(lambda x: [x[0], x[1][0]] + x[1][2:])

	data = rawdata.collect()
	data = zip(*data)
	assnmt2 = dict(zip(data[0], data[1]))

	changed = 0
	for k in assnmt1.keys():
		if assnmt1[k] != assnmt2[k]:
			changed += 1

	print 'Num of assignment updates = %d' % changed
	if changed == 0:
		print 'No updates necessary...\nFinished!'
		break
