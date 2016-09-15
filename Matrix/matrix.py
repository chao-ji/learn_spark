import numpy as np
from pyspark import SparkContext

sc = SparkContext("local", "pyspark")

A = np.random.rand(3, 3)
B = np.random.rand(3, 5)

rddA = sc.parallelize(list(enumerate(A.T)))
rddA = rddA.flatMapValues(lambda x: list(enumerate(x)))

rddB = sc.parallelize(list(enumerate(B)))
rddB = rddB.flatMapValues(lambda x: list(enumerate(x)))

C = rddA.join(rddB).map(lambda x: ((x[1][0][0], x[1][1][0]), x[1][0][1] * x[1][1][1])).reduceByKey(lambda x, y: x + y)
C = C.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(list).mapValues(lambda x: sorted(x, key=lambda y: y[0]))
C = C.mapValues(lambda x: zip(*x)[1])
C = np.array(C.sortByKey().map(lambda x: np.array(x[1])).collect())
