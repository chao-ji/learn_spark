from pyspark import SparkContext

sc = SparkContext("local", "pyspark")
links = sc.textFile("links").map(lambda x: x.split("\t")).map(lambda x: (x[0], x[1:])).partitionBy(2).persist()
ranks = sc.textFile("ranks").map(lambda x: x.split("\t")).map(lambda x: (x[0], float(x[1])))

MAX = 20

for c in range(MAX): 
	contributions = links.join(ranks).flatMap(lambda x: [(i, x[1][1]/len(x[1][0])) for i in x[1][0]])
	ranks = contributions.reduceByKey(lambda x, y: x + y)
