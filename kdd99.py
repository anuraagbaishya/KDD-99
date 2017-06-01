import sys
import os

os.environ['SPARK_HOME'] ="/usr/local/share/spark"

sys.path.append("/usr/lib/spark/python")

try:
	from pyspark import SparkContext, SparkConf
	from pyspark.mllib.clustering import KMeans
	from pyspark.mllib.feature import StandardScaler
	print("Imported Spark Modules")
except:
	print("Spark Module Import Failure")

from collections import OrderedDict
from math import sqrt
from numpy import array

def parse_interaction(line): #parses input data

	line_split = line.split(",")
	clean_line_split = [line_split[0]] + line_split[4:-1]
	return (line_split[-1], array([float(x) for x in clean_line_split]))

def distance(a, b):#calculates euclidean distance b/w 2 numeric RDDs

	return (a.zip(b).map(lambda x:(x[0]-x[1])).map(lambda x: x*x).reduce(lambda a,b: a+b))

def dist_to_centroid(datum, clusters):

	cluster = clusters.predict(datum)
	centroid = clusters.centers[cluster]
	return sqrt(sum([x**2 for x in (centroid - datum)]))

def cluster_score(data, k):

	clusters = KMeans.train(data, k, maxIterations=10, runs=5, initializationMode="random")
	result = (k, clusters, data.map(lambda datum: dist_to_centroid(datum, clusters)).mean())
	print "Clustering score for k=%(k)d is %(score)f" % {"k": k, "score": result[2]}
	return result

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g KDDCup99.py max_k kddcup.data.file"
		sys.exit(1)

max_k = int(sys.argv[1])
data_file = sys.argv[2]

conf = SparkConf().setAppName("KDD99") 	
sc = SparkContext(conf=conf)

print "Loading Data"
raw_data = sc.textFile(data_file)

print "Counting Labels"
labels = raw_data.map(lambda line: line.strip().split(",")[-1])
label_counts = labels.countByValue()
sorted_labels = OrderedDict(label_counts.items(), key=lambda t:t[1], reverse=True)
for label, count in sorted_labels.items():
	print label, count

print "Parsing Dataset..."

parsed_data = raw_data.map(parse_interaction)
parsed_data_values = parsed_data.values().cache()

print "Standardizing Data..."

standardizer = StandardScaler(True, True)
standardizer_model = standardizer.fit(parsed_data_values)
standardized_data_values = standardizer_model.transform(parsed_data_values)

print "Calculating total in within cluster distance for different k values (10 to %(max_k)d):" % {"max_k": max_k}
scores = map(lambda k: cluster_score(standardized_data_values, k), range(10, max_k+1, 10))

min_k = min(scores, key=lambda x:x[2])[0]
print "Best k value is %(best_k)d" % {"best_k": min_k}

print "Obtaining clustering result sample for k=%(min_k)d..." % {"min_k": min_k}
best_model = min(scores, key=lambda x: x[2])[1]
cluster_assigment_sample = standardized_data_values.map(lambda datum: str(best_model.predict(datum))+","+",".join(map(str,datum))).sample(False,0.05)

print "Saving sample to file"
cluster_assigment_sample.saveAsTextFile("Sample_Standardized")
print "Done"