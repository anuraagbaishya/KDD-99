import pandas as pd
import numpy
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

print('Reading Data...')

data = pd.read_csv('~/Projects/KDD-99/kdd_10.csv')
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(categorical_features=[1,2])

x = data.iloc[1:50000,:-1].values
y = data.iloc[1:50000,-1].values

x = numpy.delete(x, 2, axis=1)

print('Performing Encodings')

x[:,1] = label_encoder.fit_transform(x[:,1])
x[:,2] = label_encoder.fit_transform(x[:,2])

x = one_hot_encoder.fit_transform(x).toarray()

error_arr, time_arr = [0]*10,[0]*10
error_csv, time_csv = [[0 for i in range(10)] for j in range(6)], [[0 for i in range(10)] for j in range(6)]

print('Performing Computations. This may take a while...')

for i in range(0,5):

	time_temp, error_temp = [0]*10, [0]*10
	X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75)	
	total_error, total_time = 0, 0
	for j in range(0,10):

		start_time = time.clock()
		knn = KNeighborsClassifier(n_neighbors=j+1, n_jobs=-1)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)

		error = zero_one_loss(y_test, y_pred)
		error_arr[j] += error
		error_csv[i][j] = float('{:.5f}'.format(error))
		exec_time = time.clock() - start_time
		print("Run",i+1, "N =",j+1,"Error =",error)
		print("Run",i+1, "N =",j+1,"Time =%.5f" % exec_time)
		time_arr[j] += round(exec_time, 5)
		time_csv[i][j] = round(exec_time, 5)

for i in range(0,10):
	error_csv[5][i] = error_arr[i]/5
	time_csv[5][i] = time_arr[i]/5

error_df = pd.DataFrame(error_csv)
time_df = pd.DataFrame(time_csv)

error_df.to_csv("~/Projects/KDD-99/ErrorData.csv")
time_df.to_csv("~/Projects/KDD-99/TimeData.csv")

print('Plotting Graphs')

objects = (1,2,3,4,5,6,7,8,9,10)
y_pos = numpy.arange(len(objects))

with PdfPages('KDD_Graphs.pdf') as pdf:

	plt.figure(1)
	plt.bar(y_pos, error_csv[5], align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel("Average Error")
	plt.xlabel("N")
	plt.title("Average Error vs N")
	pdf.savefig()

	plt.figure(2)
	plt.bar(y_pos, time_csv[5], align='center', alpha=0.5)
	plt.ylabel("Average Time")
	plt.xlabel("N")
	plt.title("Average Time vs N")
	pdf.savefig()

	plt.show()