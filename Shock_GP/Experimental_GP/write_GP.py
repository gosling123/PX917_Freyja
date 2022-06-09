import csv 
import pickle
import numpy as np

#### Figure 2 ####

fname = 'GP_results/figure_2_GP.csv'

X_2 = np.array([]); Y_2 = np.array([]); err_2 = np.array([])
with open(fname, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X_2 = np.append(X_2, float(row[0]))
            Y_2 = np.append(Y_2, float(row[1]))
            err_2 = np.append(err_2, float(row[2]))



#### Figure 3a ####

fname = 'GP_results/figure_3a_GP.csv'

X_3a = np.array([]); Y_3a = np.array([]); err_3a = np.array([])
with open(fname, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X_3a = np.append(X_3a, float(row[0]))
            Y_3a = np.append(Y_3a, float(row[1]))
            err_3a = np.append(err_3a, float(row[2]))



#### Figure 3b ####

fname = 'GP_results/figure_3b_GP.csv'

X_3b = np.array([]); Y_3b = np.array([]); err_3b = np.array([])
with open(fname, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X_3b = np.append(X_3b, float(row[0]))
            Y_3b = np.append(Y_3b, float(row[1]))
            err_3b = np.append(err_3b, float(row[2]))



#### Figure 4 ####

fname = 'GP_results/figure_4_GP.csv'

X_4 = np.array([]); Y_4 = np.array([]); err_4 = np.array([])
with open(fname, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X_4 = np.append(X_4, float(row[0]))
            Y_4 = np.append(Y_4, float(row[1]))
            err_4 = np.append(err_4, float(row[2]))




with open('experimental_GP_data.pickle', 'wb') as file:

    pickle.dump([\
                X_3a, Y_3a, err_3a, \
                X_3b, Y_3b, err_3b,\
                X_4, Y_4, err_4,\
                X_2, Y_2, err_2\
                ], file)