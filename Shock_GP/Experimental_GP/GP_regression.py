import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy as sp
import GPy
from sklearn.model_selection import train_test_split


########## select figure #############

fname = 'samples/figure_3a_sample.csv'

X = np.array([], dtype = float) ; Y = np.array([], dtype = float)

with open(fname, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X = np.append(X, float(row[0]))
            Y = np.append(Y, float(row[1]))

train_frac = 0.1 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 1 - train_frac)

X_train = np.array(X_train)[:, None]
Y_train = np.array(Y_train)[:, None] # GPy needs a 2D array

# # X = np.reshape(X, (len(X), 1)); Y = np.reshape(Y, (len(Y), 1))

# # print(X.shape, Y.shape)

kern = GPy.kern.Exponential(input_dim=1, variance=1.0, lengthscale=1.0,ARD=True)
# kern = GPy.kern.Brownian(input_dim=1)
# kern = GPy.kern.Linear(input_dim=1, ARD=True)
print('GP Regression')
m = GPy.models.GPRegression(X_train, Y_train, kern, noise_var=1.0)
print('Performing Restart')
m.optimize_restarts(num_restarts = 1)

X_p = np.linspace(1, 4.2, 150)[:, None] # inputs must be 2-dimensional

# default is to return total variance, including the noise
Y_p, V_p = m.predict(X_p)

Y_error = np.sqrt(V_p)

plt.plot(X_p, Y_p)
plt.fill_between(X_p.flatten(), Y_p.flatten() - 2*Y_error.flatten(), Y_p.flatten() + 2*Y_error.flatten(), alpha = 0.5)
plt.xlabel(r'Time (ns)')
plt.ylabel(r'$U_{shock}$ ($\mu$m/ns)')
plt.show()


f = open('GP_results/figure_3a_GP.csv', 'wt')

writer = csv.writer(f)

for i in range(len(X_p)):

    writer.writerow((float(X_p[i]), float(Y_p[i]), float(Y_error[i])))

f.close()







