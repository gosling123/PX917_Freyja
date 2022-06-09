#!/usr/bin/python3

import fileinput
import sys
import numpy as np
import GPy
from sklearn.model_selection import train_test_split
import freyja_scripts as scripts

#************************************************************************
#> replace_line
##
## Function rewrite line in input.deck via python 
##
##@param line_in : original line in input.deck 
##       line_out : repacement of line_in in input.deck
#************************************************************************
#

def replace_line(line_in, line_out, fname='Data/input.txt'):
  finput = fileinput.input(fname, inplace=1)
  for i, line in enumerate(finput):
    sys.stdout.write(line.replace(line_in, line_out))
  finput.close()
  

#************************************************************************
#> write_laser
##
## Function to create laser namelist in input.deck
##
##@param ra_frac : resonant absorption fraction (range(0, 1))
##       coupling : coupling of laser to ablator (range(0, 1))
##       lpi_loss : fraction oflaser enery lost due to lpi (range(0, 1))
##       laser_file : temporalpower profile of laser as csv file 
##       lpi_dens : density fraction at which lpi loss is taken (set as 0.25) (range(0, 1))
##       lambda_ : wavelength of laser (set to 351nm to model OMEGA laser)
##       fname : filename to write into (set as input.deck in Data directory)
#************************************************************************
#

def write_laser(ra_frac, coupling, lpi_loss, laser_file, \
    lpi_dens=0.25, lambda_=351.0e-9,fname='Data/input.txt'):
  
  with open(fname, 'a') as deck:
    deck.write('\n&LASER_NML,\n')
    deck.write(f'  lambda = {lambda_},\n')
    deck.write(f'  lpi_nc_frac = {lpi_dens},\n')
    deck.write(f'  lpi_loss_frac = {lpi_loss},\n')
    deck.write(f'  ra_fraction = {ra_frac},\n')
    deck.write(f'  coupling = {coupling},\n')
    deck.write(f'  power_file = \'{laser_file}\',\n')
    deck.write('/\n')


#************************************************************************
#> log_L
##
## estimates log-likilhood of shock profile
##
##@param res : freyja result
##       Y_exp : experimental result
##       Y_err : experimental error
#  
#************************************************************************
#

def log_L(res, Y_exp, Y_err):

  N = len(Y_exp)

  std = np.sqrt(np.average(Y_err**2))

  log = - np.sum((res - Y_exp)**2) / (2.0 * std)
  log -= N * np.log(std)
  log -= (N / 2.0) * np.log(2.0 * np.pi)

  return log



#************************************************************************
#> loss_func
##
## estimates loss value between sim and exp values
##
##@param res : freyja result
##       exp : experimental result
##  
#  
#************************************************************************
#

def loss_func(res, exp):

  loss = (res - exp)

  return loss


#************************************************************************
#> GP_regression
##
## Function to perform GP_regrssion on shock velocity
##
##@param X : input x sample data
##       Y : input y sample data
##       start : start x point for predict
##       end : end x point for predict
##       npoints : number of linearspace points in predict
##       kern : choice of kernel 
##       train_frac : controls amount of data to train GP
##       restart : number of restrt optimizations performed
#************************************************************************
#

def GP_regression(X, Y, start, end, kern = 'Exponential',train_frac = 0.99, restart = 1, npoints = 150):

  # test/train split
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 1 - train_frac)

  # Gpy needs 2D array
  X_train = np.array(X_train)[:, None]
  Y_train = np.array(Y_train)[:, None] 

  # kernel choice
  if (kern == 'Exponential'):
    k = GPy.kern.Exponential(input_dim=1, variance=1.0, lengthscale=1.0,ARD=True)
  if (kern == 'Brownian'):
    k = GPy.kern.Brownian(input_dim=1)
  if (kern == 'Linear'):
    k = GPy.kern.Linear(input_dim=1, ARD=True)

  # GP regression
  m = GPy.models.GPRegression(X_train, Y_train, k, noise_var=1.0)

  # optimize
  m.optimize_restarts(num_restarts = restart, verbose=False)

  # predict
  X_p = np.linspace(start, end, npoints)[:, None]

  # default is to return total variance, including the noise
  Y_p, V_p = m.predict(X_p)
  Y_error = np.sqrt(V_p)

  return X_p.reshape(-1), Y_p.reshape(-1), Y_error.reshape(-1)



#************************************************************************
#> target_setup
##
## Function to create file for initial conditions in freyja
##
##@param th : hydrocarbon thickness
##       nr : number of grid resoloutions
#************************************************************************
#

def target_setup(th,nr):  
  data = {}

  data['Time'] = 0.0
  data['Nr'] = nr
  data['Geometry'] = 0 # Planar geometry

  # Set up grid
  grid_vacl = np.linspace(-th*1000,0.0,3)
  grid_ablator = np.linspace(0.0, th, nr + 1)
  grid_full = np.append(grid_vacl,grid_ablator[1:])
  grid_vacuum = np.linspace(th, th*1000, 3)
  grid_full = np.append(grid_full, grid_vacuum[1:])

  rho = np.zeros(nr+4)
  vel = np.zeros(nr+5)
  material_number = np.zeros(nr+4,dtype=int)
  temp_ele = np.zeros(nr+4)
  temp_ion = np.zeros(nr+4)

  temp_ion[2:-2] = 273
  temp_ele[2:-2] = 273

  data['Velocity'] = vel
  data['Ion_Temperature'] = temp_ion
  data['Electron_Temperature'] = temp_ele

  # Foil plus vacuum either side
  material_number[:2] = 2
  material_number[2:-2] = 1
  material_number[-2:] = 2

  rho[2:-2] = 1050.0

  data['Density'] = rho
  data['Material_Number'] = material_number

  data['Grid'] = grid_full
  scripts.write.write_file('Data/freyja_0000.dat', data)