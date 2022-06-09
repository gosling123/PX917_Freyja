#!/usr/bin/python3

import numpy as np
from util import *
import pickle
import matplotlib.pyplot as plt
import os



  #************************************************************************
  #> shk_vel
  ##
  ## function that finds the log liklihood of the freyja and experimental
  ## shock velocity gaussian process data
  ##
  ##@param params : array of freyja parametrs (flux limiters and laser constants)
  ##       profile : number corresponding to laser file to sim
  ##       nr : grid resoloutions of target foil
  ##       nfiles : number of file snapshots from freyja
  ##       den : desnity of CH foil
  ##       returns : number of files to take data from
  ##       t_tot : sim time
  ##       output : output freyja status in terminal
  ##       plot : logic for if to plot freyja and exp
  ##       plot_GP : logic for if to plot GP regression of freyja data in exp range
  ##       save_plot : logic if to save plot or not
  ##       liklihood : return log_liklihood 
  ##       shocktime : return time of rebound shock
  #************************************************************************
  #

def shk_vel(params, profile, nr = 2000, nfiles = 400, den = 1100, \
            returns = 100, t_tot = 6.0e-9, output = False, \
            plot = False, save_plot = False, plot_GP = False,\
            likelihood = False, shock_time = False):


  # print('READ IN PARAMS')
  flux_e, flux_i, ra_frac, coupling, lpi_loss = params

  # flux limiters
  flux_e = min(flux_e, 1.0); flux_i = min(flux_i, 1.0)
  # laser parameters
  ra_frac = min(ra_frac, 1.0); 
  lpi_loss = min(lpi_loss, 1.0)
  coupling = min(coupling,1.0)

  # sim times amd output dumps

  t_out = t_tot / nfiles
  dstep = np.round(t_tot / returns, 11)
  vel = np.zeros(returns)
  t_tot = t_tot + t_out # Add a step for derivative calculation
  file_gap = int(nfiles / returns)
  nfiles = nfiles + 1

  freyja_time = np.linspace(dstep,t_tot,returns) * 1e9

  # print('FILE GAP =', file_gap)

  dir = os.getenv('SHOCKGP')

  try:
    os.mkdir('Data')
  except:
    os.system('rm Data/*')
  
  os.system(f'cp {dir}/setup/laser_profile_{profile}.csv Data')
  os.system(f'cp {dir}/setup/input.txt Data/input.txt')

  # Amend input file
  replace_line('t_end = 5.0e-9,', f't_end = {t_tot},')
  replace_line('dt_snapshot = 1.0e-11,', f'dt_snapshot = {t_out},')
  replace_line('flux_limiter_ele = 0.06,', f'flux_limiter_ele = {flux_e},')
  replace_line('flux_limiter_ion = 0.1,', f'flux_limiter_ele = {flux_i},')

  fname = 'laser_profile_'+str(profile)+'.csv'
  write_laser(ra_frac, coupling, lpi_loss, fname)

  # Setup initial conditions and run Freyja
  th = 125e-6
  target_setup(th,nr)

  # print('RUNNING FREYJA')
  # run freyja (output or not)
  if output:
    os.system(f'{dir}/bin/freyja')
  else:
    os.system(f'{dir}/bin/freyja > run.log')

  #Loop through output files and calculate shock velocity at specified interval

  # print('GETTING SHOCK VELOCITY')
  count = 0 # loop counter - 
  shockf = nr + 1 #initialse shock front before foil
  
  for i in range(1, nfiles + 1):
    if ((i+1)%file_gap == 0):

      fname = 'Data/freyja_' + str(i).zfill(4) + '.dat'

      try:
        data = scripts.read_file(fname)
      except:
        print(f'File not found: {fname}. Time: {i*t_out}. Params: {params}')
        break

      shockf = np.argmax(data['Density'] >  den)
      denf = data['Density'][shockf-1:shockf+1]
      posf = data['Grid_Cell_Centre'][shockf-1:shockf+1]

      try:
        dis0 = np.interp(den, denf, posf)
      except:
        if (shockf < 2):
          break
        print('Interpolation array empty')
        dis0 = dfile['Grid_Cell_Centre'][shockf]

        
    elif (((i-1)%file_gap == 0) and ((i-1) != 0)):
      fname = 'Data/freyja_' + str(i).zfill(4) + '.dat'
      try:
        dfile = scripts.read_file(fname)
      except:
        print(f'File not found: {fname}. Time: {i*t_out}. Params: {params}')
        break
      # Get shock front location
      shockf = np.argmax(dfile['Density'] > den)
      denf = dfile['Density'][shockf-1:shockf+1]
      posf = dfile['Grid_Cell_Centre'][shockf-1:shockf+1]
      try:
        dis1 = np.interp(den,denf,posf)
      except:
        if (shockf < 2): # breaks into vacuum (breaks out of the foil)
          break
        print('Interpolation array empty')
        dis1 = dfile['Grid_Cell_Centre'][shockf]
      
      # Central differences second order first derivative on shock front
      vel[count] = (dis0-dis1)*1e6/(2*t_out*1e9)
      count += 1

  # Import experimental datasets
  X = {}; Y = {}; err = {}
  with open(f'{dir}/Experimental_GP/experimental_GP_data.pickle', 'rb') as file:
    X[1], Y[1], err[1], X[2], Y[2], err[2], X[3], Y[3], err[3], X[4], Y[4], err[4] = pickle.load(file)

  # select experimental results for set profile
  time_exp = np.array(X[profile]) ; vel_exp = np.array(Y[profile]) ; #err_exp = np.array(err[profile])

  err_exp = 0.05 * vel_exp

  start = time_exp[0] ; end = time_exp[-1] ; points = len(time_exp)

  # print('GP REGRESSION')
  time_GP, vel_GP, err_GP = GP_regression(freyja_time, vel, start, end, kern = 'Exponential', \
                                          train_frac = 0.99, restart = 1, npoints = points)

  if (shock_time):
    
    indices = []

    for i in range(len(freyja_time)):

      if (2.0 <= freyja_time[i] <= 4.0):
        indices.append(i)

    vel_cut = vel[indices]
    dv = np.diff(vel_cut)

    discon = np.argmax(dv)

    discon += indices[0]

    t_shock = (freyja_time[discon] + freyja_time[discon+1]) / 2.0

    dv_exp = np.diff(vel_exp)
    discon_exp = np.argmax(dv_exp)

    t_shock_exp = (time_exp[discon_exp] + time_exp[discon_exp+1]) / 2.0

  # Optionally plot

  if (plot_GP):
    plt.scatter(freyja_time,vel,label=r'Freyja')
    plt.plot(time_GP, vel_GP, label = r'GP', color = 'orange')
    plt.fill_between(time_exp, vel_GP-err_GP, vel_GP+err_GP, alpha = 0.5, label = r'GP Error', color = 'orange')
    plt.title(f'$f_e =$ {flux_e:0.4f}, $f_i =$ {flux_i:0.4f}, $RA =$ {ra_frac:0.3f}, $C =$ {coupling:0.4f}, $LPI =$ {lpi_loss:0.4f}')
    plt.ylabel(r'$U_{Shock}$ [$\mu m/ns$]',fontsize=14)
    plt.xlabel(r'Time [ns]',fontsize=14)
    plt.legend(loc='best',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if (save_plot):
      plt.savefig('vel_GP'+str(profile) + '.png')
    if (shock_time):
      plt.axvline(x  = t_shock, color = 'purple', linewidth = 3, linestyle = '--' )
    plt.show()


  if (plot):
    plt.plot(freyja_time,vel,label=r'Freyja')
    plt.plot(time_exp, vel_exp, label = r'Experimental', color = 'orange')
    plt.fill_between(time_exp, vel_exp-err_exp, vel_exp+err_exp, alpha = 0.5, label = r'Exp Error', color = 'orange')
    plt.title(f'$f_e =$ {flux_e:0.4f}, $f_i =$ {flux_i:0.4f}, $RA =$ {ra_frac:0.3f}, $C =$ {coupling:0.4f}, $LPI =$ {lpi_loss:0.4f}')
    plt.ylabel(r'$U_{Shock}$ [$\mu m/ns$]',fontsize=14)
    plt.xlabel(r'Time [ns]',fontsize=14)
    plt.legend(loc='best',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if (save_plot):
      plt.savefig('vel'+str(profile) + '.png')
    if (shock_time):
      plt.axvline(x  = t_shock, color = 'purple', linewidth = 3, linestyle = '--' )
      plt.axvline(x  = t_shock_exp, color = 'green', linewidth = 3, linestyle = '--' )
    plt.show()


  if (shock_time):
    residual = loss_func(t_shock, t_shock_exp)
    return np.array([residual])

  likelihood = log_L(vel_GP, vel_exp, err_exp)

  if (likelihood):
    # return in this format for GPY later
    return np.array([likelihood]) 
                  
          
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   