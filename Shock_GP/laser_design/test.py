import numpy as np
import matplotlib.pyplot as plt
import csv 

import glob

nano = 1e-9
exa = 1e18

fnames = glob.glob('setup/*.csv')

# data = {}

for f in fnames:

    with open(f, 'r') as file:
        reader = csv.reader(file)
        time = np.array([], dtype = float) ; power = np.array([], dtype = float)
        for row in reader:
            time = np.append(time, float(row[0]) / nano)
            power = np.append(power, float(row[1]) / exa)

    if (f == 'setup/laser_profile_3.csv'):
        plt.plot(time, power, label = f, linewidth = 3, alpha = 0.7)
    if (f == 'setup/A_laser_profile2.csv'):
        plt.plot(time, power, label = f, linewidth = 2)

    
    # plt.plot(time, power, label = f, linestyle = '--')
plt.xlabel('Time (ns)', fontsize = 16)
plt.ylabel('Laser Power (EW)', fontsize = 16)
plt.legend(fontsize = 16)
plt.show()
