import numpy as np
import matplotlib.pyplot as plt
import csv 

nano = 1e-9
pico = 1e-12

rad_to_deg = np.pi/180

res = 300

#### form laser profiles and targets from paper 

FWHM = 100 * pico

def gaussian(FWHM, peak, t0, t):

    sigma = (0.5*FWHM) / np.sqrt(2*np.log(2)) # sigma for gaussin from FWHM

    f = peak * np.exp(-(t-t0)**2 / (2*sigma**2))

    return f


area = 1

P1_pulse = 4.9e14 * np.cos(0 * rad_to_deg)#W/cm2
P1_pulse *= area*1e4 # covert W/cm2 to W/m2 and times by area to get power

P2_pulse_1 = 2.2e14 * (np.cos(23.2*rad_to_deg)) #W/cm2
P2_pulse_2 = 4.7e14 * (np.cos(47.8*rad_to_deg)) #W/cm2

P2_pulse_1 *= area*1e4
P2_pulse_2 *= area*1e4


## 2 picket profile
t_peak = 0.15 # ns # time of first peak
t_delay = 1.5 #ns # time delay for second peak relative to first
t_buff = 0.1 # ns # buffer time for linspace

t1 = np.linspace(0 , (2*t_peak + t_buff) * nano, res) 
t2 = np.linspace((2*t_peak + t_buff) * nano, (t_delay + 2*t_peak + t_buff)  * nano, res + 1)

pick1 = gaussian(FWHM, P2_pulse_1, t_peak * nano, t1)
pick2 = gaussian(FWHM, P2_pulse_2,  (t_peak + t_delay) * nano , t2)

time = np.append(t1, t2[1:])
laser_power = np.zeros(len(time))

laser_power = np.append(pick1, pick2[1:])

laser_power_2 = laser_power

# print(laser_power)

plt.plot(time*1e9, laser_power*1e-15)
plt.ylabel(r'Laser Power (PW)')
plt.xlabel(r'Time (ns)')
plt.show()

for i in range(len(laser_power)):

    if laser_power[i] < 1e-6:
        laser_power[i] = 0
        
f = open('setup/laser_profile_2.csv', 'wt')

writer = csv.writer(f)

for i in range(len(time)):

    writer.writerow((time[i], laser_power[i]))

f.close()

## 1 picket profile

laser_power = gaussian(FWHM, P1_pulse, t_peak * nano, time)

laser_power_1 = laser_power


plt.plot(time*1e9, laser_power*1e-15)
plt.ylabel(r'Laser Power (PW)')
plt.xlabel(r'Time (ns)')
plt.show()


for i in range(len(laser_power)):

    if laser_power[i] < 1e-6:
        laser_power[i] = 0

f = open('setup/laser_profile_1.csv', 'wt')

writer = csv.writer(f)

for i in range(len(time)):

    writer.writerow((float(time[i]), float(laser_power[i])))

f.close()

# double picket profile for figure 4

FWHM = 600 * pico

## 2 picket profile
t_peak = 0.6 # ns # time of first peak
t_delay = 1.7 #ns # time delay for second peak relative to first
t_buff = 0.05 #ns

t1 = np.linspace(0 , (2*t_peak + t_buff) * nano, res) 
t2 = np.linspace((2*t_peak + t_buff) * nano, (t_delay + 2*t_peak + t_buff)  * nano, res + 1)

P3_pulse_1 = 4.4e13 * (np.cos(23.2*rad_to_deg)) #W/cm2
P3_pulse_2 = 2.0e14 * (np.cos(23.2*rad_to_deg)) #W/cm2

# convert W/cm2 to W/m2 and then from Intensity to power
P3_pulse_1 *= area*1e4 
P3_pulse_2 *= area*1e4

pick1 = gaussian(FWHM, P3_pulse_1, t_peak * nano, t1)
pick2 = gaussian(FWHM, P3_pulse_2,  (t_peak + t_delay) * nano , t2)

time = np.append(t1, t2[1:])
laser_power = np.zeros(len(time))

laser_power = np.append(pick1, pick2[1:])

laser_power_3 = laser_power

# print(laser_power)


plt.plot(time*1e9, laser_power*1e-15)
plt.ylabel(r'Laser Power (PW)')
plt.xlabel(r'Time (ns)')
plt.show()


for i in range(len(laser_power)):

    if laser_power[i] < 1e-6:
        laser_power[i] = 0
        
f = open('setup/laser_profile_3.csv', 'wt')

writer = csv.writer(f)

for i in range(len(time)):

    writer.writerow((time[i], laser_power[i]))

f.close()



plt.plot(time*1e9, laser_power_1*1e-15, label = r'Profile 1')
plt.plot(time*1e9, laser_power_2*1e-15, label = r'Profile 2')
plt.plot(time*1e9, laser_power_3*1e-15, label = r'Profile 3')
plt.ylabel(r'Laser Power (PW)', fontsize = 15)
plt.xlabel(r'Time (ns)', fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

