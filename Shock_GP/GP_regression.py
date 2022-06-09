from shock import *
import numpy as np
import GPy
from sklearn.model_selection import train_test_split
import seaborn as sns
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["figure.figsize"] = (20, 10)


class LHC:

    """Class implementing LHC samples"""

    def __init__(self, sample_space, bounds, calib_samp, GP_samp):

        self.sample_space = sample_space 
        self.bounds = bounds 
        self.calib_samp = calib_samp
        self.GP_samp = GP_samp

        LH_space = LHS(xlimits=self.bounds)
        
        self.calib_samp_space = LH_space(self.calib_samp)
        self.GP_samp_space = LH_space(self.GP_samp)


    def calibrate_samples(self, new_calib = False, best_data = False):


        if (new_calib):
            
            profiles = [1,2,3]
            shock_time_profiles = [2,3]

            fnames = ['GP_calib/Log_L_samples_P1.npy',\
                      'GP_calib/Log_L_samples_P2.npy',\
                      'GP_calib/Log_L_samples_P3.npy']

            shock_fnames = ['GP_calib/shock_res_samples_P2.npy',\
                            'GP_calib/shock_res_samples_P3.npy']

            LH_samples = self.calib_samp_space

            with open('GP_calib/LH_freyja_samples.npy', 'wb') as f:
                    np.save(f, LH_samples)

            for profile, fname in tqdm(zip(profiles, fnames), total=len(profiles), miniters=0, leave=False):

                print('Calibrating Proifle' + str(profile))

                log_L_results = np.array([])

                for sample in tqdm(LH_samples, total=len(LH_samples), miniters=0, leave=False):

                    log_L = shk_vel(sample, profile = profile, likelihood=True)

                    log_L_results = np.append(log_L_results, log_L)

                with open(fname, 'wb') as f:
                    np.save(f, log_L_results)

            
            for shock_profile, shock_fname in tqdm(zip(shock_time_profiles, shock_fnames), total = len(shock_time_profiles), miniters=0, leave = False):
                
                print('Calibrating Shock Time Proifle' + str(shock_profile))
                
                shock_time_residual = np.array([])

                for sample in tqdm(LH_samples, total=len(LH_samples), miniters=0, leave=False):

                    residual = shk_vel(sample, profile = shock_profile, shock_time=True)

                    shock_time_residual = np.append(shock_time_residual, residual)

                with open(shock_fname, 'wb') as f:
                    np.save(f, shock_time_residual)


                
        if (best_data):
            self.X_data = np.load('GP_calib_best/LH_freyja_samples.npy')
            self.Y_P1_data = np.load('GP_calib_best/Log_L_samples_P1.npy')
            self.Y_P2_data = np.load('GP_calib_best/Log_L_samples_P2.npy')
            self.Y_P3_data = np.load('GP_calib_best/Log_L_samples_P3.npy')
            self.Y_ST_P2_data = np.load('GP_calib_best/shock_res_samples_P2.npy')
            self.Y_ST_P3_data = np.load('GP_calib_best/shock_res_samples_P3.npy')



        else:
            self.X_data = np.load('GP_calib/LH_freyja_samples.npy')
            self.Y_P1_data = np.load('GP_calib/Log_L_samples_P1.npy')
            self.Y_P2_data = np.load('GP_calib/Log_L_samples_P2.npy')
            self.Y_P3_data = np.load('GP_calib/Log_L_samples_P3.npy')
            self.Y_ST_P2_data = np.load('GP_calib/shock_res_samples_P2.npy')
            self.Y_ST_P3_data = np.load('GP_calib/shock_res_samples_P3.npy')


    def plot_prior_LHC(self):

        LH_samples = self.X_data

        labels = ['Flux Limiter_e', 'Flux Limiter_i', 'RA','Coupling','LPI Loss']
        fig, axs = plt.subplots(2, 3,constrained_layout=True)

        for j in range(5):
            ax = axs.flat[j]
    
            x = np.linspace(0.99*self.bounds[j][0], 1.01*self.bounds[j][1], 1000)
    
            sns.histplot(LH_samples[:,j], stat='density', kde=True, color='C1', ax=ax, label = 'LH Samples')

            ax.plot(x, self.sample_space[j].pdf(x), ls='-', label = 'Uniform')
    
            ax.set_title(labels[j])
    

        axs.flat[-1].set_visible(False)
        plt.gcf().set_size_inches(20,10)



    
class GP_reg:

    """Class implementing a Gaussian Process"""

    def __init__(self, kernel, samples, output, train_frac, restarts):

        self.kernel = kernel
        self.samples = samples
        self.output = output
        self.train_frac = train_frac
        self.restarts = restarts
        
        dim = samples.shape[1]

        self.dim = dim

    def test_train(self):

        X_train, X_test, Y_train, Y_test = train_test_split(self.samples, self.output, test_size = 1 - self.train_frac)

        self.X_train = X_train
        self.X_test = X_test

        self.Y_train = Y_train[:, None]
        self.Y_test = Y_test[:, None]

    def regression(self):
        
        kern_string = 'GPy.kern.'+self.kernel+'(input_dim=self.dim,variance=1.,lengthscale=1.,ARD=True)'
        kern = eval(kern_string)

        m = GPy.models.GPRegression(self.X_train, self.Y_train, kern)

        m.optimize_restarts(self.restarts, verbose=False)

        if (self.kernel == 'RBF'):
            len_sacles = 'm.rbf.lengthscale'

        if (self.kernel == 'Matern32'):
            len_sacles = 'm.Mat32.lengthscale'

        if (self.kernel == 'Matern52'):
            len_sacles = 'm.Mat52.lengthscale'
     
        else:
            len_sacles = 'm.'+self.kernel+'.lengthscale'

        l = eval(len_sacles)

        self.len_scales = l

        self.dQ_dx = 1.0 / self.len_scales

        self.m = m

    def GP_predict(self, X):

        Y_p, V_p = self.m.predict(X)

        return Y_p, V_p

    def test_train_plot(self, title, likelihood = False, shock_time = False):

        target_value = self.output

        Y_ptrain, V_ptrain = self.m.predict(self.X_train)
        Y_ptest, V_ptest = self.m.predict(self.X_test)

        rmse_train = np.sqrt(np.mean((Y_ptrain[:,0] - self.Y_train[:,0])**2))
        rmse_test = np.sqrt(np.mean((Y_ptest[:,0] - self.Y_test[:,0])**2))


        ### DISTRIBUTION OF QUANTITY OF INTEREST
        fig = plt.figure()
        sns.kdeplot(target_value.reshape(len(target_value)), label = 'Target Data', linestyle='dashdot', linewidth = 5, color = 'black')
        sns.kdeplot(Y_ptrain.reshape(len(Y_ptrain)), label=f'Train, RMSE = '+str(rmse_train), color = 'blue')
        sns.kdeplot(Y_ptest.reshape(len(Y_ptest)), label=f'Test, RMSE = '+str(rmse_test), color = 'orange')
        plt.legend()
        if (likelihood):
            plt.xlabel(r'$\log \mathbb{P} \left(\mathbf{y} | \mathbf{x}, \mathbf{\theta}, \sigma_{{Exp}}\right)$', fontsize = 20)
        if (shock_time):
            plt.xlabel(r'$\mathbb{R} \left( t, t_{{Exp} }\right)$', fontsize = 20)
        plt.title(title)
        plt.ylabel(r'Density', fontsize = 20)
        plt.show()

        ### STANDARD DEVIATION 
        S_ptrain = np.sqrt(V_ptrain)
        S_ptest = np.sqrt(V_ptest)

        #### ax1/2 --- train
        #### ax3/4 ---- test

        ### PLOT CORRELATION BETWEEN VALUES AND ERRORS
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.scatter(self.Y_train, Y_ptrain, label=f'Train', color = 'blue', alpha = 0.6)
        ax1.plot([target_value.min(), target_value.max()], [target_value.min(), target_value.max()], 'k:', label = 'Target')
        ax1.set_xlabel('True Value', fontsize = 20)
        ax1.set_ylabel('Predicted Value', fontsize = 20)
        ax1.legend()

        ax2.plot(abs(Y_ptrain[:,0] - self.Y_train[:, 0]), S_ptrain[:, 0], 'o', label='Train', color = 'blue')
        ax2.set_xlabel('True Error', fontsize = 20)
        ax2.set_ylabel('Predicted Error', fontsize = 20)
        # ax2.plot([0, S_ptrain.max()], [0, S_ptrain.max()], 'k:', label = 'Target')
        ax2.legend()

        ax3.scatter(self.Y_test, Y_ptest, label=f'Test', alpha=0.6, color = 'orange')
        ax3.plot([target_value.min(), target_value.max()], [target_value.min(), target_value.max()], 'k:', label = 'Target')
        ax3.set_xlabel('True Value', fontsize = 20)
        ax3.set_ylabel('Predicted Value', fontsize = 20)
        ax3.legend()


        ax4.plot(abs(Y_ptest[:,0] - self.Y_test[:,0]), S_ptest, 'o', label='Test', color='orange')
        ax4.set_xlabel('True Error', fontsize = 20)
        ax4.set_ylabel('Predicted Error', fontsize = 20)
        # ax4.plot([0, S_ptest.max()], [0, S_ptest.max()], 'k:', label = 'Target')
        ax4.legend()




        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.4)
        plt.gcf().set_size_inches(20,10)

        plt.show()

    def GP_sensitivities(self, x0, plot = True, likelihood = False, shock_time = False):

        if (plot):
            labels = [r'Flux Limiter Electron', r'Flux Limiter Ion', r'RA',r'Coupling',r'LPI Loss']

            plt.bar(labels, x0 * self.dQ_dx)

            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.ylabel(r'Scaled First Order Sensitivity $\left(\bar{\theta_{i}}\,\,\frac{d Q(\theta)}{d\theta_{i}}\right)$', fontsize = 20)
            if (likelihood):
                plt.title(r'$Q(\theta) = \log \mathbb{P} \left(\mathbf{y} | \mathbf{x}, \mathbf{\theta}, \sigma_{{Exp}}\right)$', fontsize = 25)
            if (shock_time):
                plt.title(r'$Q(\theta) = \mathbb{R} \left( t, t_{{Exp} }\right)$', fontsize = 25)
            plt.show()
        else:
            return x0 * self.dQ_dx









def first_order_sensitivities(Q, x0, profile, eps=1e-6, plot = True, scaled = True, likelihood = False, shock_time = False):
    
    Q0 = shk_vel(x0, profile, likelihood=likelihood, shock_time=shock_time)

    dQ_dx = np.zeros(len(x0))
    for i, xi in enumerate(x0):
        xp = x0.copy()
        h = eps*xi
        xp[i] = xi + h
        dQ_dx[i] = (Q(xp, profile) - Q0)/h

    if (plot):
        labels = [r'Flux Limiter Electron', r'Flux Limiter Ion', r'RA',r'Coupling',r'LPI Loss']

        plt.bar(labels, x0 * np.abs(dQ_dx))

        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.ylabel(r'Scaled First Order Sensitivity $\left(\bar{\theta_{i}}\,\,\frac{d Q(\theta)}{d\theta_{i}}\right)$', fontsize = 20)
        if (likelihood):
            plt.title(r'$Q(\theta) = \log \mathbb{P} \left(\mathbf{y} | \mathbf{x}, \mathbf{\theta}, \sigma_{{Exp}}\right)$', fontsize = 25)
        if (shock_time):
            plt.title(r'$Q(\theta) = \mathbb{R} \left( t, t_{{Exp} }\right)$', fontsize = 25)
        plt.show()

    if (scaled):
        return x0 * dQ_dx
    else:
        return dQ_dx










