"""
PSD plots for VAR(2) by MCMC

"""
import os
os.chdir('C:/MH_MCMC_PSD') # set work directory

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import spec_mh_mcmc # model module
import true_var # to simulate data

n = 256
sigma = np.array([[1., 0.9], [0.9, 1.]])  
varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
vmaCoef = np.array([[[1.,0.],[0.,1.]]])
 
Simulation = true_var.VarmaSim(n=n)
freq = (np.arange(0,np.floor_divide(n, 2), 1) / (n))
spec_true = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)
spec_true = spec_true/(np.pi/0.5)

freq = freq*(2*np.pi)

data = Simulation.simData(varCoef, vmaCoef, sigma=sigma)

Spec = spec_mh_mcmc.SpecMCMC(data)
result_list = Spec.run_VI_MCMC(N_delta=8, N_theta=8, lr_map=0.015, ntrain_map=5000, 
                            nchunks=1, time_interval=n, required_part=0.5, 
                            num_samples=2000, burn_in=100)

spec_mat_q = result_list[0]/(np.pi/0.5)
spectral_density_all = result_list[3]/(np.pi/0.5)
spec_mat_median = spec_mat_q[1]



#uniform CI-------------------------------------------------------------------------------------------

def complex_to_real(matrix):
    n = matrix.shape[0]
    real_matrix = np.zeros_like(matrix, dtype=float)
    real_matrix[np.triu_indices(n)] = np.real(matrix[np.triu_indices(n)])
    real_matrix[np.tril_indices(n, -1)] = np.imag(matrix[np.tril_indices(n, -1)])
    
    return real_matrix

n_samples, n_freq, p, _ = spectral_density_all.shape

real_spectral_density_all = np.zeros_like(spectral_density_all, dtype=float)
real_spec_mat_median = np.zeros_like(spec_mat_median, dtype=float)

for i in range(n_samples):
    for j in range(n_freq):
        real_spectral_density_all[i, j] = complex_to_real(spectral_density_all[i, j])

for j in range(n_freq):
    real_spec_mat_median[j] = complex_to_real(spec_mat_median[j])


from scipy.stats import median_abs_deviation

mad = median_abs_deviation(real_spectral_density_all, axis=0, nan_policy='omit')
mad[mad == 0] = 1e-10 

def uniformmax_multi(mSample):
    N_sample, N, d, _ = mSample.shape
    C_help = np.zeros((N_sample, N, d, d))

    for j in range(N):
        for r in range(d):
            for s in range(d):
                C_help[:, j, r, s] = uniformmax_help(mSample[:, j, r, s])

    return np.max(C_help, axis=0)

def uniformmax_help(sample):
    return np.abs(sample - np.median(sample)) / median_abs_deviation(sample)
    
max_std_abs_dev = uniformmax_multi(real_spectral_density_all) 
    
    
threshold = np.quantile(max_std_abs_dev, 0.9)    
lower_bound = real_spec_mat_median - threshold * mad
upper_bound = real_spec_mat_median + threshold * mad

def real_to_complex(matrix):
    n = matrix.shape[0]
    complex_matrix = np.zeros((n, n), dtype=complex)
    
    complex_matrix[np.diag_indices(n)] = matrix[np.diag_indices(n)]
    
    complex_matrix[np.triu_indices(n, 1)] = matrix[np.triu_indices(n, 1)] - 1j * matrix[np.tril_indices(n, -1)]
    complex_matrix[np.tril_indices(n, -1)] = matrix[np.triu_indices(n, 1)] + 1j * matrix[np.tril_indices(n, -1)]
    
    return complex_matrix

complex_lower_bound = np.zeros_like(lower_bound, dtype=complex)
complex_upper_bound = np.zeros_like(upper_bound, dtype=complex)

for i in range(n_freq):
    complex_lower_bound[i] = real_to_complex(lower_bound[i])
    complex_upper_bound[i] = real_to_complex(upper_bound[i])

spec_mat_lower = complex_lower_bound
spec_mat_upper = complex_upper_bound


coh_all = result_list[4]
coh_med = result_list[2][1]

mad = median_abs_deviation(coh_all, axis=0, nan_policy='omit')
mad[mad == 0] = 1e-10 

def uniformmax_multi(coh_whole):
    N_sample, N, d = coh_whole.shape
    C_help = np.zeros((N_sample, N, d))

    for j in range(N):
        for r in range(d):
                C_help[:, j, r] = uniformmax_help(coh_whole[:, j, r])

    return np.max(C_help, axis=0)

    
max_std_abs_dev = uniformmax_multi(coh_all) 

threshold = np.quantile(max_std_abs_dev, 0.9)    
coh_lower = coh_med - threshold * mad
coh_upper = coh_med + threshold * mad

#------------------------------------------------------------------------------------------
#plot

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.subplots_adjust(hspace=0.08, wspace=0.08)

from matplotlib.lines import Line2D

for i in range(2):
    for j in range(2):
        if i == j:
            f, Pxx_den0 = signal.periodogram(data[:,i], fs=2*np.pi)
            f = f[1:]
            Pxx_den0 = Pxx_den0[1:] 
            axes[i, j].plot(f, np.log(Pxx_den0), marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.5)
            
            axes[i, j].plot(freq, np.log(np.abs(spec_mat_median[..., i, i])), linewidth=1, color='green', linestyle="-")
            axes[i, j].fill_between(freq, np.log(np.abs(lower_bound[..., i, i])),
                            np.log(np.abs(upper_bound[..., i, i])), color='lightgreen', alpha=1)
            
            axes[i, j].plot(freq, np.log(np.real(spec_true[:,i,i])), linewidth=1, color = 'red', linestyle="-.")
            
            
            axes[i, j].text(0.95, 0.95, r'$\log(f_{{{}, {}}})$'.format(i+1, i+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)
            
            axes[i, j].grid(True)
            
        elif i < j:  
            
            y = np.apply_along_axis(np.fft.fft, 0, data)
            if np.mod(n, 2) == 0:
                # n is even
                y = y[0:int(n/2)]
            else:
                # n is odd
                y = y[0:int((n-1)/2)]
            y = y / np.sqrt(n)
            #y = y[1:]
            cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
            
            axes[i, j].plot(f, np.real(cross_spectrum_fij)/(freq[-1]/0.5),
                            marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.5)
    
            
            axes[i, j].plot(freq, np.real(spec_mat_median[..., i, j]), linewidth=1, color='green', linestyle="-")
            axes[i, j].fill_between(freq, (lower_bound[..., i, j]), (upper_bound[..., i, j]),
                           color='lightgreen', alpha=1)
            
            axes[i, j].plot(freq, np.real(spec_true[:,i,j]), linewidth=1, color = 'red', linestyle="-.")
            
            axes[i, j].text(0.95, 0.95, r'$\Re(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)
        
            axes[i, j].grid(True)
            
        else:

            y = np.apply_along_axis(np.fft.fft, 0, data)
            if np.mod(n, 2) == 0:
                # n is even
                y = y[0:int(n/2)]
            else:
                # n is odd
                y = y[0:int((n-1)/2)]
            y = y / np.sqrt(n)
            #y = y[1:]
            cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
            
            axes[i, j].plot(f, np.imag(cross_spectrum_fij)/(freq[-1]/0.5),
                            marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.5)
            
            axes[i, j].plot(freq, np.imag(spec_mat_median[..., i, j]), linewidth=1, color='green', linestyle="-")
            
         
            axes[i, j].fill_between(freq, (lower_bound[..., i, j]), (upper_bound[..., i, j]),
                           color='lightgreen', alpha=1)
            
            axes[i, j].plot(freq, np.imag(spec_true[:,i,j]), linewidth=1, color = 'red', linestyle="-.")
            
            axes[i, j].text(0.95, 0.95, r'$\Im(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)

            axes[i, j].grid(True)
fig.text(0.5, 0.08, 'Frequency', ha='center', va='center', fontsize=20)
fig.text(0.08, 0.5, 'Spectral Densities', ha='center', va='center', rotation='vertical', fontsize=20)

fig.legend(handles=[Line2D([], [], color='lightgray', label='Periodogram'),
                Line2D([], [], color='red', label='True'),
                Line2D([], [], color='green', label='HMCMC'),
            Line2D([], [], color='lightgreen', label='90% HMCMC VI')],
             loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=14)


# plot squared coherence

fig, ax = plt.subplots(1,1, figsize = (20, 6))

plt.plot(freq, np.squeeze(coh_med[:,0]), color = 'green', linestyle="-", label = 'HMCMC', linewidth=1)
plt.fill_between(freq, np.squeeze(coh_lower[:,0]), np.squeeze(coh_upper[:,0]),
                    color = ['lightgreen'], alpha = 1, label = '90% CI HMCMC')

plt.plot(freq, np.absolute(spec_true[...,0,1])**2 / np.abs(spec_true[...,0,0]) / np.abs(spec_true[...,1,1]),
         color = 'red', linestyle="-", label = 'True', linewidth=1)


plt.xlabel('Frequency', fontsize=20, labelpad=10)   
plt.ylabel('Squared Coherency', fontsize=20, labelpad=10)   
plt.title('Squared coherence', pad=20, fontsize = 20)

plt.legend(loc='upper left', fontsize='medium')
plt.grid(True)





















