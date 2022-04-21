import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
import numpy as np
import theano.tensor as tt
from astropy.table import Table
import pickle


obs_table = Table.read("freq.dat", format="ascii")
beta_table = Table.read("beta.dat", format="ascii")
freq_table = Table.read("sample_0549-freqs.dat", format="ascii")
rot_k_table = Table.read("sample_0549-freqs-rotk.dat", format="ascii")


list_pairs = []
for i in range(15):
    list_pairs.append([obs_table['l'][i],obs_table['n'][i]])

list_pairs= np.array(list_pairs)

betas = np.ones(15)
freqs = np.ones(15)
obs = np.ones(15)
err = np.ones(15)
for i in range(15):
    freq_mask = (freq_table['l']==list_pairs[i,0]) & (freq_table['n']==list_pairs[i,1])
    beta_mask = (beta_table['l']==list_pairs[i,0]) & (beta_table['n']==list_pairs[i,1])
    obs_mask = (obs_table['l']==list_pairs[i,0]) & (obs_table['n']==list_pairs[i,1])
    freqs[i] = freq_table[freq_mask]['nu']
    betas[i] = beta_table[beta_mask]['beta']
    obs[i] = obs_table[obs_mask]['delta']
    err[i] = obs_table[obs_mask]['e_delta']


x = np.array(rot_k_table['x'])
x_diffs = np.hstack([0, np.diff(x)])
l_dict = {'1':[None,11],'2':[11,None]}




l1_splittings = np.array(obs_table['delta'][:11])*1e3
l2_splittings = np.array(obs_table['delta'][11:])*1e3


def step(core,surface,point):
    init = surface * np.ones_like(x)
    dif = core-surface
    val = dif*(x < point) + init
    return  val


class Data:
    def __init__(self, p_true,s,t):
        pickle_file_name_samples = str(p_true)+str(s)+str(t)+'_pred_samples.pkl'
        #pickle_file_name_trace = str(p_true)+str(s)+str(t)+'_pred_trace.pkl'
        pickle_file_name_pred_traces = str(p_true)+str(s)+str(t)+'_pred_trace.pkl'
        try:
            self.pred_data = pickle.load(open(pickle_file_name_samples,'rb'))
            self.pred_trace = pickle.load(open(pickle_file_name_pred_traces,'rb'))

            #self.chain_data = pickle.load(open(pickle_file_name_trace,'rb'))
            #self.traces = pickle.load(open(pickle_file_name_traces,'rb'))
        except:
            print('Parameters not yet run. Refer back to modelling')
        self.p_true = p_true
        self.s = s
        self.t = t


#fix below for inference data true
    def plot_chains(self):

        trace = self.pred_trace

        core_omega_chain_1 = trace[:10000,0]
        core_omega_chain_2 = trace[10000:,0]

        surf_omega_chain_1 = trace[:10000,1]
        surf_omega_chain_2 = trace[10000:,1]


        p_chain_1 = trace[:10000,2]
        p_chain_2 = trace[10000:,2]
        fig, axs = plt.subplots(3,figsize=[10,7])
        axs[0].set_title('Chains following sampling 2 chains of 10,000 samples p = '+str(self.p_true)+' r/R, s = '+str(self.s)+', t = '+str(self.t)+'. Blue = Chain 1. Orange = Chain 2.')

        axs[0].plot(core_omega_chain_1,alpha=0.5)
        axs[0].plot(core_omega_chain_2,alpha=0.5)
        axs[0].set_ylim(0,700)
        axs[0].set_ylabel(r'$\Omega_c /2\pi$ (nHz)')

        axs[1].plot(surf_omega_chain_1,alpha=0.5)
        axs[1].plot(surf_omega_chain_2,alpha=0.5)
        axs[1].set_ylim(0,300)
        axs[1].set_ylabel(r'$\Omega_s /2\pi$ (nHz)')

        axs[2].plot(p_chain_1,alpha=0.5)
        axs[2].plot(p_chain_2,alpha=0.5)
        axs[2].set_ylabel(r'$p$ (r/R)')
        #plt.ylim(0,300)
        plt.savefig(str(self.p_true)+str(self.s)+str(self.t)+'_traces.png')

        for ax in axs.flat:
            ax.label_outer()
        return

    def plot_corners(self):
        #sample = self.trace
        sample_a = np.vstack([self.pred_trace[:10000,k] for k in [0,1,2]]).T
        sample_b = np.vstack([self.pred_trace[10000:,k] for k in [0,1,2]]).T


        fig = corner.corner(sample_a,labels=[r"$\Omega_c/2 \pi$ [nHz]",r"$\Omega_s/2 \pi$ [nHz]", 'p'],
                           #quantiles=[0.16, 0.5, 0.84],
                           truths = [525,187,self.p_true],
                          truth_color ='k',
                          color = 'tab:blue',
                          alpha = 0.5,
                           show_titles=True, title_kwargs={"fontsize": 10});

        corner.corner(sample_b,labels=[r"$\Omega_c/2 \pi$ [nHz]",r"$\Omega_s/2 \pi$ [nHz]", 'p'],
                           #quantiles=[0.16, 0.5, 0.84],
                           truths = [525,187,self.p_true],
                          truth_color ='k',
                          color = 'tab:orange',
                          alpha = 0.5,
                            fig = fig,
                           show_titles=True, title_kwargs={"fontsize": 10});
        plt.savefig(str(self.p_true)+str(self.s)+str(self.t)+'_corners.png')
        return

    def plot_corner(self):
        sample = np.vstack([self.pred_trace[:,k] for k in [0,1,2]]).T
        corner.corner(sample,labels=[r"$\Omega_c/2 \pi$ [nHz]",r"$\Omega_s/2 \pi$ [nHz]", 'p'],
                               #quantiles=[0.16, 0.5, 0.84],
                               truths = [525,187,self.p_true],
                              truth_color ='k',
                               show_titles=True, title_kwargs={"fontsize": 10});
        #plt.savefig('sampled'+str(p_true)+'.png',dpi=600)
        plt.savefig(str(self.p_true)+str(self.s)+str(self.t)+'_corner.png')

        return

    def plot_splitting_predicition(self):
        pred_samples=self.pred_data
        l1_mean = np.mean(pred_samples['l1'],axis = 0)
        l2_mean = np.mean(pred_samples['l2'],axis = 0)
        l1_std = np.std(pred_samples['l1'],axis = 0)
        l2_std = np.std(pred_samples['l2'],axis = 0)

        e_l1_splittings = np.array(obs_table['e_delta'][:11])*1e3/self.s
        e_l2_splittings = np.array(obs_table['e_delta'][11:])*1e3/self.s

        omega_samps = pred_samples['omega']
        omega_mean = np.mean(omega_samps,axis = 0)
        omega_std = np.std(omega_samps,axis = 0)

        plt.scatter(freqs[:11],l1_splittings,color = 'k',label = r'$\ell = 1$')
        plt.errorbar(freqs[:11],l1_splittings,yerr=e_l1_splittings,ls = '',color = 'k')
        plt.scatter(freqs[11:],l2_splittings,color = 'k',marker = 'x',label = r'$\ell = 2$')
        plt.errorbar(freqs[11:],l2_splittings,yerr=e_l2_splittings,ls = '',color ='k')

        col_dict = {0.2:'tab:blue',0.05:'tab:red',0.5:'tab:purple',1:'tab:pink'}

        plt.scatter(freqs[:11],l1_mean,color = col_dict[self.p_true],label = r'$\ell = 1$')
        plt.errorbar(freqs[:11],l1_mean,yerr=l1_std,ls = '',color = col_dict[self.p_true])
        plt.scatter(freqs[11:],l2_mean,color = col_dict[self.p_true],marker = 'x',label = r'$\ell = 2$')
        plt.errorbar(freqs[11:],l2_mean,yerr=l2_std,ls = '',color =col_dict[self.p_true])


        plt.ylabel('Rotational Splitting (nHz)')
        plt.xlabel(r'Oscillation Frequency ($\mu$Hz)')
        plt.savefig(str(self.p_true)+str(self.s)+str(self.t)+'_splittings.png')
        return


    def plot_profile_predicitions(self):
        pred_samples=self.pred_data
        trace = self.pred_trace


        sample = np.vstack([self.pred_trace[:,k] for k in [0,1,2]]).T


        mean_rot_prof_params = np.mean(sample,axis = 0)

        mean_param_prof = step(mean_rot_prof_params[0],mean_rot_prof_params[1],mean_rot_prof_params[2])

        omega_pred = pred_samples['omega']

        plt.plot(x,np.mean(omega_pred,axis =0),label = r'Mean of predicted $\Omega$')

        plt.plot(x,mean_param_prof,label = r'$\Omega$ from mean of profile Parameters')

        plt.xlabel('r/R')
        plt.ylabel(r'$\Omega/2\pi$')
        plt.legend(loc = 'best')

        return
