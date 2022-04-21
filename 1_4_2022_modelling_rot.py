import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
import numpy as np
import theano.tensor as tt
from astropy.table import Table
import pickle
import sys

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

def load_kernels(size=4800):
    kerns = np.nan * np.ones((15, size))
    for i in range(15):
            path = "l.{l:.0f}_n.{n:.0f}".format(l=list_pairs[i,0], n=list_pairs[i,1])
            kerns[i, :] = rot_k_table[path]
    return kerns

kern = load_kernels()

def pm_splittings(omega, l):
    #rot splittings for pymc3 model
    l_bounds = l_dict[str(l)]

    lef = l_bounds[0]
    rig = l_bounds[1]

    left = tt.dot(x_diffs[1:],(omega[1:]*kern[lef:rig,1:]).T)
    right = tt.dot(x_diffs[1:],(omega[:-1]*kern[lef:rig,:-1]).T)
    area = tt.add(left,right)
    return area/2 * betas[lef:rig]

def get_splittings(omega, l):
    #generate splittings for synthetic rot profiles
    l_bounds = l_dict[str(l)]
    lef = l_bounds[0]
    rig = l_bounds[1]
    area = np.trapz((omega * kern[lef:rig]),x)
    return area * betas[lef:rig]

def step(core,surface,point):
    init = surface * np.ones_like(x)
    dif = core-surface
    val = dif*(x < point) + init
    return  val

def run_mcmc_sampling(p_true,s=1,col = 'tab:blue',plot_chains=False, t=0.85):
    # f_out = open(str(p_true)+str(s)+str(t)+"_terminal.out", 'w')
    # sys.stdout = f_out



    #real splittings for kic 12508433
    #l1_splittings = np.array(obs_table['delta'][:11])*1e3
    #l2_splittings = np.array(obs_table['delta'][11:])*1e3

    #generate synthetic splittings
    test_omega = step(525,187,p_true)
    test_omega = test_omega

    l1_splittings = get_splittings(test_omega,1)
    l2_splittings = get_splittings(test_omega,2)


    e_l1_splittings = np.array(obs_table['e_delta'][:11])*1e3/s
    e_l2_splittings = np.array(obs_table['e_delta'][11:])*1e3/s



    with pm.Model() as model:

        mu1 = pm.Uniform("mu_omega_1",300, 600)
        #mu1 = 500

        mu2 = pm.Uniform("mu_omega_2",100, 300)
        #mu2 = 200
        point = pm.Uniform("point",0,1)



        omega = pm.Deterministic('omega', tt.switch(x < point, mu1, mu2))

        l1_ = pm.Normal(
            'l1',
            mu=pm_splittings(omega, 1),
            sd=e_l1_splittings,
            observed=l1_splittings
        )

        l2_ = pm.Normal(
            'l2',
            mu=pm_splittings(omega, 2),
            sd=e_l2_splittings,
            observed=l2_splittings
        )




        trace = pm.sample(10000, tune=2000, chains=2,cores = 4,target_accept = t,return_inferencedata=True, start = {"mu_omega_1":525,"mu_omega_2":187,"point":p_true})

    pickle_file_name_trace = str(p_true)+str(s)+str(t)+'_trace.pkl'
    outfile_trace = open(pickle_file_name_trace,'wb')
    pickle.dump(trace,outfile_trace)
    outfile_trace.close()

    

    sample = np.vstack([trace[k] for k in ["mu_omega_1","mu_omega_2", "point"]]).T

    pickle_file_name_sampled = str(p_true)+str(s)+str(t)+'_pred_trace.pkl'
    outfile_sampled = open(pickle_file_name_sampled,'wb')
    pickle.dump(sample,outfile_sampled)
    outfile_sampled.close()
    #plot corner plot
    corner.corner(sample,labels=[r"$\Omega_c/2 \pi$ [nHz]",r"$\Omega_s/2 \pi$ [nHz]", 'p'],
                           #quantiles=[0.16, 0.5, 0.84],
                           truths = [525,187,p_true],
                          truth_color =col,
                           show_titles=True, title_kwargs={"fontsize": 10});
    #plt.savefig('sampled'+str(p_true)+'.png',dpi=600)
    plt.savefig(str(p_true)+str(s)+str(t)+'_corner.png')
    #plot sampled splittings
    plt.figure(figsize=[7.5,4])
    N_samples = 100
    with model:
        pred_samples = pm.sampling.sample_posterior_predictive(trace,samples=N_samples,var_names=['l1','l2','omega'])

    pickle_file_name_samples = str(p_true)+str(s)+str(t)+'_pred_samples.pkl'
    outfile_samples = open(pickle_file_name_samples,'wb')
    pickle.dump(pred_samples,outfile_samples)
    outfile_samples.close()


    l1_mean = np.mean(pred_samples['l1'],axis = 0)
    l2_mean = np.mean(pred_samples['l2'],axis = 0)
    l1_std = np.std(pred_samples['l1'],axis = 0)
    l2_std = np.std(pred_samples['l2'],axis = 0)



    omega_samps = pred_samples['omega']
    omega_mean = np.mean(omega_samps,axis = 0)
    omega_std = np.std(omega_samps,axis = 0)

    plt.scatter(freqs[:11],l1_splittings,color = 'k',label = r'$\ell = 1$')
    plt.errorbar(freqs[:11],l1_splittings,yerr=e_l1_splittings,ls = '',color = 'k')
    plt.scatter(freqs[11:],l2_splittings,color = 'k',marker = 'x',label = r'$\ell = 2$')
    plt.errorbar(freqs[11:],l2_splittings,yerr=e_l2_splittings,ls = '',color ='k')


    plt.scatter(freqs[:11],l1_mean,color = col,label = r'$\ell = 1$')
    plt.errorbar(freqs[:11],l1_mean,yerr=l1_std,ls = '',color = col)
    plt.scatter(freqs[11:],l2_mean,color = col,marker = 'x',label = r'$\ell = 2$')
    plt.errorbar(freqs[11:],l2_mean,yerr=l2_std,ls = '',color =col)


    plt.ylabel('Rotational Splitting (nHz)')
    plt.xlabel(r'Oscillation Frequency ($\mu$Hz)')
    plt.savefig(str(p_true)+str(s)+str(t)+'_splittings.png')



    if (plot_chains == True):
        fig, axs = plt.subplots(3,figsize=[10,7])
        axs[0].set_title('Chains following sampling 2 chains of 10,000 samples p = 0.5 r/R, s = 1. Blue = Chain 1. Orange = Chain 2.')

        axs[0].plot(trace['mu_omega_1'][:10000],alpha=0.5)
        axs[0].plot(trace['mu_omega_1'][10000:],alpha=0.5)
        axs[0].set_ylim(0,700)
        axs[0].set_ylabel(r'$\Omega_c /2\pi$ (nHz)')

        axs[1].plot(trace['mu_omega_2'][:10000],alpha=0.5)
        axs[1].plot(trace['mu_omega_2'][10000:],alpha=0.5)
        axs[1].set_ylim(0,300)
        axs[1].set_ylabel(r'$\Omega_s /2\pi$ (nHz)')

        axs[2].plot(trace['point'][:10000],alpha=0.5)
        axs[2].plot(trace['point'][10000:],alpha=0.5)
        axs[2].set_ylabel(r'$p$ (r/R)')
        #plt.ylim(0,300)
        plt.savefig(str(p_true)+str(s)+str(t)+'_traces.png')

        for ax in axs.flat:
            ax.label_outer()

        sample_a = sample[:10000]
        sample_b = sample[10000:]
        fig = corner.corner(sample_a,labels=[r"$\Omega_c/2 \pi$ [nHz]",r"$\Omega_s/2 \pi$ [nHz]", 'p'],
                               #quantiles=[0.16, 0.5, 0.84],
                               truths = [525,187,0.5],
                              truth_color ='k',
                              color = 'tab:blue',
                              alpha = 0.5,
                               show_titles=True, title_kwargs={"fontsize": 10});

        corner.corner(sample_b,labels=[r"$\Omega_c/2 \pi$ [nHz]",r"$\Omega_s/2 \pi$ [nHz]", 'p'],
                               #quantiles=[0.16, 0.5, 0.84],
                               truths = [525,187,0.5],
                              truth_color ='k',
                              color = 'tab:orange',
                              alpha = 0.5,
                                fig = fig,
                               show_titles=True, title_kwargs={"fontsize": 10});
    plt.savefig(str(p_true)+str(s)+str(t)+'_corners.png')
    # f_out.close()
    return trace,pred_samples

if __name__ == '__main__':
    ps = np.array([0.05,0.2,0.5])
    ss = np.array([1])
    ts = np.array([0.8,0.85,0.88])
    for pp in ps:
        for sss in ss:
            for tm in ts:
                run_mcmc_sampling(pp,s=sss,plot_chains=True,t = tm)
    # run_mcmc_sampling(0.05,plot_chains=True)
    # run_mcmc_sampling(0.05,plot_chains=True)
    # run_mcmc_sampling(0.05,plot_chains=True)
    # run_mcmc_sampling(0.05,plot_chains=True)
