#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns;# sns.set()
import numpy.linalg as linalg
#plt.ioff()



def FLIP(before):
    if before==0:
        return 1
    else:
        return 0

def load_competitor(base_phenotype, skip=0, n_flip=0, prob_flip=0):
    
    competitor = list.copy(base_phenotype)
    
    if (n_flip != 0):
        #select n_flip random pathways (after the fixed ones) to flip
        idx_flip =  np.random.choice( np.arange(skip,len(base_phenotype)), n_flip, replace = False) 
        for i in idx_flip:
            competitor[i] = FLIP(competitor[i])
        return competitor
    
    else:
        #randomly flip each pathway (after the fixed ones) with given prob_flip
        for i in range(skip,len(base_phenotype)):
            if(np.random.rand() > (1 - prob_flip) ):
                competitor[i] = FLIP(competitor[i])
        return competitor


def compute_ind_per_res(populations,phenotypes):
    return np.array( [sum(phenotypes[:,i]*populations) for i in range(len(phenotypes[0])) ] )


def compute_surplus(phenotypes,resources,populations,costs):
    ind_per_resource = compute_ind_per_res(populations,phenotypes)
    res_per_ind = resources/ind_per_resource
    surplus = [ sum(phenotypes[i]*res_per_ind) - costs[i] for i in range(len(phenotypes)) ]
    return np.array(surplus)


def compute_costs(phenotypes,noise):
    return np.array( [ sum(phenotypes[i])+np.random.rand()*noise for i in range(len(phenotypes)) ] )


def compute_ADJ(phenotypes):
    adj = []
    for i in range(len(phenotypes)):
        adj.append( [ sum(phenotypes[i]*phenotypes[j]) for j in range(len(phenotypes)) ] )
        
    return np.array(adj)


def ODE_system_ivp(t, z, phenotypes,resources,costs):
    surplus = compute_surplus(phenotypes,resources,z,costs)
    next_step = z*surplus/costs
#    if np.all(abs(next_step) < 1e-5) :
#        print(t)
#        print(next_step)
 #   else:
    return next_step

def evolve(n_steps, resources, populations, phenotypes, costs, plot=True,verb=True):
    if verb==True:
        print(resources[0])
    #solve the ODE system
    sol = integrate.solve_ivp(ODE_system_ivp, [0, n_steps], populations, args=(phenotypes, resources, costs), dense_output=True)
    #save and visualise the solution
    t = np.linspace(0, n_steps, int(n_steps/10))
    z = sol.sol(t)
    if plot == True:
        [plt.plot(t, z.T[:,i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
        plt.show()
    #return the final population to a list
    return z.T[-1]


def compute_mask(populations, threshold):
    masks = [ populations[i] > threshold for i in range(len(populations)) ]
    masks = np.array(masks)
    [ print(sum(masks[i])) for i in range(len(masks)) ]
    final_mask = [ np.all(masks[:,i]) for i in range(len(masks[0])) ]
    return final_mask


def compute_M_ab(populations,phenotypes,costs,resources):
    M_ab = []
    T = compute_ind_per_res(populations,phenotypes)
    #print(len(T),len(X),len(phenotypes[0]),len(resources))
    for i in range(len(phenotypes)):
        M_ab.append( [ -(populations[i]/costs[i])*sum(phenotypes[i]*phenotypes[j]*resources/T/T) for j in range(len(phenotypes)) ] )
    return np.array(M_ab)


def plot_spectrum(evl):
    x = np.ones(len(evl))
    #plt.scatter(x,evl,linewidths=0)
    plt.errorbar(x,abs(evl),xerr=0.01,fmt='none', ecolor='k')
    plt.xticks([])
    plt.xlabel('')
    plt.ylabel('|$\lambda$|')
    plt.show()

'''
# # Single patch

# ## Ex1

# In[110]:


np.random.seed(2)
N_resources = 20
N_phenotypes = 20
resources = np.ones(N_resources)*1#np.random.rand(N_sources)#
noise = 0.01

core_phenotypes = []
core_phenotypes.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1])
core_phenotypes.append([0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0])
core_phenotypes.append([1,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
core_phenotypes.append([0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])

populations = np.ones(N_phenotypes)

phenotypes = []
for i in range(N_phenotypes):
    comp_per_phen = int(N_phenotypes/len(core_phenotypes))
    #print(i,comp_per_phen,int(i/comp_per_phen))
    phenotypes.append( load_competitor( core_phenotypes[int(i/comp_per_phen)], prob_flip=0.05 ) )

phenotypes = np.array(phenotypes)


# In[111]:


ax = sns.heatmap(phenotypes)


# In[112]:


#ind_per_resource = compute_ind_per_res(populations,phenotypes)
costs = compute_costs(phenotypes,noise)
#adj = compute_ADJ(phenotypes)
#surplus = compute_surplus(phenotypes,resources,populations,costs)


# In[119]:


t_end = 100000
sol = integrate.solve_ivp(ODE_system_ivp, [0, t_end], populations, args=(phenotypes, resources, costs), dense_output=True)


# In[120]:


t = np.linspace(0, t_end, int(t_end/10))
z = sol.sol(t)
[plt.plot(t, z.T[:,i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
#[plt.plot(sol.t, sol.y[i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
#plt.xlim(0,40)
#plt.ylim(0,100)
#plt.legend()
plt.show()


# In[121]:


z.T[-1]


# In[122]:


mask = z.T[-1]>1e-4


# In[123]:


active_pheno = phenotypes[mask]
active_pop = z.T[-1][mask]
active_costs = costs[mask]
ax = sns.heatmap(active_pheno)
active_pop


# In[126]:


M_ab = compute_M_ab(active_pop,active_pheno,active_costs,resources)


# In[127]:


sns.heatmap(M_ab)


# In[128]:


evl, evc = linalg.eig(M_ab)
idx = evl.argsort()
evl = evl[idx]
evc = evc[:,idx]
T = evc.T


# In[129]:


evl


# In[130]:


plot_spectrum(evl)


# In[131]:


sns.heatmap(np.real(evc),cmap=eig_cmap);


# In[132]:


sns.heatmap(np.dot(np.real(T),active_pheno),cmap=eig_cmap);


# ## EX2

# In[12]:


np.random.seed(1)
N_resources = 21
N_phenotypes = 30
resources = np.ones(N_resources)*1#np.random.rand(N_sources)#
noise = 0.01

core_phenotypes = []
core_phenotypes.append([1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1])
core_phenotypes.append([0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1])
core_phenotypes.append([0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1])
core_phenotypes.append([1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
core_phenotypes.append([0,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
core_phenotypes.append([0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])

populations = np.ones(N_phenotypes)

phenotypes = []
for i in range(N_phenotypes):
    comp_per_phen = int(N_phenotypes/len(core_phenotypes))
    #print(i,comp_per_phen,int(i/comp_per_phen))
    phenotypes.append( load_competitor( core_phenotypes[int(i/comp_per_phen)],skip=3,n_flip=2 ) )

phenotypes = np.array(phenotypes)
sns.heatmap(phenotypes);


# In[13]:


#ind_per_resource = compute_ind_per_res(populations,phenotypes)
costs = compute_costs(phenotypes,noise)
#adj = compute_ADJ(phenotypes)
#surplus = compute_surplus(phenotypes,resources,populations,costs)


# t_end = 10000
# sol = integrate.solve_ivp(ODE_system_ivp, [0, t_end], populations, args=(phenotypes, resources, costs), dense_output=True)

# t = np.linspace(0, t_end, int(t_end/10))
# z = sol.sol(t)
# [plt.plot(t, z.T[:,i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
# #[plt.plot(sol.t, sol.y[i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
# #plt.xlim(0,40)
# #plt.ylim(0,100)
# #plt.legend()
# plt.show()

# active_pheno = phenotypes[z.T[-1]>1e-5]
# active_pop = z.T[-1][z.T[-1]>1e-5]
# ax = sns.heatmap(active_pheno)
# active_pop

# M_ab = compute_M_ab(active_pop,active_pheno,resources)

# sns.heatmap(M_ab)

# evl, evc = linalg.eig(M_ab)
# idx = evl.argsort()
# evl = evl[idx]
# evc = evc[:,idx]
# T = evc.T

# evl

# plot_spectrum(evl)

# sns.heatmap(np.real(evc),cmap=eig_cmap);

# sns.heatmap(np.dot(np.real(T),active_pheno),cmap=eig_cmap);

# In[14]:


final_pop_list = []
mask_list = []
res_list = []


# In[15]:


#decide how to evolve the resources across the simulation
res_i = np.logspace(0,3.91,20,base=np.e)
print(res_i)


# In[16]:


#set the number of integration steps
t_end=15000
#set the resources
res_temp = np.ones(N_resources)*1#np.random.rand(N_sources)#
for i in res_i:
    #change certain resources
    res_temp[0]=i
    res_temp[1]=i
    res_temp[2]=i
    print(i)
    #create the list of reources (in order to compute the interaction matrix)
    res_list.append(np.copy(res_temp))
    #solve the ODE system
    sol = integrate.solve_ivp(ODE_system_ivp, [0, t_end], populations, args=(phenotypes, res_temp, costs), dense_output=True)
    #visualise the solution
    t = np.linspace(0, t_end, int(t_end/10))
    z = sol.sol(t)
    [plt.plot(t, z.T[:,i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
    plt.show()
    #add the final population to a list
    final_pop_list.append( z.T[-1] )
    #create the mask for this parameters
    mask_list.append( z.T[-1]>1e-5 )
    


# In[17]:


res_list = np.array(res_list)
mask_list = np.array(mask_list)
#create the final mask with only the populations that survived to all the different equilibrations
final_mask = [ np.all(mask_list[:,i]) for i in range(len(mask_list[0])) ]


# In[18]:


sum(final_mask)


# In[19]:


#prepare the surviving species for the new equilibration
active_pop = populations[final_mask]
active_pop1 = [ final_pop_list[i][final_mask] for i in range(len(res_list)) ]
active_phen = phenotypes[final_mask]
active_cost = costs[final_mask]


# In[20]:


sns.heatmap(active_phen)


# In[196]:


evl_list = []
evc_list = []
final_active_pop = []


# In[ ]:


for i in range(len(res_list)):
    print(i)
    sol = integrate.solve_ivp(ODE_system_ivp, [0, t_end], active_pop, args=(active_phen, res_list[i], active_cost), dense_output=True)
    t = np.linspace(0, t_end, int(t_end/10))
    z = sol.sol(t)
    [plt.plot(t, z.T[:,i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
    plt.show()
    final_active_pop.append( z.T[-1] )


# In[ ]:


for i in range(len(res)):
    M_ab = compute_M_ab(final_active_pop[i],active_phen,active_costs,res_list[i])
    evl, evc = linalg.eig(M_ab)
    idx = evl.argsort()
    evl_list.append(evl[idx])
    evc = evc[:,idx]
    evc_list.append(evc.T)
evl_list = np.array(evl_list)
evc_list = np.array(evc_list)


# M_ab = compute_M_ab(active_pop[0],active_phen,res_list[0])
# evl, evc = linalg.eig(M_ab)
# idx = evl.argsort()
# evl = evl[idx]
# evc = evc[:,idx]
# evc = evc.T

# In[178]:


plot_spectrum(evl_list[2])


# In[ ]:


[ plt.scatter(res_list[:,0],abs(evl_list[:,i]) ) for i in range(len(evl_list[0])) ];
plt.xscale('log')
plt.show()


# In[114]:


sns.heatmap(np.dot(evc_list[1],active_phen),cmap=eig_cmap)


# In[ ]:




#old way of solving ODE
def ODE_system(n, t, phenotypes,resources,costs):

    surplus = compute_surplus(phenotypes,resources,n,costs)
    next_step =  n*surplus/costs
    
    return next_step

t = np.linspace(0, 100, 1000)
X, infodict = integrate.odeint(ODE_system, populations, t, args=(phenotypes, resources, costs),full_output=True)

[plt.plot(t,X[:,i], label='phen'+ str(i)) for i in range(len(X[0]))];
#plt.ylim(0,1)
#plt.yscale('log')
plt.legend()
plt.show()

'''