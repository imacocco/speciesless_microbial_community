#!/usr/bin/env python
# coding: utf-8

from py_module.multi_patch import *

np.random.seed(2)
max_iter = 1000
N_patches = 15
phen_per_patch = 40
resources = np.ones(20)*N_patches
test = Multi_Patch(N_patches, phen_per_patch,resources,perm=0.75,ekpyrosis=True)


test.evolve_async(n_steps=5000,parallel=4, max_iter=max_iter, plt = False)


#plt.figure()
#[ test.patches[alpha].print_patch() for alpha in range(test.n_patches) ]
#plt.show()

plt.figure()
[plt.scatter(np.ones(phen_per_patch)*(i+1),test.final_populations[i]) for i in range(N_patches)]
plt.savefig('final_pops.png')


mask = test.compute_multi_mask()

new = Multi_Patch(multi_patch = test, mask = mask)

new.compute_M_ab()

new.print_M_ab('M_ab.png')

new.compute_spectrum(plot=False)

new.plot_spectrum(yscale='linear',fileout='evl_lin.png')

new.print_eigenpatch(part=False, fileout = 'eigen_modes.png')
new.print_eigenpatch(part=True, fileout = 'eigen_modes_cut.png')

new.plot_spectrum(yscale='log',fileout='evl_log.png')