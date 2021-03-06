from py_module.patch import *
from py_module.no_species import *
import sys
from joblib import Parallel, delayed

class Multi_Patch:

	def __init__(self, n_patches = 15, phen_per_patch = 30, resources = np.ones(20)*100, noise = 0.0001, perm = 1, conv_thres = 1e-4, verbose = True,\
					multi_patch = None, mask = None, ekpyrosis = False):

		if multi_patch is not None and mask is not None:
			
			self.patches = [ Patch( patch = multi_patch.patches[alpha], mask = mask[alpha]) for alpha in range(multi_patch.n_patches)]
			self.n_patches = multi_patch.n_patches
			self.total_resources = multi_patch.total_resources
			self.base_resources = multi_patch.base_resources
			self.permeability = multi_patch.permeability
			self.total_populations = self.extract_populations()
			self.final_populations = self.total_populations
			self.flux = [ self.base_resources for i in range(n_patches) ]
			self.threshold = multi_patch.threshold
			self.verbose = multi_patch.verbose

		else:
			
			self.n_patches = n_patches
			self.phen_per_patch = phen_per_patch
			self.total_resources = resources
			self.base_resources = self.total_resources/n_patches
			self.patches = [ Patch(noise = noise, n_phen = phen_per_patch, resources = self.base_resources, ekpyrosis = ekpyrosis) for i in range(n_patches) ]
			self.permeability = perm
			self.total_populations = self.extract_populations()
			self.final_populations = self.total_populations
			self.flux = [ self.base_resources for i in range(n_patches) ]
			self.threshold = conv_thres
			self.verbose = verbose

		self.M_ab = None 		# jacobian near equilibrium
		self.evc  = None 		# jacobian's eigenvectors
		self.evl  = None 		# jacobian's eqigenvalues


	def extract_populations(self):

		return [ self.patches[i].populations for i in range(self.n_patches) ]


	def extract_final_populations(self):

		return [ self.patches[i].final_populations for i in range(self.n_patches) ]


	def divide_resources_async(self):

		#compute resource availability in each patch, starting from the base_resources
		h = np.array( [ self.base_resources/self.patches[alpha].compute_individuals_per_resource( self.patches[alpha].final_populations ) for alpha in range(self.n_patches) ] )
		#alternative way of computing h, by using the resources at the previous step instead the equal ones
#		h = np.array( [ self.patches[alpha].resources/self.patches[alpha].compute_individuals_per_resource( self.patches[alpha].final_populations ) for alpha in range(self.n_patches) ] )
		#compute average availability for each resource across each patch
		h_star = np.array( [ sum(h[:,i])/self.n_patches for i in range(len(self.base_resources)) ] )
		#compute the flux for each resource across each patch 
		flux = np.array( [ self.permeability*( h[alpha] - h_star ) for alpha in range(self.n_patches) ] )
		#compute total flux for each patch
		relative_flux_diff = np.array( [ ( (sum( abs( flux[alpha] ) ) - sum( abs( self.flux[alpha] ) ) )/sum( abs( self.flux[alpha] ) ) ) for alpha in range(self.n_patches) ] )
		
		#check whether the relative magnitudes of change in resource influx is below a threshold for each patch
#		print(flux)
		if self.verbose:
			print('relative flux:',abs(relative_flux_diff))
		if np.all(  abs(relative_flux_diff) < self.threshold ):
			return False 	#and interrupt the while interation

		else: #otherwise assign new resources to the patches
			new_resources = np.array( [ self.base_resources - flux[alpha] for alpha in range(self.n_patches) ] )
			[ self.patches[alpha].set_resources(new_resources[alpha]) for alpha in range(self.n_patches) ]
			self.flux = flux
			return True


	def evolve_async(self, n_steps = None, thres = 1e-5, parallel = False, max_iter=20, plt = True):

		#asyncronous evolution: each path evolve independently and when they reach equilibrium
		#the resources flux are calculated with the final populations and updated
		#for the computation of the new flux

		counter = 0

		while(self.divide_resources_async() and counter<max_iter):

			print('iter_no:', counter)
			counter+=1
			if parallel is False:
				for alpha in range(self.n_patches):
#					print('alpha ',alpha)
					self.patches[alpha].evolve(n_steps=n_steps,thres=thres)

			else:
				#compute new_pops in parallel and update the final pops of the patches
				#NB parallel is a wrapper and is not able to modify the class variables!!!
				new_pops = Parallel(n_jobs=parallel)( delayed(self.patches[alpha].evolve)(n_steps=n_steps,thres=thres, traj = -1, plot = plt)\
					for alpha in range(self.n_patches) )
				#update final pops into the patches
				for alpha in range(self.n_patches):
					self.patches[alpha].final_populations = new_pops[alpha]
		

		#	THIS IS QUITE SURELY WRONG
		#	#update the actual populations to the final_populations for the next iteration
		#	for alpha in range(self.n_patches):
		#		self.patches[alpha].populations = self.patches[alpha].final_populations

#			print(self.extract_final_populations())
		
		self.final_populations = self.extract_final_populations()


	def print_M_ab(self, fileout = None):

		if self.M_ab is None:
			self.compute_M_ab()

		sns.heatmap(self.M_ab)

		if fileout is None:
			plt.show()
		else:
			plt.savefig(fileout)
			plt.close()


	def compute_M_ab(self):

		#compute interaction matrix

		M_ab = []

		#cycle over all patches
		for alpha in range(self.n_patches):
			#cycle over any pop in that patch
			for a in range(len(self.patches[alpha].populations)):
				#cycle over every patch 
				line = []
				for beta in range(self.n_patches):
					#cycle over evey pop in that patch
					for b in range(len(self.patches[beta].populations)):
						line.append(self.compute_M_ab_ele(a,b,alpha,beta))
#						M_ab.append(self.compute_M_ab_ele(a,b,alpha,beta))
				M_ab.append(line)

#		self.M_ab = np.array(M_ab).reshape( ( len(self.populations),len(self.populations) ) )

		self.M_ab = np.array(M_ab)


	def compute_M_ab_ele(self,a,b,alpha,beta):

		#compute element of interaction mtrix (BETA VERSION)

		A = self.patches[alpha]
		B = self.patches[beta]

		T_alpha = A.compute_individuals_per_resource()
		T_beta = B.compute_individuals_per_resource()
		sigma_a = A.phenotypes[a]
		sigma_b = B.phenotypes[b]

		temp = self.permeability*self.base_resources/self.n_patches/T_beta/T_beta 

		if alpha == beta:
			temp += ( A.resources - self.permeability*self.base_resources/T_alpha )/T_alpha

		temp *= sigma_a*sigma_b/T_alpha

		return -A.populations[a]/A.costs[a]*sum(temp)


	def compute_spectrum(self, plot = True):

		#compute eigenvalues and eigenvectors

		if self.M_ab is None:
			self.compute_M_ab()

		evl, evc = linalg.eig(self.M_ab)
		idx = evl.argsort()
		self.evl = evl[idx]
		self.evc = evc[:,idx]
#		T = evc.T
		if plot is True:
			self.plot_spectrum()


	def plot_spectrum(self, yscale = 'log', fileout = None):

		#plot the eigenvalues

		if self.evl is None:
			self.compute_spectrum()

		x = np.ones(len(self.evl))
		#plt.scatter(x,evl,linewidths=0)
		plt.errorbar(x,abs(self.evl),xerr=0.01,fmt='none', ecolor='k')
		plt.xticks([])
		plt.xlabel('')
		plt.ylabel('|$\lambda$|')
		plt.yscale(yscale)
		if yscale == 'log':
			plt.ylim((0.01,1.5))
		if fileout is None:
			plt.show()
		else:
			plt.savefig(fileout)
			plt.close()


	def print_eigenpatch(self, part = True, fileout = False):

		#print the eigenvector projected onto the pathways basis
		
		eig_cmap = sns.diverging_palette(264.5, 10.8, as_cmap=True, sep=40, s=100, center='light', l=50)
		
		if self.evc is None:
			self.compute_spectrum()

		appo = np.concatenate( [ self.patches[alpha].phenotypes for alpha in range(self.n_patches) ] )
		heatmap = np.dot( np.real(self.evc.T),appo )

		if part is True:
			heatmap = heatmap[:self.n_patches+6]

		sns.heatmap( heatmap, cmap=eig_cmap);
		
		if fileout is None:
			plt.show()
		else:
			plt.savefig(fileout)
			plt.close()


	def compute_multi_mask(self,threshold=1e-4):

		#compute the mask to consider only the surviving phenotypes

		mask = []
		for alpha in range(self.n_patches):
			mask.append( [ self.patches[alpha].final_populations[i] > threshold for i in range(self.phen_per_patch)] ) 

		return np.array( mask )

		
#--------------SYNCHRONOUS FUNCTIONS, GIVING SRTANGE RESULTS------------------



	def divide_resources_sync(self, pop = None):

		#function to divide resources according to populations of each patch

		if pop is None:
			pop = np.array(self.total_populations).flatten()

		# N.B. pop[ alpha:(alpha+self.phen_per_patch) ] comes from the fact that the population must be 1 dimensional -> flatten
		# the population for each patch can be found partitioning pop according to self.phen_per_patch
		h = np.array( [ self.base_resources/self.patches[alpha].compute_individuals_per_resource( pop[ alpha*self.phen_per_patch : (alpha+1)*self.phen_per_patch ] ) for alpha in range(self.n_patches) ] )
		h_star = np.array( [ sum(h[:,i])/self.n_patches for i in range(len(self.base_resources)) ] )
		return np.array( [ self.base_resources - self.permeability*( h[alpha] - h_star ) for alpha in range(self.n_patches) ] )
		


	def g_sync(self, t, z):

		#dn/dt = g(n), here we compute g(n)
		if np.any(z<0):
			sys.exit('populations lower than 0')

		temp_res = self.divide_resources(pop = z)
		#print(temp_res)

		return np.array([ z[ alpha*self.phen_per_patch : (alpha+1)*self.phen_per_patch ]*self.patches[alpha].compute_surplus( z[ alpha*self.phen_per_patch : (alpha+1)*self.phen_per_patch ], temp_res[alpha] )\
			/self.patches[alpha].costs for alpha in range(self.n_patches) ]).flatten()
		

	def evolve_sync(self, n_steps=10000, temp_resource = None, plot=False,verb=False):

		#let the system evolve through the coupled differential equations

		if temp_resource is not None:
			self.set_resources(temp_resource)

		if verb==True:
			print(self.resources[0])

		#solve the ODE system
		
		sol = integrate.solve_ivp(self.g, [0, n_steps], np.array(self.total_populations).flatten(), dense_output=True)
		#save and visualise the solution
		t = np.linspace(0, n_steps, int(n_steps))#/10))
		#access to full solution, save z if necessary
		z = sol.sol(t)
		if plot == True:
			[plt.plot(t, z.T[:,i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
			plt.show()
		#return the final population to a list
		self.final_populations =  z.T[-1]
		return self.final_populations



	