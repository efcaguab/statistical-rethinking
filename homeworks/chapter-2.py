#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pymc3 as pm

#%% 2.1
ways = np.array([0,3,8,9,0])
ways / float(ways.sum())
 
#%% 2.2
stats.binom.pmf(6, n = 9, p = 0.99999)

#%% 2.3 & 2.4
def posterior_grid_approx(prior, success = 6, tosses = 9, grid_points = 20):
  p_grid = np.linspace(0, 1, num = grid_points)
  likelihood = stats.binom.pmf(success, tosses, p_grid)
  unstd_posterior = likelihood * prior
  posterior = unstd_posterior / unstd_posterior.sum()
  return p_grid, posterior

def plot_posterior(p_grid, posterior, w, n, grid_points = 20):
  plt.plot(p_grid, posterior, 'o-', label='success = {}\ntosses = {}'.format(w, n))
  plt.xlabel('probability of water', fontsize=14)
  plt.ylabel('posterior probability')
  plt.title('{} points'.format(grid_points))
  plt.legend(loc = 0)

grid_points, w, n = 50, 6, 9
prior = np.repeat(1, grid_points)
p_grid, posterior = posterior_grid_approx(prior, w, n, grid_points)
plot_posterior(p_grid, posterior, w, n, grid_points)

#%% 2.5
# first prior
prior = np.linspace(0,1, num = grid_points) >= 0.5
prior = prior.astype(int)
p_grid, posterior = posterior_grid_approx(prior, w, n, grid_points)
plot_posterior(p_grid, posterior, w, n, grid_points)
# second prior
prior = np.linspace(0,1,num= grid_points) - 0.5
prior = np.exp(abs(prior) * (-5)) 
p_grid, posterior = posterior_grid_approx(prior, w, n, grid_points)
plot_posterior(p_grid, posterior, w, n, grid_points)

#%% 2.6
data = np.repeat((0, 1), (3, 6))
with pm.Model() as normal_approximation:
  p = pm.Uniform('p', 0, 1)
  w = pm.Binomial('w', n = len(data), p = p, observed = data.sum())
mean_q = pm.find_MAP(model = normal_approximation)
std_q = ((1/pm.find_hessian(mean_q, vars=[p], model = normal_approximation))**0.5)[0]

#%% 2.7
# quadratic approximation
w, n = 6, 9
x = np.linspace(0,1)
norm = stats.norm.pdf(x, mean_q['p'], std_q)
plt.plot(x, norm, label = 'Quadratic approximation')
# exact solution
exact = stats.beta.pdf(x, w + 1, n - w + 1)
plt.plot(x, exact, label = 'True posterior')
plt.legend(loc = 0, fontsize = 13)
