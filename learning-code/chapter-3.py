#%% Libraries
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm


#%% 3.2
def posterior_grid_approx(prior, success = 6, tosses = 9):
  grid_points = len(prior)
  p_grid = np.linspace(0, 1, num = grid_points)
  likelihood = stats.binom.pmf(success, tosses, p_grid)
  unstd_posterior = likelihood * prior
  posterior = unstd_posterior / unstd_posterior.sum()
  return p_grid, posterior

#%% 3.3
n, w = 6,9
grid_points = 1000
prior = np.repeat(1, grid_points)
p_grid, posterior = posterior_grid_approx(prior, success = n, tosses = w)
samples = np.random.choice(p_grid, size = 10000, replace = True, p = posterior) 

#%% 3.4
plt.plot(samples, 'o')
plt.xlabel('sample number')
plt.ylabel('proportion water (p)')

#%% 3.5
plt.hist(samples, bins = 50)
plt.xlabel('proportion water (p)')
plt.ylabel('frequency')

#%% Alternative 3.3-3.4
_, (ax0, ax1) = plt.subplots(1,2)
ax0.plot(samples, 'o', alpha = 0.05)
ax1.hist(samples, bins = 50)

#%% 3.6
sum(posterior[p_grid < 0.5])

#%% 3.7
sum(samples < 0.5)/float(len(samples))

#%% 3.8
sum((samples > 0.5) & (samples < 0.75))/float(len(samples))

#%% 3.9
np.percentile(samples, 80)

#%% 3.10
np.percentile(samples, [10, 90])

#%% 3.11
n, w = 3, 3
_grid, posterior = posterior_grid_approx(prior, success = n, tosses = w)
samples = np.random.choice(p_grid, size = 10000, replace = True, p = posterior) 
plt.hist(samples, bins = 50)

#%% 3.12
np.percentile(samples, [25, 75])

#%% 3.13
pm.hpd(samples, alpha = 0.5)

#%% 3.14
p_grid[posterior == max(posterior)]

#%% 3.15
stats.mode(samples)

#%% 3.16
np.mean(samples)
np.median(samples)

#%% 3.17
sum(posterior * abs(0.5 - p_grid))

#%% 3.18
loss = [sum(posterior * abs(p - p_grid)) for p in p_grid]
plt.plot(p_grid, loss, '-')
plt.plot(p_grid[loss == min(loss)], min(loss), 'o')

#%% 3.19
p_grid[loss == min(loss)]

#%% 3.20
stats.binom.pmf(range(3), n = 2, p = 0.7)

#%% 3.21
stats.binom.rvs(size = 1, n = 2, p = 0.7)

#%% 3.22
stats.binom.rvs(size = 10, n = 2, p = 0.7)

#%% 3.23
w = stats.binom.rvs(size = 100000, n = 2, p = 0.7)
[(w == i).mean() for i in range(3)]

#%% 3.24
w = stats.binom.rvs(size = 100000, n = 9, p = 0.7)
plt.hist(w, bins = 9)

#%% 3.25
w = stats.binom.rvs(size = 100000, n = 9, p = 0.6)
plt.hist(w, bins = 9)

#%% 3.26
w = stats.binom.rvs(n = 9, p = samples)
plt.hist(w, bins = 9)
