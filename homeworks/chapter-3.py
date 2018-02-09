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

#%% Exercises
def posterior_grid_approx(prior, success = 6, tosses = 9):
  grid_points = len(prior)
  p_grid = np.linspace(0, 1, num = grid_points)
  likelihood = stats.binom.pmf(success, tosses, p_grid)
  unstd_posterior = likelihood * prior
  posterior = unstd_posterior / unstd_posterior.sum()
  return p_grid, posterior

n, w = 6,9
grid_points = 1000
prior = np.repeat(1, grid_points)
p_grid, posterior = posterior_grid_approx(prior, success = n, tosses = w)
np.random.seed(100)
samples = np.random.choice(p_grid, size = 10000, replace = True, p = posterior) 
plt.plot(p_grid, posterior)
#%% 3E1
np.mean(samples < 0.2)

#%% 3E2
np.mean(samples >0.8)

#%% 3E3
np.mean((samples > 0.2) & (samples < 0.8))

#%% 3E4
np.percentile(samples, 20)

#%% 3E5
np.percentile(samples, 80)

#%% 3E6
pm.hpd(samples, alpha = 1-0.66)

#%% 3E7
np.percentile(samples, [66/2/2, 100-66/2/2])

#%% 3M1
n, w = 8,15
grid_points = 1000
prior = np.repeat(1, grid_points)
p_grid, posterior = posterior_grid_approx(prior, success = n, tosses = w)
plt.plot(p_grid, posterior)

#%% 3M2
np.random.seed(100)
samples = np.random.choice(p_grid, size = 10000, replace = True, p = posterior) 
pm.hpd(samples, alpha = 0.1)

#%% 3M3
n = 15
w = stats.binom.rvs(n = n, p = samples)
plt.hist(w, bins = n)
np.mean(w == 8)

#%% 3M4
w = stats.binom.rvs(n = 9, p = samples)
np.mean(w == 6)

#%% 3M5
n, w = 8,15
grid_points = 1000
prior = np.linspace(0,1, num = grid_points) >= 0.5
prior = prior.astype(int)
p_grid, posterior = posterior_grid_approx(prior, success = n, tosses = w)
plt.plot(p_grid, posterior)
np.random.seed(100)
samples = np.random.choice(p_grid, size = 10000, replace = True, p = posterior) 
pm.hpd(samples, alpha = 0.1)

n = 15
w = stats.binom.rvs(n = n, p = samples)
plt.hist(w, bins = n)
np.mean(w == 8)

#%% 3H1
birth1 = np.array([1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0, 0,0,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0, 1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,1,0,1,1,1,0,1,1,1,1])
birth2 = np.array([0,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,
1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,1,
0,0,0,1,1,1,0,0,0,0])

n, w = sum(birth1) + sum(birth2) ,len(birth1) + len(birth2)
grid_points = 1000
prior = np.repeat(1, grid_points)
p_grid, posterior = posterior_grid_approx(prior, success = n, tosses = w)
plt.plot(p_grid, posterior)
p_grid[posterior == max(posterior)]

#%% 3H2
samples = np.random.choice(p_grid, size = 10000, replace = True, p = posterior) 
alphas = [0.50, 1-0.89, 1-0.97]
[pm.hpd(samples, alpha = i) for i in alphas]

#%% 3H3
w = stats.binom.rvs(200, samples)
plt.hist(w, bins = range(200))
plt.plot(n, 0, 'o')

#%% 3H4
w = stats.binom.rvs(100, samples)
plt.hist(w, bins = range(100))
plt.plot(sum(birth1), 0, 'o')

#%% 3H5
n_girls_first = sum(birth1 == 0)
w_girls = stats.binom.rvs(n_girls_first, samples)
plt.hist(w_girls, bins = range(n_girls_first))
n_boys_sec_empirical = sum((birth1 == 0) & (birth2 == 1))
plt.plot(n_boys_sec_empirical, 0, 'o')