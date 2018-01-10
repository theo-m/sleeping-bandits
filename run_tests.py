
import numpy as np

import arms.stochastic as sto

import tests.viz
import tests.test_experts
import tests.test_partial
import tests.test_bandits


""" Parameters """
params = {
  "board_size": 50,
  "horizon": 10000,
  "nbepochs": 10,
  "gameoflife_init": .5,
  "mus": 1,
  "sigmas": 5
}

arms = list()
for _ in range(params["board_size"]):
  for _ in range(params["board_size"]):
    mu = np.random.rand() * params["mus"]
    sigma = params["sigmas"]
    arms.append(sto.ArmGaussian1D(mu, sigma))

params["arms"] = arms


""" Tests """

# print("experts")
# tests.test_experts.run(params)

print("pure bandits")
tests.test_bandits.run(params)

# print("partial obs bandits")
# tests.test_partial.run(params)

# print("viz, experts")
# tests.viz.run(params)
