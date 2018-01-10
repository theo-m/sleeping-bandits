
import numpy as np
import matplotlib.pyplot as plt

import gameoflife
import tests.util
import strategies.bandits as bandits


def run(params):
  arms = params["arms"]
  env = gameoflife.GameOfLife(arms, params["gameoflife_init"])
  algos = {
    "AUER": bandits.AUER(env),
    "Epsilon 0.4": bandits.EpsilonGreedy(env, .4),
    "Epsilon 0.8": bandits.EpsilonGreedy(env, .8),
    "Epsilon 0.1": bandits.EpsilonGreedy(env, .1)
  }

  fig, ax = plt.subplots()
  # ax.plot(np.repeat(bandits.oracle(env, params["horizon"]), params["horizon"]), label="oracle")
  tests.util.run(params, env, algos, ax, "Sleeping bandits")
