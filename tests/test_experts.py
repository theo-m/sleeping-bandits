
import matplotlib.pyplot as plt

import gameoflife
import tests.util
import strategies.experts as experts


def run(params):
  arms = params["arms"]
  env = gameoflife.GameOfLife(arms, params["gameoflife_init"])
  algos = {
    "FTAL": experts.FTAL(env),
    "Epsilon 0.4": experts.EpsilonGreedy(env, .4),
    "Epsilon 0.1": experts.EpsilonGreedy(env, .1),
    "Annealing Eps 0.6": experts.AnnealingEpsilonGreedy(env, .6)
  }

  fig, ax = plt.subplots()
  tests.util.run(params, env, algos, ax, "Sleeping experts")
