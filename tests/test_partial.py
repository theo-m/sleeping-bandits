
import matplotlib.pyplot as plt

import gameoflife
import tests.util
import strategies.partial as partial


def run(params):
  arms = params["arms"]
  env = gameoflife.GameOfLife(arms, params["gameoflife_init"])
  algos = {
    "FTAL": partial.FTAL(env),
    "AUER": partial.AUER(env),
    "Epsilon 0.4": partial.EpsilonGreedy(env, .4),
    "AnnealingEpsilon 0.6": partial.AnnealingEpsilonGreedy(env, .6),
  }

  fig, ax = plt.subplots()
  tests.util.run(params, env, algos, ax, "Sleeping bandits with side information")
