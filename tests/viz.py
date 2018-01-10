
import numpy as np
import matplotlib.pyplot as plt

import gameoflife
import strategies.experts as experts


def run(params):

  arms = params["arms"]
  env = gameoflife.GameOfLife(arms, params["gameoflife_init"])

  horizon = params["horizon"]

  fig, ax = plt.subplots()
  player = experts.EpsilonGreedy(env, epsilon=.4)
  player.run(horizon, render=True)
