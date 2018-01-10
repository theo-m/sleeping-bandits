
import numpy as np
import matplotlib.pyplot as plt

from strategies.abstract import RandomStrategy


def run_regret(player, horizon):

  pulls, rewards, params, history = player.run(horizon)
  regret = player.regret(history, pulls)
  full = np.zeros(horizon)
  full[:len(regret)] = regret

  return np.cumsum(full)


def run(params, env, players, ax, title):

  nbepochs = params["nbepochs"]
  horizon = params["horizon"]

  players["random"] = RandomStrategy(env)  # add systematically the random strat

  algosregrets = {algoname: [] for algoname in players.keys()}
  init_board = env.board.copy()

  for algoname, player in players.items():

    for epoch in range(nbepochs):

      env.board = init_board.copy()
      cumregret = run_regret(player, horizon)
      algosregrets[algoname].append(cumregret)

      env.reinit()
      player.reinit()

    ax.plot(np.array(algosregrets[algoname]).mean(0), label=algoname)

  paramstr = "mus: {}, sigmas: {}, nbepochs: {}, nbarms: {}".format(
    params["mus"], params["sigmas"], params["nbepochs"], env.nbarms)

  ax.legend()
  ax.set_title("{}\n{}".format(title, paramstr))
  ax.set_xlabel("Horizon")
  ax.set_ylabel("Cumulative regret")
  plt.tight_layout()
  plt.savefig("../images/%i.png" % np.random.randint(1e3, 1e4))
  plt.show()
