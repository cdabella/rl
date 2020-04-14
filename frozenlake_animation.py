
import numpy as np
import pandas as pd

import gym
import hiive.mdptoolbox.mdp as mdp

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation

from joblib import Parallel, delayed


def makeEnv():
    env = gym.make('FrozenLake8x8-v0')
    env.reset()

    rewards = {'H': -1, 'G': 1}
    movePenalty = -0.01

    numActions, numSpaces = env.nA, env.nS
    vectorMap = ''
    for row in env.desc:
        for val in row:
            vectorMap += val.decode('UTF-8')

    P = np.zeros([numActions, numSpaces, numSpaces])
    R = np.zeros([numSpaces, numActions])
    for s in range(numSpaces):
        for a in range(numActions):
            transitions = env.P[s][a]
            for p_trans, next_s, reward, done in transitions:
                P[a, s, next_s] += p_trans
                R[s, a] = rewards.get(vectorMap[next_s], movePenalty)
            P[a, s, :] /= np.sum(P[a, s, :])

    env.reset()

    return env, P, R


def getVIFrames(env, P, R):
    vi = mdp.ValueIteration(P, R, 0.9, epsilon=0.01)
    vi.run()
    run_stats = vi.run_stats
    return [step['Value'].reshape(env.nrow, env.ncol) for step in run_stats]

def getQLFrames(env, P, R):
    # epsilon_min=0.1
    # ln(0.1)/ln(epsilon_decay) == iteration of e_min
    # alpha_min=0.001
    # ln(0.001)/ln(alpha_decay) == iteration of a_min
    np.random.seed(1)
    ql = mdp.QLearning(P, R, 0.9, n_iter=100000, alpha_decay=0.99999, epsilon_decay=0.9)
    ql.setVerbose()
    run_stats = ql.run()
    return [step['Value'].reshape(env.nrow, env.ncol) for step in run_stats]


def main():
    env, P, R = makeEnv()
    # frames = getVIFrames(env, P, R)

    frames = getQLFrames(env, P, R)

    fig = plt.figure()
    ax = plt.axes()
    im = plt.imshow(frames[0], interpolation='none', norm=colors.SymLogNorm(linthresh=0.02, base=10))

    def init():
        im.set_data(frames[0])
        # return plt.imshow(frames[0], interpolation='none', norm=colors.SymLogNorm(linthresh=0.02, base=10)),

    def animate(i):
        print(f'Frame {i}')
        # return plt.imshow(frames[i], interpolation='none', norm=colors.SymLogNorm(linthresh=0.02, base=10))
        im.set_data(frames[i])
        return im,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames),
                                   interval=1,
                                   # blit=True,
                                   )
    anim.save('frozenlake_ql_animation.mp4', fps=30, ) # extra_args=['-vcodec', 'libx264'])

    plt.show()


if __name__ == '__main__':
    main()
