import numpy as np

# dims for action and observation
n_act_dim = 6
n_obs_dim = 9

def disc_to_cont(action):
    throttle_mag = 0.75
    if type(action) == np.ndarray:
        return action
    # main engine
    if action < 3:
        m = -throttle_mag
    elif action < 6:
        m = throttle_mag
    else:
        raise ValueError
    # steering
    if action % 3 == 0:
        s = -throttle_mag
    elif action % 3 == 1:
        s = 0
    else:
        s = throttle_mag
    return np.array([m, s])

def onehot_encode(i, n=n_act_dim):
    x = np.zeros(n)
    x[i] = 1
    return x

def onehot_decode(x):
    # print(x)
    l = np.nonzero(x)[0]
    assert len(l) == 1
    return l[0]