import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)


def gaussian_policy(sigma):
    return np.random.normal(loc=0, scale=sigma)


def compute_reward(target, observation):
    reward = -(target - observation) ** 2
    return reward


def has_converged(final_u):
    if final_u > 1.99 and final_u < 2.01:
        return True
    else:
        return False


def reinforce(sigma, target, lr, lr_i, num_steps, num_episodes):
    total_mat = np.zeros((num_episodes, num_steps))
    rewards_mat = np.zeros((num_episodes, num_steps))
    convergence_bool = []
    for episode in range(num_episodes):
        u_list = []
        rewards = []
        u = 0

        for step in range(num_steps):
            y = u + gaussian_policy(sigma)
            reward = compute_reward(target, y)
            rewards.append(reward)
            du = (1 / (sigma ** 2)) * lr * reward * (y - u)
            u = u + du
            u_list.append(u)
        convergence_bool.append(has_converged(u_list[-1]))
        rewards_mat[episode] = rewards
        total_mat[episode] = u_list

    # plotting
    plt.subplot(2, 3, lr_i + 1)
    for i in range(5):
        plt.plot(total_mat[i], label=f"run={i}")
    avg_run = np.mean(total_mat, axis=0)
    plt.plot(avg_run, label=f"mean")
    plt.legend()
    plt.title(f"lr={lr:.3e}")

    return (np.sum(convergence_bool) / 200) * 100


if __name__ == "__main__":
    TARGET = 2.0  # The true hidden target
    SIGMA = 0.1  # Standard deviation of the noise in the player's aiming
    LR = np.logspace(-5, -2, 6)
    NUM_STEPS = 10000
    NUM_EPISODES = 200

    convergence_rates = []
    for i, learning_rate in tqdm(enumerate(LR)):
        convergence = reinforce(SIGMA, TARGET, learning_rate, i, NUM_STEPS, NUM_EPISODES)
        convergence_rates.append(convergence)
    plt.show()

    plt.plot(LR, convergence_rates)
    plt.title("Probability of convergence as function of learning rate")
    plt.xlabel("Learning rate")
    plt.ylabel("Probability of convergence")
    plt.xscale("log")
    plt.show()
