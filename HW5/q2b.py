import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def gaussian_policy(sigma):
    return np.random.normal(loc=0, scale=sigma)


def compute_reward(target, observation):
    if observation > target:
        reward = -(target - observation) ** 2
    else:
        reward = -2 * (target - observation) ** 2
    return reward


def reinforce(sigma, target, lr, i, num_steps, num_episodes):
    total_mat = np.zeros((num_episodes, num_steps))
    rewards_mat = np.zeros((num_episodes, num_steps))
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
        rewards_mat[episode] = rewards
        total_mat[episode] = u_list

    avg_run = np.mean(total_mat, axis=0)

    plt.subplot(1, 2, i + 1)
    for i in range(5):
        plt.plot(rewards_mat[i], label=f"run={i}")
    avg_reward = np.mean(rewards_mat, axis=0)
    plt.plot(avg_reward, label=f"mean")
    plt.legend()
    plt.title(f"Reward vs step #, sigma={sigma}")

    return avg_run[-1]


if __name__ == "__main__":
    TARGET = 2.0  # The true hidden target
    SIGMA = [0.1, 0.5]  # Standard deviation of the noise in the player's aiming
    LR = 0.001
    NUM_STEPS = 10000
    NUM_EPISODES = 200

    convergence_vals = []
    for i, sigma in enumerate(SIGMA):
        final_u = reinforce(sigma, TARGET, LR, i, NUM_STEPS, NUM_EPISODES)
        print(f"sigma={sigma}, final_u={final_u}")
        convergence_vals.append(final_u)
    plt.show()

    # plot reward vs y
    y_vals = np.linspace(-1, 5, 100)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    rewards = []
    for y in y_vals:
        reward = compute_reward(TARGET, y)
        rewards.append(reward)
    plt.plot(y_vals, rewards)
    for i, val in enumerate(convergence_vals):
        plt.axvline(x=val, color=colors[i], linestyle='--', label=f"sigma={SIGMA[i]}, u={np.round(val, decimals=3)}")
    plt.title(f"Reward vs Y - {NUM_STEPS} steps")
    plt.legend()
    plt.xlabel("Y")
    plt.ylabel("Reward")
    plt.show()
