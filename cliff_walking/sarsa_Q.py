import numpy as np
import gymnasium as gym
import time

def run_environment(env, policy=None, n_iterations=1): # 함수 수정이 크게 없습니다.
    n_success = 0
    total_reward = 0

    for i in range(n_iterations):
        state, _ = env.reset()
        episode_reward = 0
        n_steps = 0
        while True:
            if policy is None:
                action = env.action_space.sample()
            else:
                action = policy[state]

            next_state, reward, terminated, truncated, info = env.step(action)
            n_steps += 1
            episode_reward += reward

            if terminated:
                n_success += 1
                print("success {}/{}".format(i + 1, n_iterations))
                print("Episode {}: Step {}, Episode_rewards {}".format(i + 1, n_steps, episode_reward))
                break

            state = next_state

        total_reward += episode_reward

    average_reward = total_reward / n_iterations

    return n_success, average_reward


def sarsa(env, alpha=0.8, gamma=0.98, epsilon=1.0, epsilon_decay=0.001, num_episodes=1000):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []
    cliff_penalties = 0

    for episode in range(num_episodes):
        state, _ = env.reset()

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        episode_reward = 0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if reward == -100:  # Cliff penalty
                cliff_penalties += 1

            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])

            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        epsilon = max(epsilon - epsilon_decay, 0)
        rewards_per_episode.append(episode_reward)

    optimal_policy = np.argmax(Q, axis=1)
    return Q, rewards_per_episode, optimal_policy, cliff_penalties



def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=1000):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_per_episode = []
    cliff_penalties = 0

    for i in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            if reward == -100:  # Cliff penalty
                cliff_penalties += 1

            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode, cliff_penalties


def epsilon_greedy(Q, state, epsilon): # 무작위 탐험
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])


if __name__ == "__main__":
    env_name = 'CliffWalking-v0'
    env = gym.make(env_name)

    start_time = time.time()
    sarsa_Q, sarsa_rewards, sarsa_policy, sarsa_cliff_penalties = sarsa(env)
    sarsa_time = time.time() - start_time
    print(f"SARSA 훈련 시간: {sarsa_time:.2f} 초")
    print(f"SARSA Cliff 패널티 횟수: {sarsa_cliff_penalties}") # 알고리즘의 문제일지 모르겠지만 훈련시간이나 SARSA Cliff 패널티 횟수가 압도적으로 큼

    start_time = time.time()
    q_learning_Q, q_learning_rewards, q_learning_cliff_penalties = q_learning(env)
    q_learning_time = time.time() - start_time
    print(f"Q-learning 훈련 시간: {q_learning_time:.2f} 초")
    print(f"Q-learning Cliff 패널티 횟수: {q_learning_cliff_penalties}")

    env_render = gym.make(env_name, render_mode="human")
    print("SARSA 성능:")
    run_environment(env_render, policy=sarsa_policy, n_iterations=1)
    print("\nQ-learning 성능:") # 리워드가 더 높음
    run_environment(env_render, policy=np.argmax(q_learning_Q, axis=1), n_iterations=1)
    env_render.close()