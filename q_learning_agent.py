"""
=============================================================================
q_learning_agent.py
MARL Q-Learning Agent for IoT Load Balancing
Research: A Multi-Agent Reinforcement Learning Framework for Adaptive,
          Scalable, Fault-Tolerant and Energy-Efficient Load Balancing
          in Decentralized IoT Networks
Author: Prakriti
=============================================================================

HOW IT WORKS:
  - Each Base Station is one Q-learning Agent
  - State  = (my_load_level, neighbor_load_level, congestion_flag)
  - Action = 0: Process locally | 1: Offload to neighbor with least load
  - Reward = based on latency + energy + fairness improvement
  - Agents learn independently but observe neighbors (decentralized MARL)
=============================================================================
"""

import numpy as np
import random
import json
import os


# =============================================================================
# SINGLE Q-LEARNING AGENT (represents one Base Station)
# =============================================================================
class QLearningAgent:
    """
    One Q-learning agent = One Base Station in the IoT network.

    State space:
        - my_load    : 0=low, 1=medium, 2=high  (3 levels)
        - neigh_load : 0=low, 1=medium, 2=high  (3 levels of avg neighbor)
        - congested  : 0=no,  1=yes             (2 levels)
        Total states = 3 x 3 x 2 = 18 states

    Action space:
        - 0 = Process traffic locally
        - 1 = Offload to least-loaded neighbor
    """

    def __init__(self, agent_id, num_actions=2,
                 learning_rate=0.1,
                 discount_factor=0.9,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995):

        self.agent_id       = agent_id
        self.num_actions    = num_actions
        self.lr             = learning_rate
        self.gamma          = discount_factor
        self.epsilon        = epsilon        # exploration rate
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay

        # State dimensions: load_self x load_neighbor x congestion
        self.state_dims = (3, 3, 2)
        self.num_states = 3 * 3 * 2  # = 18

        # Q-table: shape (3, 3, 2, num_actions)
        self.q_table = np.zeros((*self.state_dims, num_actions))

        # Tracking for analysis
        self.episode_rewards = []
        self.total_steps     = 0
        self.actions_taken   = {0: 0, 1: 0}  # count per action

    def get_state(self, my_load, neighbor_loads, congestion_threshold=0.7):
        """
        Convert raw load values into discrete state tuple.

        Args:
            my_load          : float, current load ratio (0.0 - 1.0)
            neighbor_loads   : list of floats, neighbor load ratios
            congestion_threshold : float, above this = congested

        Returns:
            tuple: (my_load_level, avg_neigh_level, congested)
        """
        def discretize(load):
            if load < 0.33:   return 0   # low
            elif load < 0.66: return 1   # medium
            else:             return 2   # high

        my_level   = discretize(my_load)
        neigh_avg  = np.mean(neighbor_loads) if neighbor_loads else 0.0
        neigh_level = discretize(neigh_avg)
        congested  = 1 if my_load > congestion_threshold else 0

        return (my_level, neigh_level, congested)

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        High epsilon = explore randomly (early training)
        Low epsilon  = exploit best known action (later training)
        """
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.randint(0, self.num_actions - 1)
        else:
            # Exploit: best known action from Q-table
            action = np.argmax(self.q_table[state])

        self.actions_taken[action] += 1
        self.total_steps += 1
        return action

    def compute_reward(self, latency_ms, throughput_mbps,
                       energy_j, fairness_index,
                       packet_loss_pct):
        """
        Reward function — designed for your 6 research metrics.

        Reward is POSITIVE when:
          - Latency is low
          - Throughput is high
          - Energy is low
          - Fairness is high (close to 1.0)
          - Packet loss is low

        All terms normalized and weighted.
        """
        # Normalize each metric (target ranges from your NS-3 output)
        latency_norm    = max(0, 1.0 - latency_ms / 50.0)     # 0ms=best, 50ms=worst
        throughput_norm = min(1.0, throughput_mbps / 0.1)      # 0.1 Mbps = target
        energy_norm     = max(0, 1.0 - energy_j / 10.0)        # 0J=best, 10J=worst
        fairness_norm   = fairness_index                        # already 0-1
        loss_norm       = max(0, 1.0 - packet_loss_pct / 10.0) # 0%=best, 10%=worst

        # Weighted reward (weights reflect research priorities)
        reward = (
            0.30 * latency_norm    +   # most important for IoT
            0.25 * throughput_norm +
            0.20 * fairness_norm   +
            0.15 * energy_norm     +
            0.10 * loss_norm
        )

        return reward

    def learn(self, state, action, reward, next_state):
        """
        Q-table update using Bellman equation:
        Q(s,a) = Q(s,a) + lr * [reward + gamma * max(Q(s')) - Q(s,a)]
        """
        current_q  = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q      = current_q + self.lr * (
                        reward + self.gamma * max_next_q - current_q
                     )
        self.q_table[state][action] = new_q

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return new_q

    def save_q_table(self, filepath):
        """Save Q-table to file for later use."""
        np.save(filepath, self.q_table)
        print(f"[Agent-{self.agent_id}] Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        """Load previously trained Q-table."""
        if os.path.exists(filepath):
            self.q_table = np.load(filepath)
            self.epsilon = self.epsilon_min  # no more exploration
            print(f"[Agent-{self.agent_id}] Q-table loaded from {filepath}")
        else:
            print(f"[Agent-{self.agent_id}] No saved Q-table found, starting fresh")

    def get_best_action(self, state):
        """Get best action without exploration (for deployment)."""
        return np.argmax(self.q_table[state])

    def get_stats(self):
        """Return agent statistics."""
        total = self.total_steps
        return {
            "agent_id"       : self.agent_id,
            "total_steps"    : total,
            "epsilon"        : round(self.epsilon, 4),
            "action_0_pct"   : round(100*self.actions_taken[0]/max(total,1), 1),
            "action_1_pct"   : round(100*self.actions_taken[1]/max(total,1), 1),
            "q_table_max"    : round(float(np.max(self.q_table)), 4),
            "q_table_mean"   : round(float(np.mean(self.q_table)), 4),
        }


# =============================================================================
# MULTI-AGENT SYSTEM (all Base Stations together)
# =============================================================================
class MARLSystem:
    """
    Manages all Q-learning agents (Base Stations) together.
    Implements decentralized MARL — agents observe neighbors
    but make decisions independently.
    """

    def __init__(self, num_agents=3, **agent_kwargs):
        self.num_agents = num_agents
        self.agents = [
            QLearningAgent(agent_id=i, **agent_kwargs)
            for i in range(num_agents)
        ]
        self.episode_count = 0
        self.history = []  # stores per-episode metrics

        print(f"[MARL] Initialized {num_agents} Q-learning agents")
        print(f"[MARL] State space: 18 states per agent")
        print(f"[MARL] Action space: 2 actions per agent")
        print(f"[MARL] Total Q-values: {num_agents} x 18 x 2 = {num_agents*36}")

    def get_all_actions(self, loads, congestion_threshold=0.7):
        """
        Get action for each agent based on current network state.

        Args:
            loads : list of float, current load ratio for each BS

        Returns:
            actions : list of int, action for each BS
            states  : list of tuple, state for each BS
        """
        actions = []
        states  = []

        for i, agent in enumerate(self.agents):
            # Neighbors = all other BS loads
            neighbor_loads = [loads[j] for j in range(self.num_agents) if j != i]

            state  = agent.get_state(loads[i], neighbor_loads,
                                     congestion_threshold)
            action = agent.choose_action(state)

            states.append(state)
            actions.append(action)

        return actions, states

    def update_all(self, states, actions, metrics_per_agent, next_loads):
        """
        Update Q-tables for all agents after observing results.

        Args:
            states          : list of state tuples (before action)
            actions         : list of int actions taken
            metrics_per_agent : list of dicts with latency/throughput/etc
            next_loads      : list of float, new load ratios after actions
        """
        rewards = []

        for i, agent in enumerate(self.agents):
            m = metrics_per_agent[i]

            # Compute reward from this agent's observed metrics
            reward = agent.compute_reward(
                latency_ms      = m.get("latency_ms",      5.0),
                throughput_mbps = m.get("throughput_mbps", 0.05),
                energy_j        = m.get("energy_j",        1.0),
                fairness_index  = m.get("fairness_index",  0.9),
                packet_loss_pct = m.get("packet_loss_pct", 0.1),
            )

            # Next state
            neighbor_next = [next_loads[j]
                             for j in range(self.num_agents) if j != i]
            next_state = agent.get_state(next_loads[i], neighbor_next)

            # Learn
            agent.learn(states[i], actions[i], reward, next_state)
            rewards.append(reward)

        return rewards

    def run_training_episode(self, ns3_runner, episode_num):
        """
        Run one training episode:
        1. Run NS-3 simulation
        2. Parse results
        3. Update Q-tables
        4. Return metrics

        Args:
            ns3_runner : NS3Runner instance (from run_experiments.py)
            episode_num : int

        Returns:
            dict of episode metrics
        """
        self.episode_count += 1

        # --- Get current load estimates (start of episode) ---
        # Initially uniform, then based on previous episode
        if len(self.history) > 0:
            last = self.history[-1]
            loads = last.get("bs_loads_normalized",
                             [0.5] * self.num_agents)
        else:
            loads = [0.5] * self.num_agents

        # --- Decide actions ---
        actions, states = self.get_all_actions(loads)

        # --- Run NS-3 with current LB mode ---
        # MARL = mode 1 (round robin as foundation, Q-learning adjusts weights)
        lb_mode = 1
        results = ns3_runner.run_simulation(
            lb_mode  = lb_mode,
            num_iot  = 10,
            sim_time = 30.0,
            failure  = False
        )

        if results is None:
            print(f"[MARL] Episode {episode_num}: NS-3 run failed, skipping")
            return None

        # --- Build per-agent metrics from NS-3 output ---
        metrics_per_agent = []
        for i in range(self.num_agents):
            metrics_per_agent.append({
                "latency_ms"      : results["avg_latency_ms"],
                "throughput_mbps" : results["total_throughput_mbps"] / self.num_agents,
                "energy_j"        : results.get("total_energy_j", 1.0) / self.num_agents,
                "fairness_index"  : results["jain_fairness_index"],
                "packet_loss_pct" : results["packet_loss_pct"],
            })

        # --- Compute next loads ---
        bs_loads = results.get("bs_loads_mbps", [0.1] * self.num_agents)
        max_load = max(bs_loads) if max(bs_loads) > 0 else 1.0
        next_loads = [l / max_load for l in bs_loads]

        # --- Update Q-tables ---
        rewards = self.update_all(states, actions, metrics_per_agent, next_loads)

        # --- Store episode record ---
        episode_data = {
            "episode"              : episode_num,
            "actions"              : actions,
            "avg_reward"           : round(float(np.mean(rewards)), 4),
            "avg_latency_ms"       : results["avg_latency_ms"],
            "total_throughput_mbps": results["total_throughput_mbps"],
            "packet_loss_pct"      : results["packet_loss_pct"],
            "jain_fairness_index"  : results["jain_fairness_index"],
            "total_energy_j"       : results.get("total_energy_j", 0),
            "bs_loads_normalized"  : next_loads,
            "epsilon"              : round(self.agents[0].epsilon, 4),
        }
        self.history.append(episode_data)

        print(f"[Episode {episode_num:03d}] "
              f"Reward={episode_data['avg_reward']:.4f} | "
              f"Lat={results['avg_latency_ms']:.2f}ms | "
              f"Tput={results['total_throughput_mbps']:.4f}Mbps | "
              f"Loss={results['packet_loss_pct']:.3f}% | "
              f"JFI={results['jain_fairness_index']:.4f} | "
              f"ε={episode_data['epsilon']}")

        return episode_data

    def save_all(self, save_dir="marl_saved"):
        """Save all Q-tables and training history."""
        os.makedirs(save_dir, exist_ok=True)
        for agent in self.agents:
            agent.save_q_table(f"{save_dir}/agent_{agent.agent_id}_qtable.npy")
        with open(f"{save_dir}/training_history.json", "w") as f:
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super().default(obj)
            json.dump(self.history, f, indent=2, cls=NumpyEncoder)
        print(f"[MARL] All agents saved to {save_dir}/")

    def load_all(self, save_dir="marl_saved"):
        """Load all Q-tables."""
        for agent in self.agents:
            agent.load_q_table(f"{save_dir}/agent_{agent.agent_id}_qtable.npy")

    def print_summary(self):
        """Print Q-table summary for all agents."""
        print("\n" + "="*50)
        print("  MARL AGENT SUMMARY")
        print("="*50)
        for agent in self.agents:
            stats = agent.get_stats()
            print(f"\nAgent (BS-{stats['agent_id']}):")
            print(f"  Total steps    : {stats['total_steps']}")
            print(f"  Epsilon        : {stats['epsilon']}")
            print(f"  Action-0 (local)   : {stats['action_0_pct']}%")
            print(f"  Action-1 (offload) : {stats['action_1_pct']}%")
            print(f"  Max Q-value    : {stats['q_table_max']}")
            print(f"  Mean Q-value   : {stats['q_table_mean']}")
        print("="*50)


# =============================================================================
# QUICK TEST — run this file directly to verify agents work
# =============================================================================
if __name__ == "__main__":
    print("="*50)
    print("  Testing MARL Q-Learning Agents")
    print("="*50)

    # Create 3 agents (one per Base Station)
    marl = MARLSystem(num_agents=3, learning_rate=0.1,
                      discount_factor=0.9, epsilon=1.0)

    # Simulate 100 learning steps with fake data
    print("\n[TEST] Simulating 100 learning steps with dummy data...")
    for step in range(100):
        # Fake load values (would come from NS-3 in real training)
        fake_loads = [random.uniform(0.2, 0.9) for _ in range(3)]

        # Get actions
        actions, states = marl.get_all_actions(fake_loads)

        # Fake metrics (would come from NS-3 results)
        fake_metrics = [{
            "latency_ms"      : random.uniform(2, 15),
            "throughput_mbps" : random.uniform(0.04, 0.08),
            "energy_j"        : random.uniform(0.1, 2.0),
            "fairness_index"  : random.uniform(0.7, 1.0),
            "packet_loss_pct" : random.uniform(0, 2),
        } for _ in range(3)]

        fake_next_loads = [random.uniform(0.2, 0.9) for _ in range(3)]

        # Update Q-tables
        rewards = marl.update_all(states, actions, fake_metrics, fake_next_loads)

        if (step+1) % 20 == 0:
            print(f"  Step {step+1:3d} | "
                  f"Actions={actions} | "
                  f"Avg Reward={np.mean(rewards):.4f} | "
                  f"ε={marl.agents[0].epsilon:.4f}")

    # Print final Q-tables
    marl.print_summary()

    print("\n[TEST] Q-learning agent working correctly!")
    print("[NEXT] Run: python3 run_experiments.py")