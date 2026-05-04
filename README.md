# MARL-Based Load Balancing in Decentralized IoT Networks

[![NCE Research Grant 2025](https://img.shields.io/badge/NCE-Research%20Grant%202025-blue)]()
[![NS-3](https://img.shields.io/badge/Simulator-NS--3.43-green)]()
[![Python](https://img.shields.io/badge/Language-Python%20%2B%20C%2B%2B-yellow)]()

## Research Title
A Multi-Agent Reinforcement Learning Framework for Adaptive, Scalable,
Fault-Tolerant and Energy-Efficient Load Balancing in Decentralized IoT Networks

## Researcher
- **Name:** Prakriti Shrestha
- **Institution:** National College of Engineering, Tribhuvan University
- **Grant:** NCE Research Grant 2025 — Proposal #10
- **GitHub:** https://github.com/prakriti515

## Project Structure
marl-iot/
├── iot-marl-topology.cc    # NS-3 network simulation (C++)
├── q_learning_agent.py     # MARL Q-learning agents (Python)
├── run_experiments.py      # Experiment runner (Python)
├── plot_graphs.py          # Graph generator (Python)
├── results/                # JSON experiment results
└── graphs/                 # Publication-ready figures (8 graphs)
## Requirements
- NS-3.43 (Network Simulator)
- Python 3.x
- numpy, matplotlib, pandas, scipy

## How to Run

### 1. Run NS-3 Baseline Simulation
```bash
cd ~/ns-allinone-3.43/ns-3.43
./ns3 run scratch/marl-iot/iot-marl-topology
```

### 2. Run All Experiments
```bash
python3 scratch/marl-iot/run_experiments.py
```

### 3. Generate Research Graphs
```bash
python3 scratch/marl-iot/plot_graphs.py
```

### 4. Test MARL Agents Standalone
```bash
python3 scratch/marl-iot/q_learning_agent.py
```

## Experiments Conducted
| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Static LB | All IoT → BS-0, baseline worst case |
| 2 | Round Robin | Even distribution baseline |
| 3 | Scalability | 5, 10, 15, 20, 30 IoT devices |
| 4 | Fault Tolerance | BS-1 failure at t=10s |
| 5 | MARL Training | 20 Q-learning episodes |

## Key Results
| Metric | Value |
|--------|-------|
| Throughput (10 IoT) | 0.524 Mbps |
| Throughput (30 IoT) | 1.525 Mbps (5.8x scaling) |
| Avg End-to-End Latency | 3.51 ms |
| Packet Loss | 0.028% |
| Jain Fairness Index | 0.98 – 1.00 |
| MARL Training Episodes | 20 |
| MARL Agents | 3 (one per Base Station) |

## Network Topology
10 IoT Devices (UDP traffic generators)
↓ CSMA LAN
3 Base Stations (Q-learning MARL agents)
↓ 100Mbps Point-to-Point
1 Gateway (receives all traffic)
## MARL Configuration
- Algorithm: Q-learning (tabular)
- State space: 18 states per agent
- Action space: 2 (process locally / offload)
- Learning rate α: 0.10
- Discount factor γ: 0.90
- Epsilon decay: 1.00 → 0.12 over 20 episodes

## License
Research use only — NCE Research Grant 2025
