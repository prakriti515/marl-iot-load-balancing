"""
=============================================================================
run_experiments.py
Runs all NS-3 experiments and coordinates MARL training
Author: Prakriti

FIX: NS3 NS_LOG_UNCOND writes to STDERR not stdout.
     We now capture stderr for parsing.
=============================================================================

USAGE:
    cd ~/ns-allinone-3.43/ns-3.43
    python3 scratch/marl-iot/run_experiments.py
=============================================================================
"""

import subprocess
import os
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NS3_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

sys.path.insert(0, SCRIPT_DIR)
from q_learning_agent import MARLSystem


# =============================================================================
# NS-3 RUNNER
# =============================================================================
class NS3Runner:

    def __init__(self, ns3_dir=NS3_DIR):
        self.ns3_dir    = ns3_dir
        self.ns3_cmd    = os.path.join(ns3_dir, "ns3")
        self.xml_output = os.path.join(ns3_dir, "marl-iot-results.xml")
        self.run_count  = 0
        print(f"[Runner] NS-3 dir : {self.ns3_dir}")

    def run_simulation(self, lb_mode=1, num_iot=10, num_bs=3,
                       sim_time=30.0, failure=False):

        self.run_count += 1
        args = (f"scratch/marl-iot/iot-marl-topology"
                f" --lbMode={lb_mode}"
                f" --numIoT={num_iot}"
                f" --numBS={num_bs}"
                f" --simTime={sim_time}"
                f" --failure={'true' if failure else 'false'}")

        cmd = [self.ns3_cmd, "run", args]
        print(f"\n[NS3 Run #{self.run_count}] lbMode={lb_mode} "
              f"numIoT={num_iot} simTime={sim_time}s "
              f"failure={'YES' if failure else 'NO'}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.ns3_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            # -------------------------------------------------------
            # KEY FIX: NS_LOG_UNCOND writes to STDERR in NS-3
            # We combine both stdout and stderr for parsing
            # -------------------------------------------------------
            combined_output = result.stdout + result.stderr

            if result.returncode != 0:
                # Check if it's a real error or just NS-3 warnings
                if "PAPER METRICS SUMMARY" not in combined_output:
                    print(f"[NS3] ERROR (returncode={result.returncode})")
                    print(f"[NS3] Last error: {result.stderr[-300:]}")
                    return None

            metrics = self._parse_output(combined_output)
            xml_metrics = self._parse_xml(self.xml_output)
            if xml_metrics:
                metrics.update(xml_metrics)

            print(f"[NS3] ✓ Tput={metrics.get('total_throughput_mbps',0):.4f}Mbps "
                  f"Lat={metrics.get('avg_latency_ms',0):.3f}ms "
                  f"Loss={metrics.get('packet_loss_pct',0):.4f}% "
                  f"JFI={metrics.get('jain_fairness_index',0):.4f}")

            return metrics

        except subprocess.TimeoutExpired:
            print("[NS3] ERROR: Simulation timed out!")
            return None
        except Exception as e:
            print(f"[NS3] ERROR: {e}")
            return None

    def _parse_output(self, output):
        """Parse metrics from combined NS-3 stdout+stderr output."""
        metrics = {
            "total_throughput_mbps" : 0.0,
            "avg_latency_ms"        : 0.0,
            "packet_loss_pct"       : 0.0,
            "jain_fairness_index"   : 0.0,
            "total_energy_j"        : 0.0,
            "bs_loads_mbps"         : [],
            "tx_packets"            : 0,
            "rx_packets"            : 0,
            "flow_count"            : 0,
        }

        for line in output.splitlines():
            line = line.strip()
            try:
                if "Total Throughput" in line:
                    metrics["total_throughput_mbps"] = float(line.split(":")[1].split()[0])

                elif "Avg Latency" in line:
                    metrics["avg_latency_ms"] = float(line.split(":")[1].split()[0])

                elif "Packet Loss" in line:
                    metrics["packet_loss_pct"] = float(line.split(":")[1].split()[0])

                elif "Jain Fairness" in line:
                    # Format: "Jain Fairness Index    : 0.980359  (1.0 = perfect)"
                    metrics["jain_fairness_index"] = float(line.split(":")[1].split()[0])

                elif "Total Energy" in line:
                    metrics["total_energy_j"] = float(line.split(":")[1].split()[0])

                elif "Tx Packets" in line:
                    metrics["tx_packets"] = int(line.split(":")[1].strip())

                elif "Rx Packets" in line:
                    metrics["rx_packets"] = int(line.split(":")[1].strip())

                elif "Flows " in line and ":" in line:
                    metrics["flow_count"] = int(line.split(":")[1].strip())

                elif "BS-" in line and "Load" in line and ":" in line:
                    val = float(line.split(":")[1].split()[0])
                    metrics["bs_loads_mbps"].append(val)

            except (ValueError, IndexError):
                continue  # skip lines that don't parse cleanly

        return metrics

    def _parse_xml(self, xml_path):
        """Parse FlowMonitor XML for per-flow detail."""
        if not os.path.exists(xml_path):
            return {}
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            flows = root.find("FlowStats")
            if flows is None:
                return {}

            latencies, throughputs, losses = [], [], []
            tx_total = rx_total = 0

            for flow in flows.findall("Flow"):
                tx       = int(flow.get("txPackets", 0))
                rx       = int(flow.get("rxPackets", 0))
                rx_bytes = int(flow.get("rxBytes", 0))
                delay_s  = flow.get("delaySum", "+0.0ns")

                # Parse NS-3 time string like "+3.51393s" or "+3513930000ns"
                delay_val = 0.0
                try:
                    d = delay_s.strip("+")
                    if d.endswith("ns"):
                        delay_val = float(d[:-2]) / 1e9
                    elif d.endswith("ms"):
                        delay_val = float(d[:-2]) / 1e3
                    elif d.endswith("s"):
                        delay_val = float(d[:-1])
                except:
                    pass

                tx_total += tx
                rx_total += rx

                if rx > 0:
                    latencies.append((delay_val / rx) * 1000.0)
                    throughputs.append((rx_bytes * 8.0) / (29.0 * 1e6))

                if tx > 0:
                    losses.append(100.0 * (tx - rx) / tx)

            return {
                "xml_avg_latency_ms"        : round(np.mean(latencies), 4) if latencies else 0,
                "xml_total_throughput_mbps" : round(sum(throughputs), 4),
                "xml_avg_loss_pct"          : round(np.mean(losses), 4) if losses else 0,
                "xml_tx_total"              : tx_total,
                "xml_rx_total"              : rx_total,
                "xml_flow_count"            : len(list(flows.findall("Flow"))),
            }
        except Exception as e:
            print(f"[XML] Parse warning: {e}")
            return {}


# =============================================================================
# EXPERIMENT MANAGER
# =============================================================================
class ExperimentManager:

    def __init__(self):
        self.runner   = NS3Runner()
        self.results  = {}
        self.save_dir = os.path.join(SCRIPT_DIR, "results")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[ExpMgr] Saving to: {self.save_dir}\n")

    def run_baseline_static(self):
        print("\n" + "="*55)
        print("  EXPERIMENT 1: Static Load Balancing")
        print("="*55)
        r = self.runner.run_simulation(lb_mode=0, num_iot=10, sim_time=30.0)
        if r:
            r["method"] = "Static"
            self.results["static"] = r
            self._save("static", r)
        return r

    def run_baseline_roundrobin(self):
        print("\n" + "="*55)
        print("  EXPERIMENT 2: Round Robin")
        print("="*55)
        r = self.runner.run_simulation(lb_mode=1, num_iot=10, sim_time=30.0)
        if r:
            r["method"] = "Round Robin"
            self.results["round_robin"] = r
            self._save("round_robin", r)
        return r

    def run_scalability_test(self):
        print("\n" + "="*55)
        print("  EXPERIMENT 3: Scalability Test")
        print("="*55)
        iot_counts = [5, 10, 15, 20, 30]
        scalability = []
        for n in iot_counts:
            print(f"\n  → Testing {n} IoT devices...")
            r = self.runner.run_simulation(lb_mode=1, num_iot=n, sim_time=20.0)
            if r:
                r["num_iot"] = n
                scalability.append(r)
        self.results["scalability"] = scalability
        self._save("scalability", scalability)
        return scalability

    def run_fault_tolerance_test(self):
        print("\n" + "="*55)
        print("  EXPERIMENT 4: Fault Tolerance")
        print("="*55)
        print("  Phase 1: Normal operation...")
        normal = self.runner.run_simulation(lb_mode=1, num_iot=10,
                                            sim_time=30.0, failure=False)
        print("  Phase 2: With BS-1 failure at t=10s...")
        failed = self.runner.run_simulation(lb_mode=1, num_iot=10,
                                            sim_time=30.0, failure=True)
        fault = {"normal": normal, "failure": failed}
        if normal and failed:
            fault["latency_impact_ms"] = round(
                failed["avg_latency_ms"] - normal["avg_latency_ms"], 4)
            fault["loss_impact_pct"] = round(
                failed["packet_loss_pct"] - normal["packet_loss_pct"], 4)
            print(f"\n  Latency impact : +{fault['latency_impact_ms']} ms")
            print(f"  Loss impact    : +{fault['loss_impact_pct']} %")
        self.results["fault_tolerance"] = fault
        self._save("fault_tolerance", fault)
        return fault

    def run_marl_training(self, num_episodes=20):
        print("\n" + "="*55)
        print(f"  MARL TRAINING: {num_episodes} Episodes")
        print("="*55)

        marl = MARLSystem(
            num_agents=3,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.90,
        )

        episode_results = []
        for ep in range(1, num_episodes + 1):
            r = marl.run_training_episode(self.runner, ep)
            if r:
                episode_results.append(r)

        marl.save_all(os.path.join(self.save_dir, "marl_saved"))
        marl.print_summary()

        self.results["marl_training"] = episode_results
        self._save("marl_training", episode_results)
        return episode_results, marl

    def _save(self, name, data):
        path = os.path.join(self.save_dir, f"{name}_results.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[SAVED] {path}")

    def print_comparison(self):
        print("\n" + "="*68)
        print("  COMPARISON TABLE — Research Paper Results")
        print("="*68)
        print(f"{'Method':<20} {'Throughput':>13} {'Latency':>11} "
              f"{'Loss':>9} {'JFI':>8}")
        print("-"*68)
        for name, res in self.results.items():
            if isinstance(res, dict) and "total_throughput_mbps" in res:
                print(f"{name:<20} "
                      f"{res['total_throughput_mbps']:>11.4f} Mbps "
                      f"{res['avg_latency_ms']:>9.3f} ms "
                      f"{res['packet_loss_pct']:>7.4f}% "
                      f"{res['jain_fairness_index']:>8.4f}")
        print("="*68)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*55)
    print("  MARL IoT Research — Experiment Runner")
    print("="*55)

    mgr = ExperimentManager()

    print("Select experiments:")
    print("  1. Baselines only (Static + Round Robin) — ~2 min")
    print("  2. Baselines + Scalability + Fault Tolerance — ~10 min")
    print("  3. Full research with MARL training — ~30+ min")
    print()
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        mgr.run_baseline_static()
        mgr.run_baseline_roundrobin()

    elif choice == "2":
        mgr.run_baseline_static()
        mgr.run_baseline_roundrobin()
        mgr.run_scalability_test()
        mgr.run_fault_tolerance_test()

    elif choice == "3":
        mgr.run_baseline_static()
        mgr.run_baseline_roundrobin()
        mgr.run_scalability_test()
        mgr.run_fault_tolerance_test()
        ep = int(input("MARL training episodes? (recommend 20): "))
        mgr.run_marl_training(num_episodes=ep)
    else:
        print("Running baselines only...")
        mgr.run_baseline_static()
        mgr.run_baseline_roundrobin()

    mgr.print_comparison()
    print("\n✅ Done! Run: python3 scratch/marl-iot/plot_graphs.py")