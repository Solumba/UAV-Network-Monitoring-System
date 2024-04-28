import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import time
import csv
import os
import time
from datetime import datetime
from neo4j import GraphDatabase
from pymongo import MongoClient

# Neo4j configuration
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "Shady5000$")


class UAVNetworkSimulation:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def hello_world(self):
        num = 10
        return num

    def hello_number(self):
        num = self.hello_world()
        print("hello_world" + str(num))

    def create_initial_graph(
        self, num_uavs, connection_range, ground_station_pos, backbone_range
    ):
        # Create undirected graph
        G = nx.Graph()

        # Define typical throughput and latency ranges (for example purposes)
        typical_throughput = (50, 100)  # Mbps
        typical_latency = (10, 50)  # Milliseconds
        typical_battery = (100, 100)  # %

        # Add UAV nodes with properties
        for i in range(num_uavs):
            throughput = random.randint(*typical_throughput)
            latency = random.randint(*typical_latency)
            battery = random.randint(*typical_battery)
            G.add_node(
                i,
                pos=(random.randint(0, 100), random.randint(0, 100)),
                throughput=throughput,
                latency=latency,
                battery=battery,
            )

        # Add The Backbone UAV
        backbone_uav_id = num_uavs
        G.add_node(
            backbone_uav_id,
            pos=(random.randint(0, 100), random.randint(0, 100)),
            is_backbone=True,
            throughput=throughput,
            latency=latency,
            battery=battery,
        )

        # Connect Backbone UAV to other UAVs Based on Euclidean Distance
        for i in range(num_uavs):
            pos_i = G.nodes[i]["pos"]
            pos_backbone = G.nodes[backbone_uav_id]["pos"]
            distance = np.linalg.norm(np.array(pos_i) - np.array(pos_backbone))
            if distance <= backbone_range:
                G.add_edge(i, backbone_uav_id, weight=distance)

        # Add and connect the Ground Station To The Backbone Only
        ground_station_id = num_uavs + 1
        G.add_node(
            ground_station_id,
            pos=ground_station_pos,
            is_ground_station=True,
            throughput=throughput,
            latency=latency,
            battery=battery,
        )
        G.add_edge(backbone_uav_id, ground_station_id)

        # Add edges between UAVs within connection range
        for i in range(num_uavs):
            for j in range(i + 1, num_uavs):
                pos_i = G.nodes[i]["pos"]
                pos_j = G.nodes[j]["pos"]
                distance = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                if distance <= connection_range:
                    G.add_edge(i, j, weight=distance)

        return G

    # Upload Current Graph To Neo4j
    def upload_to_neo4j(self, G):
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")

            # Add nodes
            for node in G.nodes:
                is_backbone = G.nodes[node].get("is_backbone", False)
                is_ground_station = G.nodes[node].get("is_ground_station", False)
                is_attack_node = G.nodes[node].get("is_attack_node", False)

                if is_backbone:
                    session.run(
                        f"""
                        CREATE (b:BackboneUAV {{id: $id, pos: $pos, uavType: $uavType,
                            throughput: $throughput, latency: $latency, battery: $battery  + '%'}})
                        """,  # noqa: F541
                        id=node,
                        pos=G.nodes[node]["pos"],
                        uavType="Backbone UAV",
                        throughput=G.nodes[node]["throughput"],
                        latency=G.nodes[node]["latency"],
                        battery=G.nodes[node]["battery"],
                    )
                elif is_ground_station:
                    session.run(
                        f"""
                        CREATE (g:GroundStation {{id: $id, pos: $pos, gsName: $GS, battery: $battery + '%'}})
                        """,
                        id=node,
                        pos=G.nodes[node]["pos"],
                        GS="Ground Station",
                        battery=G.nodes[node]["battery"],
                    )
                else:
                    session.run(
                        f"""
                        CREATE (u:UAV {{id: $id, pos: $pos, uavType: $uavType, uavName: $uavName,
                            throughput: $throughput + 'mb/s', latency: $latency + 'ms', battery: $battery + '%'}})
                        """,
                        id=node,
                        pos=G.nodes[node]["pos"],
                        uavType="UAV",
                        uavName="UAV" + str(node),
                        throughput=G.nodes[node]["throughput"],
                        latency=G.nodes[node]["latency"],
                        battery=G.nodes[node]["battery"],
                    )

            # Add relationships
            for source, target in G.edges:
                session.run(
                    """
                    MATCH (a), (b)
                    WHERE a.id = $source AND b.id = $target
                    CREATE (a)-[:COMMUNICATES_WITH]->(b)
                    """,
                    source=source,
                    target=target,
                )

    # Draw Graph
    def draw_graph(self, G):
        pos = nx.get_node_attributes(G, "pos")

        # Labelling Each Type Of Node
        labels = {}
        for node in G.nodes:
            if G.nodes[node].get("is_backbone", False):
                labels[node] = "Backbone UAV"
            elif G.nodes[node].get("is_ground_station", False):
                labels[node] = "Ground Station"
            elif G.nodes[node].get("is_attack_node", False):
                labels[node] = "Attack Node"
            else:
                labels[node] = f"UAV{node}"

        # Differentiate node types
        regular_nodes = [
            node
            for node in G.nodes
            if not G.nodes[node].get("is_backbone", False)
            and not G.nodes[node].get("is_ground_station", False)
        ]
        backbone_nodes = [
            node for node in G.nodes if G.nodes[node].get("is_backbone", False)
        ]
        ground_station_nodes = [
            node for node in G.nodes if G.nodes[node].get("is_ground_station", False)
        ]
        attacking_nodes = [
            node for node in G.nodes if G.nodes[node].get("is_attack_node", False)
        ]

        # Draw regular UAV nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=regular_nodes,
            node_color="green",
            node_size=300,
            label="UAV",
        )

        # Draw backbone UAV node
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=backbone_nodes,
            node_color="lightgreen",
            node_size=500,
            label="Backbone UAV",
        )

        # Draw ground station node
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=ground_station_nodes,
            node_color="lightblue",
            node_size=500,
            label="Ground Station",
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=attacking_nodes,
            node_color="red",
            node_size=500,
            label="Attack Node",
        )

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="gray")

        # Draw custom labels
        nx.draw_networkx_labels(G, pos, labels)

        plt.title("Multilayer UAV Network")
        plt.legend()
        plt.show()

    def update_uav_positions(self, G, move_range):
        # Update the positions of UAVs randomly within a given range.

        for node in G.nodes:
            if (
                "is_backbone" not in G.nodes[node]
                and "is_ground_station" not in G.nodes[node]
            ):
                current_pos = G.nodes[node]["pos"]
                new_pos = (
                    current_pos[0] + random.uniform(-move_range, move_range),
                    current_pos[1] + random.uniform(-move_range, move_range),
                )
                G.nodes[node]["pos"] = new_pos
        return G

    def update_network_connections(self, G, connection_range, backbone_range):
        # Update the network connections based on new positions and connection range.
        # Excluding backbone UAV and ground station
        num_uavs = len(G.nodes) - 2
        backbone_uav_id = num_uavs

        # Update connections for regular UAVs
        for i in range(num_uavs):
            for j in range(i + 1, num_uavs):
                if G.has_edge(i, j):
                    G.remove_edge(i, j)
                pos_i = G.nodes[i]["pos"]
                pos_j = G.nodes[j]["pos"]
                distance = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                if distance <= connection_range:
                    G.add_edge(i, j, weight=distance)

        # Update connections for the Backbone UAV
        for i in range(num_uavs):
            if G.has_edge(i, backbone_uav_id):
                G.remove_edge(i, backbone_uav_id)
            pos_i = G.nodes[i]["pos"]
            pos_backbone = G.nodes[backbone_uav_id]["pos"]
            distance = np.linalg.norm(np.array(pos_i) - np.array(pos_backbone))
            if distance <= backbone_range:
                G.add_edge(i, backbone_uav_id, weight=distance)

        return G

    def update_neo4j_database(self, G):
        # Update the Neo4j database with the new positions and connections.

        with self.driver.session() as session:
            # Update positions
            for node in G.nodes:
                pos = G.nodes[node]["pos"]
                session.run(
                    """
                    MATCH (n)
                    WHERE n.id = $id
                    SET n.pos = $pos
                    """,
                    id=node,
                    pos=pos,
                )

            # Update relationships
            session.run("MATCH (n)-[r:COMMUNICATES_WITH]->() DELETE r")
            for source, target in G.edges:
                session.run(
                    """
                    MATCH (a), (b)
                    WHERE a.id = $source AND b.id = $target
                    CREATE (a)-[:COMMUNICATES_WITH]->(b)
                    """,
                    source=source,
                    target=target,
                )

    def generate_network_traffic(self, G, num_packets):
        # Generate network traffic considering the throughput of UAVs.

        packets = []
        for _ in range(num_packets):
            # Exclude backbone and ground station
            source = random.randint(0, len(G.nodes) - 3)
            target = random.randint(0, len(G.nodes) - 3)

            # Packet size should not exceed the throughput of the source UAV
            max_packet_size = G.nodes[source]["throughput"]
            size = random.randint(1, max_packet_size)

            packets.append({"source": source, "target": target, "size": size})
        return packets

    def route_packets(self, G, packets):
        # Determine the path for each packet and calculate cumulative latency.

        for packet in packets:
            source = packet["source"]
            target = packet["target"]
            if nx.has_path(G, source, target):
                path = nx.shortest_path(G, source, target)
                packet["path"] = path
                packet["delivered"] = True

                # Calculate cumulative latency
                total_latency = sum(G.nodes[node]["latency"] for node in path)
                packet["total_latency"] = total_latency
            else:
                packet["path"] = []
                packet["delivered"] = False
                packet["total_latency"] = None
        return packets

    def add_attack_node(
        self,
        G,
        target_ids,
        num_uavs,
        attack_node_pos=(random.randint(0, 100), random.randint(0, 100)),
    ):
        # Calculate a new ID for the attack node by taking the maximum current ID + 1

        attack_node_id = max(G.nodes()) + 1
        attk_battery = (100, 100)
        battery = random.randint(*attk_battery)

        # Add the attack node with specified properties
        G.add_node(
            attack_node_id,
            pos=attack_node_pos,
            nodeName="Attack Node",
            is_attack_node=True,
            throughput=1000,
            latency=1,
            battery=battery,
        )

        # Directly connect the attack node to the target node
        for target_id in target_ids:
            if target_id in G.nodes():
                G.add_edge(
                    attack_node_id, target_id, weight=5
                )  # Weight can be adjusted or calculated based on actual metrics

        return attack_node_id

    def upload_attack_node_to_neo4j(self, G, attack_node_id, target_ids):
        with self.driver.session() as session:
            # Retrieve node data from the graph
            attack_node = G.nodes[attack_node_id]

            # Create the attack node in Neo4j
            session.run(
                """
                CREATE (a:AttackNode {
                    id: $id, 
                    pos: $pos,
                    nodeName: $nodeName,
                    throughput: $throughput, 
                    latency: $latency, 
                    battery: $battery, 
                    is_attack_node: $is_attack_node
                })
                """,
                id=attack_node_id,
                pos=str(attack_node["pos"]),
                nodeName=attack_node["nodeName"],
                throughput=attack_node["throughput"],
                latency=attack_node["latency"],
                battery=attack_node["battery"],
                is_attack_node=True,
            )

            # Create a relationship from the attack node to the target UAV
            for target_id in target_ids:
                if target_id in G.nodes():
                    session.run(
                        """
                        MATCH (a:AttackNode {id: $attack_id}), (u:UAV {id: $target_id})
                        CREATE (a)-[:ATTACKS]->(u)
                        """,
                        attack_id=attack_node_id,
                        target_id=target_id,
                    )

    def update_network_metrics(self, G):
        for node in G.nodes():
            # Randomly increase latency by up to 10ms and decrease throughput by up to 10 Mbps
            G.nodes[node]["latency"] = max(
                1, G.nodes[node]["latency"] + random.randint(-5, 5)
            )
            G.nodes[node]["throughput"] = max(
                1, G.nodes[node]["throughput"] + random.randint(-10, 10)
            )

    def update_network_ddos_metrics(self, G):
        for node in G.nodes():
            # Randomly increase latency by up to 10ms and decrease throughput by up to 10 Mbps
            G.nodes[node]["latency"] += random.randint(1, 5)
            G.nodes[node]["throughput"] = max(
                1, G.nodes[node]["throughput"] - random.randint(1, 10)
            )

    def run_simulation(
        self,
        G,
        total_time,
        update_interval,
        move_range,
        connection_range,
        backbone_range,
        num_packets,
    ):
        # Run the simulation for a specified period of time.
        results = []
        start_time = time.time()
        iteration = 0
        while time.time() - start_time < total_time:
            print(f"Iteration {iteration}:")

            # Update UAV positions
            self.update_uav_positions(G, move_range)

            # Update network connections
            self.update_network_connections(G, connection_range, backbone_range)
            self.update_network_metrics(G)

            # Update Neo4j database and redraw the graph
            # self.update_neo4j_database(G)
            # self.draw_graph(G)

            packets = self.generate_network_traffic(G, num_packets)
            routed_packets = self.route_packets(G, packets)
            # Capture and record metrics for each node
            current_time = time.time() - start_time
            for node_id in G.nodes():
                results.append(
                    {
                        "time": current_time,
                        "node_id": node_id,
                        "latency": G.nodes[node_id]["latency"],
                        "throughput": G.nodes[node_id]["throughput"],
                    }
                )
            print(results)

            # Log the routed packets information
            for i, packet in enumerate(routed_packets):
                print(
                    f"  Packet {i}: Source: {packet['source']}, Target: {packet['target']}, "
                    f"Size: {packet['size']}, Delivered: {packet['delivered']}, "
                    f"Path: {packet['path']}, Total Latency: {packet['total_latency']}"
                )

            # Wait for the next update
            time.sleep(update_interval)
            iteration += 1
        return results

    def simulate_ddos_attack(
        self, G, attack_node_id, target_ids, increase_factor, decrease_factor
    ):
        for target_id in target_ids:
            if target_id in G.nodes():
                G.nodes[target_id]["latency"] = int(
                    G.nodes[target_id]["latency"] * increase_factor
                )
                G.nodes[target_id]["throughput"] = int(
                    G.nodes[target_id]["throughput"] * decrease_factor
                )
                G.nodes[target_id]["battery"] = int(
                    G.nodes[target_id]["battery"] * decrease_factor
                )
                # Apply effects to neighboring nodes, ensuring integer operations
                for neighbor in G.neighbors(target_id):
                    G.nodes[neighbor]["latency"] = int(
                        G.nodes[neighbor]["latency"] * (increase_factor - 0.1)
                    )
                    G.nodes[neighbor]["throughput"] = int(
                        G.nodes[neighbor]["throughput"] * (decrease_factor + 0.1)
                    )
                    # G.nodes[neighbor]['battery'] = int(G.nodes[neighbor]['battery'] * (decrease_factor + 0.1))
                    if "battery" in G.nodes[neighbor]:
                        G.nodes[neighbor]["battery"] = int(
                            G.nodes[neighbor]["battery"] * (decrease_factor + 0.1)
                        )

    def run_ddos_simulation(
        self,
        G,
        total_time,
        update_interval,
        move_range,
        connection_range,
        backbone_range,
        num_packets,
        attack_node_id,
        target_ids,
    ):
        # Setup MongoDB connection
        client = MongoClient(
            "mongodb://localhost:27017/"
        )  # Adjust the connection URL if necessary
        db = client["ddos_simulation"]
        collection = db["metrics"]

        start_time = time.time()
        iteration = 0

        while time.time() - start_time < total_time:
            print(f"Iteration {iteration}:")

            # Simulation updates
            self.update_uav_positions(G, move_range)
            self.update_network_connections(G, connection_range, backbone_range)
            self.update_network_metrics(G)

            # Trigger DDoS attack
            if iteration == 1:
                self.simulate_ddos_attack(G, attack_node_id, target_ids, 0.5, 0.5)

            # Traffic generation and routing
            packets = self.generate_network_traffic(G, num_packets)
            routed_packets = self.route_packets(G, packets)

            # Capture and log metrics
            current_metrics = [
                {
                    "node": node,
                    "time": time.time() - start_time,
                    "latency": G.nodes[node]["latency"],
                    "throughput": G.nodes[node]["throughput"],
                    "battery": G.nodes[node]["battery"],
                }
                for node in G.nodes()
            ]

            # Insert metrics into MongoDB
            collection.insert_many(current_metrics)

            # Print packet info
            for i, packet in enumerate(routed_packets):
                print(
                    f"Packet {i}: Source: {packet['source']}, Target: {packet['target']}, Size: {packet['size']}, Delivered: {packet['delivered']}, Path: {packet['path']}, Total Latency: {packet['total_latency']}"
                )

            # Delay for next update
            time.sleep(update_interval)
            iteration += 1

        print("Metrics have been saved to MongoDB.")
        return current_metrics

    def plot_time_series(self, results):
        import matplotlib.pyplot as plt

        # Filter results for a specific node for clarity, e.g., node 0
        node_results = [res for res in results if res["node_id"] == 0]
        times = [res["time"] for res in node_results]
        latencies = [res["latency"] for res in node_results]
        throughputs = [res["throughput"] for res in node_results]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(times, latencies, marker="o", color="r")
        plt.title("Latency Over Time for Node 0")
        plt.xlabel("Time (s)")
        plt.ylabel("Latency (ms)")

        plt.subplot(1, 2, 2)
        plt.plot(times, throughputs, marker="o", color="b")
        plt.title("Throughput Over Time for Node 0")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (Mbps)")

        plt.tight_layout()
        plt.show()

    def analyze_network(self, G):
        # Calculate basic properties
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        sparsity = nx.density(G)  # Density close to 0 indicates a sparse graph

        # Centrality measures
        degree_centrality = nx.degree_centrality(G)  # Normalized to (n-1)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)

        # Identifying the most central node
        most_central_node = max(degree_centrality, key=degree_centrality.get)

        # Network robustness (using node connectivity)
        connectivity = nx.node_connectivity(G)

        # Plotting
        pos = nx.spring_layout(G)  # positions for all nodes
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_edges(G, pos, width=6)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        plt.axis("off")
        plt.show()

        return {
            "Number of Nodes": num_nodes,
            "Number of Edges": num_edges,
            "Sparsity": sparsity,
            "Degree Centrality": degree_centrality,
            "Closeness Centrality": closeness_centrality,
            "Betweenness Centrality": betweenness_centrality,
            "Most Central Node": most_central_node,
            "Connectivity": connectivity,
        }
