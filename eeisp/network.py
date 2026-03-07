#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Thanks to: https://qiita.com/Ihori/items/0944b3b344d65c95372a

import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import igraph as ig

ITER_LIMIT_PER_LOCALMOVE = -1
MIN = 0.0000001

def load_graph_from_file(filename, threshold):
    G = nx.Graph()
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_id1 = parts[0]
            gene_id2 = parts[1]
            gene_name1 = parts[2]
            gene_name2 = parts[3]
            weight = float(parts[4])
            if weight >= threshold:
                G.add_node(gene_id1, name=gene_name1)
                G.add_node(gene_id2, name=gene_name2)
                G.add_edge(gene_id1, gene_id2, weight=weight)
    return G

def ig_load_graph_from_file(filename, threshold):# エッジリストを初期化
    edges = []
    weights = []
    node_names = {}

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_id1 = parts[0]
            gene_id2 = parts[1]
            gene_name1 = parts[2]
            gene_name2 = parts[3]
            weight = float(parts[4])
            if weight >= threshold:
                edges.append((gene_id1, gene_id2))
                weights.append(float(weight))
                node_names[gene_name1] = gene_id1
                node_names[gene_name2] = gene_id2

    # グラフを作成
    g = ig.Graph.TupleList(edges, weights=True)
    g.vs['name'] = list(node_names.keys())
    g.es['weight'] = weights
    return g

def count_nodes_in_communities(partition):
    community_counts = {}
    for node, community in partition.items():
        if community in community_counts:
            community_counts[community] += 1
        else:
            community_counts[community] = 1

    for community, count in sorted(community_counts.items()):
        print(f"Community {community}: {count} nodes")


def generate_signednetwork(community_size, num_communities, intra_edges, inter_edges, p1, p2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    G_positive = nx.Graph()
    G_negative = nx.Graph()

    for i in range(num_communities):
        G_tmp = nx.gnm_random_graph(community_size, intra_edges)
        G_tmp = nx.relabel_nodes(G_tmp, {node: node + i*community_size for node in G_tmp.nodes()})
        G_positive = nx.compose(G_positive, G_tmp)

    inter_edge_candidates = [(i, j) for i in range(community_size*num_communities) for j in range(i+1, community_size*num_communities) if abs(i//community_size - j//community_size) == 1]
    inter_edge_selected = np.random.choice(len(inter_edge_candidates), inter_edges * num_communities, replace=False)
    negative_edges = [inter_edge_candidates[i] for i in inter_edge_selected]
    G_negative.add_edges_from(negative_edges)

    G_positive.add_nodes_from(G_negative.nodes())
    G_negative.add_nodes_from(G_positive.nodes())

    positive_edges = list(G_positive.edges())
    for edge_index in np.random.choice(len(positive_edges), int(intra_edges * p1), replace=False):
        edge = positive_edges[edge_index]
        if edge[0]//community_size == edge[1]//community_size and G_positive.degree(edge[0]) > 1 and G_positive.degree(edge[1]) > 1:
            G_positive.remove_edge(*edge)
            G_negative.add_edge(*edge)

    negative_edges = list(G_negative.edges())
    for edge_index in np.random.choice(len(negative_edges), int(inter_edges * p2), replace=False):
        edge = negative_edges[edge_index]
        if edge[0]//community_size != edge[1]//community_size and G_negative.degree(edge[0]) > 1 and G_negative.degree(edge[1]) > 1:
            G_negative.remove_edge(*edge)
            G_positive.add_edge(*edge)

    return G_positive, G_negative


def visualize_top_weighted_nodes(graph, gene_name, top_n):
    gene_id = None
    for node, data in graph.nodes(data=True):
        if data['name'] == gene_name:
            gene_id = node
            break
    
    if gene_id is None:
        print(f"No gene named {gene_name} found in the graph.")
        return

    edges = [(gene_id, neighbor, data['weight']) for neighbor, data in graph[gene_id].items()]
    
    top_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]

    top_nodes = {edge[1] for edge in top_edges}
    top_nodes.add(gene_id)
    subgraph = graph.subgraph(top_nodes)

    pos = nx.spring_layout(subgraph)  # ノードの位置を決定
    labels = {n: graph.nodes[n]['name'] for n in subgraph.nodes()}  # ノードIDではなく遺伝子名でラベルを付ける
    
    node_colors = ['red' if node == gene_id else 'lightblue' for node in subgraph.nodes()]
    
    nx.draw(subgraph, pos, labels=labels, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500, font_size=12)
    plt.title(f"Top {top_n} genes connections to {gene_name}")
    plt.show()


def recursive_louvain_partition(graph, *, min_size=100):
    def partition_and_recurse(subgraph, current_partition, start_comm_id):
        partition = community_louvain.best_partition(subgraph)
        subgraphs = {}
        
        for node, community_id in partition.items():
            if community_id not in subgraphs:
                subgraphs[community_id] = []
            subgraphs[community_id].append(node)
        
        next_comm_id = start_comm_id
        
        for community_id, nodes in subgraphs.items():
            if len(nodes) > min_size:
                sub_subgraph = graph.subgraph(nodes)
                current_partition, next_comm_id = partition_and_recurse(
                    sub_subgraph, current_partition, next_comm_id)
            else:
                for node in nodes:
                    current_partition[node] = next_comm_id
                next_comm_id += 1
        
        return current_partition, next_comm_id

    initial_partition = {}
    final_partition, _ = partition_and_recurse(graph, initial_partition, 0)
    return final_partition


def visualize_community(graph, partition, community_id):
    nodes_in_community = [node for node, comm_id in partition.items() if comm_id == community_id]
    subgraph = graph.subgraph(nodes_in_community)
    
    pos = nx.kamada_kawai_layout(subgraph)
    plt.figure(figsize=(9, 9))
    nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color="lightblue")
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5)
    nx.draw_networkx_labels(subgraph, pos, labels={n: subgraph.nodes[n]['name'] for n in subgraph.nodes()})
    plt.title(f"Community {community_id}")
    plt.show()


def visualize_top_weighted_nodes_between_genes(graph, gene_names, top_n):
    gene_ids = []
    for node, data in graph.nodes(data=True):
        if data['name'] in gene_names:
            gene_ids.append(node)

    if len(gene_ids) != len(gene_names):
        print("Some genes not found in the graph.")
        return

    node_weights = {}
    for gene_id in gene_ids:
        for neighbor in graph.neighbors(gene_id):
            if neighbor not in node_weights:
                node_weights[neighbor] = 0
            node_weights[neighbor] += graph[gene_id][neighbor]['weight']

    top_nodes = sorted(node_weights, key=node_weights.get, reverse=True)[:top_n]

    subgraph_nodes = set(top_nodes).union(set(gene_ids))
    subgraph = graph.subgraph(subgraph_nodes)

    pos = nx.spring_layout(subgraph) 
    node_colors = ['red' if node in gene_ids else 'lightblue' for node in subgraph.nodes()]

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5)
    nx.draw_networkx_labels(subgraph, pos, labels={n: subgraph.nodes[n]['name'] for n in subgraph.nodes()})

    plt.title(f"Top {top_n} Weighted Connections for {', '.join(gene_names)}")
    plt.axis('off')  # 軸をオフにする
    plt.show()


def display_communities_by_name(graph, partition):
    community_dict = {}
    for node, community in partition.items():
        gene_name = graph.nodes[node]['name']
        if community not in community_dict:
            community_dict[community] = []
        community_dict[community].append(gene_name)
    
    for community, names in community_dict.items():
        print(f'Community {community}: {" | ".join(set(names))}')


def find_communities_of_genes(graph, partition, gene_names):
    results = {}
    for gene_name in gene_names:
        node_id = next((node for node, data in graph.nodes(data=True) if data['name'] == gene_name), None)
        if node_id is None:
            results[gene_name] = "Gene not found in the graph."
            continue
        community_id = partition.get(node_id, None)
        if community_id is None:
            results[gene_name] = "Community for gene not found."
        else:
            results[gene_name] = community_id
    return results


def count_nodes_in_communities(partition):
    community_counts = {}
    for node, community in partition.items():
        if community in community_counts:
            community_counts[community] += 1
        else:
            community_counts[community] = 1
            
    for community, count in sorted(community_counts.items()):
        print(f"Community {community}: {count} nodes")


def extract_subgraph(graph, partition, community_id):
    nodes_in_community = [node for node in graph.nodes if partition[node] == community_id]
    subgraph = graph.subgraph(nodes_in_community)
    return subgraph


def plot_community(subgraph):
    pos = nx.kamada_kawai_layout(subgraph)
    edge_widths = [subgraph[u][v]['weight'] / max(subgraph[u][v]['weight'] for u, v in subgraph.edges()) * 2 for u, v in subgraph.edges()]
    node_sizes = [subgraph.degree(n) * 10 for n in subgraph.nodes()]
    font_size = 8 if len(subgraph.nodes()) > 100 else 12
    nx.draw(subgraph, pos, with_labels=True, labels={node: subgraph.nodes[node]['name'] for node in subgraph.nodes()},
            node_color='skyblue', node_size=node_sizes, font_size=font_size, font_weight='bold', edge_color='gray', width=edge_widths)
    plt.show()


def calc_entropy(partition):
    cluster_sizes = list(partition.values())
    unique, counts = np.unique(cluster_sizes, return_counts=True)
    cluster_size_distribution = dict(zip(unique, counts))

    sizes = list(cluster_size_distribution.values())
    total_nodes = sum(sizes)
    num_clusters = len(sizes)

    entropy = -sum((size / total_nodes) * np.log(size / total_nodes) for size in sizes)
    normalized_entropy = entropy / np.log(num_clusters)

    return normalized_entropy
    
#    print(f"エントロピー: {entropy}")
#    print(f"正規化エントロピー: {normalized_entropy}")