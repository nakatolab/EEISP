#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Thanks to: https://qiita.com/Ihori/items/0944b3b344d65c95372a; https://github.com/taynaud/python-louvain

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from eeisp.network import generate_signednetwork, calc_entropy

ITER_LIMIT_PER_LOCALMOVE = -1
MIN = 0.0000001

def renumber(dictionary):
    values = set(dictionary.values())
    renumbering = dict(zip(values, range(len(values))))
    return {k: renumbering[v] for k, v in dictionary.items()}


def plot_alphas_distributions(G_positive, G_negative, resolution, seed):

    alphas = np.arange(0, 1.05, 0.05)
    intra_pos_ratios = []
    inter_pos_ratios = []
    intra_neg_ratios = []
    inter_neg_ratios = []
    entropies = []
    n_communities_list = []

    for alpha in alphas:
        partition = LouvainSigned(G_positive, G_negative).best_partition(alpha=alpha, resolution=resolution, seed=seed)

        communities = set(partition.values())
        n_communities = len(communities)
        entropy = calc_entropy(partition)

        intra_pos_edges, inter_pos_edges, intra_neg_edges, inter_neg_edges = 0, 0, 0, 0
        for u, v in G_positive.edges():
            if partition[u] == partition[v]:
                intra_pos_edges += 1
            else:
                inter_pos_edges += 1
        for u, v in G_negative.edges():
            if u not in partition or v not in partition:
                continue
            if partition[u] == partition[v]:
                intra_neg_edges += 1
            else:
                inter_neg_edges += 1

        total_pos_edges = G_positive.number_of_edges()
        total_neg_edges = G_negative.number_of_edges()

        intra_pos_ratio = intra_pos_edges / total_pos_edges
        inter_pos_ratio = inter_pos_edges / total_pos_edges
        intra_neg_ratio = intra_neg_edges / total_neg_edges
        inter_neg_ratio = inter_neg_edges / total_neg_edges

        # Append the results to the lists
        intra_pos_ratios.append(intra_pos_ratio)
        inter_pos_ratios.append(inter_pos_ratio)
        intra_neg_ratios.append(intra_neg_ratio)
        inter_neg_ratios.append(inter_neg_ratio)
        n_communities_list.append(n_communities)
        entropies.append(entropy)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(alphas, intra_pos_ratios, label='Intra-community Positive Ratio')
    ax1.plot(alphas, inter_pos_ratios, label='Inter-community Positive Ratio')
    ax1.plot(alphas, intra_neg_ratios, label='Intra-community Negative Ratio')
    ax1.plot(alphas, inter_neg_ratios, label='Inter-community Negative Ratio')
    ax1.plot(alphas, entropies,        label='Normalized entropy')
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Ratio')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax1.invert_xaxis()
    # Plot the number of communities with the second y-axis
    ax2.bar(alphas, n_communities_list, width=0.03, alpha=0.6, color='gray', label='Number of Communities')
    ax2.set_ylabel('Number of Communities')

    # Add the legend for the second y-axis
    ax2.legend(loc='upper right')

    #plt.show()

class GraphInfo:
    def __init__(self, ref_graph, graph, partition, weight_key):
        self.graph_whole = nx.Graph()
        self.graph_whole.add_nodes_from(ref_graph.nodes(data=True))
#        self.graph_whole.add_edges_from(graph.edges(data=True))
        for u, v, data in graph.edges(data=True):
            if u in self.graph_whole.nodes() and v in self.graph_whole.nodes():
                self.graph_whole.add_edge(u, v, **data)

        self.weight_key = weight_key
        self.node_degrees = dict()
        self.loops = dict()
        self.size = float(self.graph_whole.size(weight=weight_key))
        self.community_degrees = dict()
        self.community_sizes = dict()

        for node in self.graph_whole.nodes():
            node_degree = float(self.graph_whole.degree(node, weight=weight_key))
            self.node_degrees[node] = node_degree
            loop_edge = graph.get_edge_data(node, node, default={weight_key: 0.})
            self.loops[node] = float(loop_edge.get(weight_key, 0))

        self._initialize_community_info(partition)

    def _initialize_community_info(self, partition):
        self.community_degrees = {community: 0 for community in set(partition.values())}
        self.community_sizes   = {community: 0 for community in set(partition.values())}
        self.linksum_dict = {node: self.get_linksum_dict(node, partition) for node in self.graph_whole.nodes()}

        for node, community in partition.items():
            node_degree = self.node_degrees[node]
            loop = self.loops[node]
            self.community_degrees[community] += node_degree
            self.community_sizes[community] += loop

    def calc_modularity(self, partition, *, resolution=1.):
        m = self.size
        modularity = 0.
        for community in set(partition.values()):
            K_c = self.community_degrees.get(community, 0.)
            m_c = self.community_sizes.get(community, 0.)
            modularity += m_c / m - resolution * ((K_c / (2. * m)) ** 2)
#            print (f"community: {community} K_c: {K_c} m: {m} m_c: {m_c} modu: {modularity} res: {resolution}")
        return modularity

    def _remove_node(self, node, partition, linksum_dict):
        community = partition.get(node)
        linksum = self.linksum_dict[node].get(community, 0.)
        self.community_degrees[community] = self.community_degrees.get(community, 0.) - self.node_degrees.get(node, 0.)
        self.community_sizes[community] = self.community_sizes.get(community, 0.) - linksum - self.loops.get(node, 0.)

    def _insert_node(self, node, partition, linksum_dict, community):
        linksum = self.linksum_dict[node].get(community, 0.)
        partition[node] = community
        self.community_degrees[community] = self.community_degrees.get(community, 0.) + self.node_degrees.get(node, 0.)
        self.community_sizes[community] = self.community_sizes.get(community, 0.) + linksum + self.loops.get(node, 0.)

    def get_linksum_dict(self, node, partition):
        weight_key = self.weight_key
        graph = self.graph_whole
        linksum_dict = defaultdict(float)

        for neighbor_node, edge in graph[node].items():
            if neighbor_node != node:
                neighbor_community = partition[neighbor_node]
#                linksum_dict[neighbor_community] += edge.get(weight_key, 0)
                linksum_dict[neighbor_community] += edge.get(weight_key, 1)

        return linksum_dict

    def _delta_q_1(self, node, linksum_dict, partition, resolution):
        ki = self.node_degrees.get(node, 0.)
        community = partition.get(node)
        ac2m = self.community_degrees.get(community, 0.)
        m = self.size
        linksum = linksum_dict.get(community, 0.) 
        q = - linksum + resolution * (ac2m * ki - ki ** 2) / (2 * m)
        return q

    def _delta_q_2(self, node, linksum_dict, neighboring_community, resolution):
        ki = self.node_degrees.get(node, 0.)
        ac2m = self.community_degrees.get(neighboring_community, 0.)
        m = self.size
        linksum = linksum_dict.get(neighboring_community, 0.)
        q = linksum - resolution * (ac2m * ki / (2 * m))
        return q


class LouvainSigned:
    def __init__(self, positive_graph, negative_graph, *, mode="positive"):
        self.weight_key = 'weight'
        self.graph_whole = self.get_signed_graph(positive_graph, negative_graph, mode)
        self.graph_original = self.graph_whole.copy()

        self.partition = {node: i for i, node in enumerate(self.graph_whole.nodes())}

        self.ginfo_pos = GraphInfo(self.graph_whole, positive_graph, self.partition, self.weight_key)
        self.ginfo_neg = GraphInfo(self.graph_whole, negative_graph, self.partition, self.weight_key)

        self.dendrogram = list()
        self.mode = mode

    def get_signed_graph(self, positive_graph, negative_graph, mode):
        graph = nx.Graph()
        if mode == "Full":
            graph.add_nodes_from(positive_graph.nodes(data=True))
            graph.add_nodes_from(negative_graph.nodes(data=True))
            # graph.add_edges_from(positive_graph.edges(data=True), sign='positive')
            # graph.add_edges_from(negative_graph.edges(data=True), sign='negative')
        else:
            graph.add_nodes_from(positive_graph.nodes(data=True))

        return graph

    def _randomize(self, items, random_generator):
        randomized_items = list(items)
        random_generator.shuffle(randomized_items)
        return randomized_items

    def _modularity(self, partition, *, alpha=1., resolution=1.):
        Q_pos = self.ginfo_pos.calc_modularity(partition, resolution=resolution)
        Q_neg = self.ginfo_neg.calc_modularity(partition, resolution=resolution)
#        modularity = alpha * Q_pos + (1-alpha) * (1 - Q_neg)
        modularity = alpha * Q_pos - (1 - alpha) * Q_neg
#        print ("modularity", modularity, Q_pos, Q_neg, alpha, resolution)
        return modularity

    def _move_nodes(self, random_generator, *, alpha=1., resolution=1.):
        modified = True
        nb_pass_done = 0
        new_Q = self._modularity(self.partition, alpha=alpha, resolution=resolution)

        while modified and nb_pass_done != ITER_LIMIT_PER_LOCALMOVE:
            current_Q = new_Q
            modified = False
            nb_pass_done += 1

            for node in self._randomize(self.graph_whole.nodes(), random_generator):
                original_community = self.partition.get(node)
                linksum_dict_pos = self.ginfo_pos.get_linksum_dict(node, self.partition)
                linksum_dict_neg = self.ginfo_neg.get_linksum_dict(node, self.partition)
                q1_pos = self.ginfo_pos._delta_q_1(node, linksum_dict_pos, self.partition, resolution)
                q1_neg = self.ginfo_neg._delta_q_1(node, linksum_dict_neg, self.partition, resolution)
                self.ginfo_pos._remove_node(node, self.partition, linksum_dict_pos)
                self.ginfo_neg._remove_node(node, self.partition, linksum_dict_neg)
                self.partition[node] = -1

                best_community = original_community
                best_increase = 0
                for neighboring_community in self._randomize(linksum_dict_pos.keys(), random_generator):
                    q2_pos = self.ginfo_pos._delta_q_2(node, linksum_dict_pos, neighboring_community, resolution)
                    q2_neg = self.ginfo_neg._delta_q_2(node, linksum_dict_neg, neighboring_community, resolution)

#                    delta_Q = alpha * (q1_pos + q2_pos) - (1 - alpha) * (1 - q1_neg - q2_neg)
                    delta_Q = alpha * (q1_pos + q2_pos) - (1 - alpha) * (q1_neg + q2_neg)
#                    print ("deltaQ", neighboring_community, delta_Q)
                    if delta_Q > best_increase:
                        best_increase = delta_Q
                        best_community = neighboring_community

                self.ginfo_pos._insert_node(node, self.partition, linksum_dict_pos, best_community)
                self.ginfo_neg._insert_node(node, self.partition, linksum_dict_neg, best_community)
                self.partition[node] = best_community
#                print ("best com", best_community)
                if best_community != original_community:
                    modified = True

            new_Q = self._modularity(self.partition, alpha=alpha, resolution=resolution)
            if new_Q - current_Q < MIN:
                break

    def _aggregate_nodes(self, partition, graph):
        weight = self.weight_key
        aggregated_graph = nx.Graph()
#        aggregated_graph.add_nodes_from(partition.values())

        for node, community in partition.items():
            if community not in aggregated_graph:
                aggregated_graph.add_node(community)
                # 最初のノードの属性をコミュニティノードの属性としてコピー
                aggregated_graph.nodes[community].update(graph.nodes[node])

        for node1, node2, _data in graph.edges(data=True):
            edge_weight = _data.get(weight, 1)
            c1 = partition[node1]
            c2 = partition[node2]
            w_prec = aggregated_graph.get_edge_data(c1, c2, {weight: 0}).get(weight, 0)

            aggregated_graph.add_edge(c1, c2, **{weight: w_prec + edge_weight})
        return aggregated_graph

    def generate_dendrogram(self, mode, random_generator, *, alpha=1., resolution=1.):
        current_graph_pos = self.ginfo_pos.graph_whole.copy()
        current_graph_neg = self.ginfo_neg.graph_whole.copy()
        partition_list = list()
        Q = -1.0

        while True:
            self._move_nodes(random_generator, alpha=alpha, resolution=resolution)
            renumberd_partition = renumber(self.partition)
            partition_list.append(renumberd_partition)
            current_graph_pos = self._aggregate_nodes(renumberd_partition, current_graph_pos)
            current_graph_neg = self._aggregate_nodes(renumberd_partition, current_graph_neg)
            self.graph_whole = self.get_signed_graph(current_graph_pos, current_graph_neg, mode)

            self.partition = {node: i for i, node in enumerate(self.graph_whole.nodes())}
            self.ginfo_pos = GraphInfo(self.graph_whole, current_graph_pos, self.partition, self.weight_key)
            self.ginfo_neg = GraphInfo(self.graph_whole, current_graph_neg, self.partition, self.weight_key)
            new_Q = self._modularity(self.partition, alpha=alpha, resolution=resolution)
#            print("generate_dendrogram", new_Q)
            if new_Q - Q < MIN:
                break
            Q = new_Q

        self.dendrogram = partition_list[:]

    def best_partition(self, *, alpha=1., resolution=1., seed=None):
        if not isinstance(alpha, (int, float)):
            print("Error: Alpha value must be a number. Please provide a valid value.")
            return
        elif not 0 <= alpha <= 1:
            print("Error: Alpha value must be in the range [0, 1]. Please provide a valid value.")
            return
        print (f"alpha: {alpha}, resolution: {resolution}")

        self.generate_dendrogram(self.mode, np.random.default_rng(seed=seed), alpha=alpha, resolution=resolution)
        partition = self.dendrogram[0].copy()
        for level in range(1, len(self.dendrogram)):
            for node, community in partition.items():
                partition[node] = self.dendrogram[level][community]

        print(f"Final Modularity: {self._modularity(partition, alpha=alpha, resolution=resolution)}")
        print(f"Positive Modularity (Q+): {self.ginfo_pos.calc_modularity(partition, resolution=resolution)}")
        print(f"Negative Modularity (Q-): {self.ginfo_neg.calc_modularity(partition, resolution=resolution)}")

        return partition


    def find_communities_of_genes(self, partition, gene_names):
        graph = self.graph_original
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


def main():
    parser = argparse.ArgumentParser(prog='eeispcommunity')
    parser.add_argument("--alpha", help="alpha parameter (from 0 to 1, default: 0.5)", type=float, default=0.5)
    parser.add_argument("--resolution", help="resolution for louvain (default: 1.0)", type=float, default=1.)
    parser.add_argument("--seed", help="seed for LouvainSigned", type=int, default=10)

    args = parser.parse_args()
    print(args)

    alpha = args.alpha
    community_size = 8
    num_communities = 4
    intra_edges = 12
    inter_edges = 10
    p1 = 0.05
    p2 = 0.05

    G_positive, G_negative = generate_signednetwork(community_size, num_communities, intra_edges, inter_edges, p1, p2)

    print(f"G_positive: Nodes={G_positive.number_of_nodes()}, Edges={G_positive.number_of_edges()}")
    print(f"G_negative: Nodes={G_negative.number_of_nodes()}, Edges={G_negative.number_of_edges()}")
    print(f"G: Nodes={nx.compose(G_positive, G_negative).number_of_nodes()}, Edges={nx.compose(G_positive, G_negative).number_of_edges()}")

    l = LouvainSigned(G_positive, G_negative, alpha, seed=args.seed)
    partition = l.best_partition()
    print(f"partition: {partition}")


if __name__ == "__main__":
    main()
