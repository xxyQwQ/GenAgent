from collections import defaultdict


class Graph:
    def __init__(self, directed=True):
        self.graph = defaultdict(list)
        self.directed = directed

    def get_nodes(self):
        return list(self.graph.keys())

    def get_edges(self):
        edges = []
        for source in self.graph:
            for target in self.graph[source]:
                edges.append((source, target))
        return edges

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, source, target):
        self.graph[source].append(target)
        if not self.directed:
            self.graph[target].append(source)
