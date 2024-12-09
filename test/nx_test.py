import networkx as nx

# 创建一个有向图
residual_digraph = nx.DiGraph()
residual_digraph.add_edge('A', 'B')
residual_digraph.add_edge('C', 'A')
residual_digraph.add_edge('D', 'A')
residual_digraph.add_edge('A', 'E')

# 获取节点 'A' 的所有邻居节点（包括传入节点和传出节点）
all_neighbors = list(nx.neighbors(residual_digraph, 'A'))
print("All neighbors of 'A':", all_neighbors)

# 获取节点 'A' 的传入节点
in_neighbors = list(residual_digraph.predecessors('A'))
print("In-neighbors of 'A':", in_neighbors)
