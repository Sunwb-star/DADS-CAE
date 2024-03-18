import sys
from collections import deque
from decimal import Decimal

import networkx as nx


def create_residual_network(origin_digraph):
    """
    根据传入的原始有向图构建初始化残差网络图
    初始的residual_network就是origin digraph的拷贝
    :param origin_digraph: 原始构建好的有向图
    :return: 构建好的初始残差图 residual_graph
    """
    return origin_digraph.copy()


def bfs_for_level_digraph(residual_digraph):
    """
    根据传入的残留图residual_digraph，使用BFS增广构建层次图level_digraph
    :param residual_digraph: 残差网络
    :return: 构建好的层级网络信息_level_dict
             以及最后一个节点(汇入点)是否在dict中(boolean):cloud_node_in_dict,
             用于dinic算法终止条件的判断，当汇点不在层次网络中时就意味着在残留网络中不存在一条从源点到汇点的路径，即没有增广路，就停止增广
    """
    # 记录节点是否已经被被访问过 同时记录节点的层数
    level_dict = {}
    # 源点设置为'edge'
    start_node = 'edge'
    # 将原点的层次设置为1
    level_dict[start_node] = 1
    # 初始化一个双端队列 用于BFS遍历
    Q = deque()
    # 将起始节点添加到队列当中，append表示从右侧添加
    Q.append(start_node)
    # 开始进行BFS的遍历 -> 构建层次图level_digraph
    while True:
        # 如果双端队列的长度为0的话就表示残差图中的所有的节点已经使用BFS遍历完了直接跳出循环即可
        if len(Q) == 0:
            break
        # print("-------------")w
        # popleft表示从左侧弹出上一层次的节点
        node = Q.popleft()
        # print(f"弹出 : {node}")
        now_level = level_dict[node]
        # neighbor_node是当前节点的后继节点，当前节点的传入节点不算在邻居节点当中
        for neighbor_nodes in nx.neighbors(residual_digraph, node):
            # 如果neighbor_nodes已经在队列里面或者邻居节点已经进行了层次划分，就不需要进行重复添加
            # 同时需要在残留网络图中从该节点到邻居节点的边的容量是大于0的
            if (neighbor_nodes not in level_dict.keys()) and (neighbor_nodes not in Q) \
                    and residual_digraph.get_edge_data(node, neighbor_nodes)["capacity"] > 0:
                # 将邻居节点加入到层次图当中，记录邻居节点的层次
                level_dict[neighbor_nodes] = now_level + 1
                # 在双端队列中添加该邻居节点
                Q.append(neighbor_nodes)

    # 判断汇入节点end_node是否已经保存在层级图中
    end_node = 'cloud'
    cloud_node_in_dict = end_node in level_dict.keys()
    # 返回层级图以及判断dinic算法终止的标志cloud_node_in_dict
    return level_dict, cloud_node_in_dict


def dfs_once(residual_graph, level_dict, dfs_start_node, augment_value):
    """
    使用DFS算法来不断选取增广路径，一次DFS可以实现多次增广，并在DFS过程中不断修改residual_graph的权重值
    在层次网络中用一次DFS过程进行增广，DFS执行完毕，该阶段的增广也执行完毕。
    :param residual_graph: 残留网络信息
    :param level_dict: 层级网络信息
    :param dfs_start_node: DFS的出发点
    :param augment_value: 此次增广的增广值
    :return: 返回增广路径的值
    """
    # augment_value为增广值
    tmp = augment_value
    # 汇入点是"cloud"
    end_node = "cloud"
    # 首先排除特殊情况，即DFS算法的起始点和最终的汇入点是相同的
    if dfs_start_node == end_node:
        return augment_value
    # 遍历图中所有顶点
    for node in residual_graph.nodes():
        # 寻找起始节点在层次图中的下一层次的节点：
        if level_dict[dfs_start_node] + 1 == level_dict[node]:
            # capacity = 0的话就表示该条路径已经没有容量了，可以不通过这个路径
            if (residual_graph.has_edge(dfs_start_node, node) and
                    residual_graph.get_edge_data(dfs_start_node, node)["capacity"] > 0):
                # 获取当前边的容量capacity
                capacity = residual_graph.get_edge_data(dfs_start_node, node)["capacity"]
                # print(f"{dfs_start_node} -> {node} : {capacity}")
                # 开始进行DFS算法找到一个增广路径，并记录增广值（根据木桶效应取最小值）
                flow_value = dfs_once(residual_graph, level_dict, node, min(tmp, capacity))
                # print(f"flow value : {flow_value}")
                # 增加反向边或者修改反向边的值
                if flow_value > 0:
                    # 如果flow_value大于0并且没有反向边的话就增加反向边
                    if not residual_graph.has_edge(node, dfs_start_node):
                        residual_graph.add_edge(node, dfs_start_node, capacity=flow_value)
                    # 如果有反向边的话，就修改反向边的值
                    else:
                        neg_flow_value = residual_graph.get_edge_data(node, dfs_start_node)["capacity"]
                        residual_graph.add_edge(node, dfs_start_node, capacity=flow_value + neg_flow_value)

                # 修改残留图中的正向边的容量
                # print(f"{dfs_start_node} -> {node} : {capacity-flow_value}")
                # print("-------------------------------")
                residual_graph.add_edge(dfs_start_node, node, capacity=capacity - flow_value)
                # 如果正向边的边权重为0，就可以删除掉这个边了，防止level digraph构建错误
                if capacity - flow_value <= 0:
                    residual_graph.remove_edge(dfs_start_node, node)
                # 进行数值更新
                tmp -= flow_value
    # 返回数据
    return augment_value - tmp


def dinic_algorithm(origin_digraph):
    """
    对有向图使用Dinic算法找到最大流和最小割的解决策略
    :param origin_digraph: 原始构建好的有向图
    :return: min_cut_value, reachable, non_reachable
    """

    # min_cut_value表示最小割的割数
    min_cut_value = 0
    # 设定一个无穷大的数
    inf = sys.maxsize

    # 通过原始的有向图创建一个初始的残留网络residual_digraph
    # 初始的残留网络就是原始网络的拷贝
    residual_graph = create_residual_network(origin_digraph)
    # print(residual_graph.edges(data=True))
    # 对残留网络进行容量替换，将其容量从int类型替换为Decimal类型，用于精确计算，其本身的网络架构并没有改变
    for edge in residual_graph.edges(data=True):
        u = edge[0]
        v = edge[1]
        # 每一条edge的内容是这样的：('edge', 'v0', {'capacity': 9223372036854775807})
        # 所以edge[2]表示的就是一个字典，字典的键'capacity'对应的是表示该条边的容量
        # 创建Decimal对象，使用转换后的容量字符串作为构造函数的参数，能够准确表示边的容量，防止出现精度问题。
        # Decimal类提供了精确的十进制运算，避免了浮点数的精度问题。
        # quantize(Decimal('0.000')): 对Decimal对象进行量化操作，指定精度为小数点后三位（0.000）。
        # 这一步可以理解为对容量进行舍入，确保精度在小数点后三位。
        c = Decimal(str(edge[2]['capacity'])).quantize(Decimal('0.000'))
        # print(u, v, c)
        residual_graph.add_edge(u, v, capacity=c)

    # 通过bfs算法构建level_dict信息，也可以当成构建层次图level_graph
    level_dict, cloud_node_in_dict = bfs_for_level_digraph(residual_graph)
    # 当cloud_node_in_dict为true也就是汇点仍然存在于层次图当中，意味着残留网络中仍然存在着增广路径
    while cloud_node_in_dict:
        # print("bfs construction")
        # 首先进行一次dfs遍历
        dfs_value = dfs_once(residual_graph, level_dict, dfs_start_node="edge", augment_value=inf)
        # 更新最小割的割数
        min_cut_value += dfs_value
        # print(dfs_value)
        # dfs_value > 0说明还可以继续进行DFS搜索其他增广路径
        while dfs_value > 0:
            # print(residual_graph.edges(data=True))
            # print("dfs search")
            dfs_value = dfs_once(residual_graph, level_dict, dfs_start_node="edge", augment_value=inf)
            min_cut_value += dfs_value

        # 当本阶段DFS遍历结束之后，重新根据BFS算法生成新的层次图level_digraph进行循环，直到终点不能表示在level_digraph中
        level_dict, cloud_node_in_dict = bfs_for_level_digraph(residual_graph)

    # 根据最后的residual_graph(level_dict)，从edge可以到达的点属于reachable，其他顶点属于non_reachable
    reachable, non_reachable = set(), set()
    for node in residual_graph:
        if node in level_dict.keys():
            reachable.add(node)
        else:
            non_reachable.add(node)
    # 返回最小割的割数以及edge和cloud分别能到达的点
    return min_cut_value, reachable, non_reachable


def get_min_cut_set(graph, min_cut_value, reachable, non_reachable):
    """
    根据min_cut_value,reachable,non_reachable参数获取最小割集，即在网络图中的哪个顶点进行切割
    :param graph: 构建好的有向图
    :param min_cut_value: 最小割的值，用于assert验证，确保划分正确
    :param reachable: 划分后可以到达的顶点
    :param non_reachable: 划分后不可到达的顶点
    :return: partition_edge 表示在DNN模型中的划分点（即不包含 edge 和 cloud 相关的边）
    """

    # 起始节点
    start = 'edge'
    # 终止节点
    end = 'cloud'
    # cut_set = []
    cut_set_sum = 0.000
    # 存储要划分的点
    graph_partition_edge = []
    # 遍历边缘端可以到达的节点，获取当前节点以及邻居节点的信息
    for u, nbrs in ((n, graph[n]) for n in reachable):
        # 在邻居节点当中
        for v in nbrs:
            # 如果在云端可到达的节点中的话
            if v in non_reachable:
                # 如果当前节点不是'edge'，邻居节点不是'cloud'的话
                if u != start and v != end:
                    # 就将该条边添加到可以分割的边种
                    graph_partition_edge.append((u, v))
                # cut_set.append((u, v))
                # 更新cut_set_sum
                cut_set_sum += graph.edges[u, v]["capacity"]

    # 通过 cut-set 得到的最小割值
    # 规范化更新cut_set_sum的小数位数
    cut_set_sum = "{:.3f}".format(round(cut_set_sum, 3))
    # 通过 dinic 算法得到的最小割值
    min_cut_value = "{:.3f}".format(round(min_cut_value, 3))
    # 确保二者相等才可以得正确的划分
    if cut_set_sum != min_cut_value:
        raise RuntimeError("Dinic算法选择的最优策略有瑕疵，请检查")
    return graph_partition_edge
