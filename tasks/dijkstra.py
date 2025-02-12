import networkx as nx

def dijkstra_all_paths(grid_graph, targets):
    """
    Находит кратчайшие пути от каждой доступной вершины до каждой цели с использованием алгоритма Дейкстры.
    
    :param grid_graph: Взвешенный граф (networkx.Graph)
    :param targets: Список целевых вершин
    :return: Словарь {целевой узел: {узел: [список узлов пути]}}
    """
    all_paths = {target: {} for target in targets}  # Словарь для хранения кратчайших путей
    
    for target in targets:
        for node in grid_graph.nodes():
            if nx.has_path(grid_graph, node, target):  # Проверяем, существует ли путь
                path = nx.shortest_path(grid_graph, source=node, target=target, weight='weight')
                all_paths[target][node] = path
    
    return all_paths

# Пример создания графа-сетки (4x4) с весами
G = nx.grid_2d_graph(4, 4)  # Создаём 4x4 сетку
for (u, v) in G.edges():
    G[u][v]['weight'] = 1  # Назначаем вес 1 всем рёбрам (можно менять)

target_nodes = [(3, 3), (0, 0), (1,1)]  # Несколько целевых точек
shortest_paths = dijkstra_all_paths(G, target_nodes)

# Выводим пути
for target, paths in shortest_paths.items():
    for start, path in paths.items():
        print(f"Путь из {start} -> {target}: {path}")
