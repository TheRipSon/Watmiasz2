import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for Matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
import networkx as nx
import numpy as np
import sys

# This dictionary will hold edges and their "current flow" values.
# Example: newlabels[(u, v)] = {capacity: current_flow}
edge_flow_map = {}

def edmonds_karp_or_ford_fulkerson(capacity_matrix, source, sink, method_flag):
    """
    Calculate the maximum flow using either the Ford-Fulkerson (method_flag=1) 
    or Edmonds-Karp (method_flag=0) strategy to find augmenting paths.

    :param capacity_matrix: 2D list (adjacency matrix) with capacities
    :param source: index of the source node
    :param sink: index of the sink node
    :param method_flag: 1 for BFS (Ford-Fulkerson), 0 for DFS (Edmonds-Karp)
    :return: total maximum flow from source to sink
    """

    n = len(capacity_matrix)
    flow_matrix = [[0] * n for _ in range(n)]  # keeps track of flow
    path_index = 1

    # Depending on method_flag, we use BFS or DFS to find augmenting path
    if method_flag == 1:
        path = find_augmenting_path_bfs(capacity_matrix, flow_matrix, source, sink)
    else:
        path = find_augmenting_path_dfs(capacity_matrix, flow_matrix, source, sink)

    print(f"Path: {path_index}")

    # While there is an augmenting path, send flow through it
    while path is not None:
        path_index += 1
        # Find the minimum residual capacity along this path
        min_capacity = min(
            capacity_matrix[u][v] - flow_matrix[u][v]
            for (u, v) in path
        )
        # Update flows along the path
        for (u, v) in path:
            flow_matrix[u][v] += min_capacity
            flow_matrix[v][u] -= min_capacity  # backflow for residual graph

            # Update the flow for visualization/tracking
            edge_tuple = (u, v)
            if edge_tuple not in edge_flow_map:
                # If not found in direct direction, check reverse direction
                min_capacity *= -1
                edge_tuple = (v, u)

            existing_capacity = list(edge_flow_map[edge_tuple].keys())[0]
            edge_flow_map[edge_tuple][existing_capacity] += min_capacity

            # Revert if we negated min_capacity above
            if min_capacity < 0:
                min_capacity *= -1

            print(f"Edge {u} -> {v} : {edge_flow_map[edge_tuple]}")

        print(f"Path: {path_index}")

        # Find the next augmenting path
        if method_flag == 1:
            path = find_augmenting_path_bfs(capacity_matrix, flow_matrix, source, sink)
        else:
            path = find_augmenting_path_dfs(capacity_matrix, flow_matrix, source, sink)

    # The max flow is the sum of flows out of the source
    return sum(flow_matrix[source][v] for v in range(n))


def find_augmenting_path_dfs(capacity_matrix, flow_matrix, source, sink):
    """
    Find an augmenting path using Depth-First Search (DFS).
    Returns a list of edges (u, v) representing the path if found,
    or None if not found.
    """
    stack = [source]
    paths = {source: []}

    if source == sink:
        return paths[source]

    while stack:
        u = stack.pop()
        for v in range(len(capacity_matrix)):
            # If there's available capacity in the residual graph and we haven't visited v yet
            if capacity_matrix[u][v] - flow_matrix[u][v] > 0 and v not in paths:
                paths[v] = paths[u] + [(u, v)]
                if v == sink:
                    return paths[v]
                stack.append(v)

    return None


def find_augmenting_path_bfs(capacity_matrix, flow_matrix, source, sink):
    """
    Find an augmenting path using Breadth-First Search (BFS).
    Returns a list of edges (u, v) representing the path if found,
    or None if not found.
    """
    queue = [source]
    paths = {source: []}

    if source == sink:
        return paths[source]

    while queue:
        u = queue.pop(0)
        for v in range(len(capacity_matrix)):
            # If there's available capacity in the residual graph and we haven't visited v yet
            if capacity_matrix[u][v] - flow_matrix[u][v] > 0 and v not in paths:
                paths[v] = paths[u] + [(u, v)]
                if v == sink:
                    return paths[v]
                queue.append(v)

    return None


def build_graph_visual(adjacency_matrix, labeled_edges, root_window):
    """
    Build and display an interactive graph visualization using NetworkX and Matplotlib.

    :param adjacency_matrix: 2D NumPy array of capacities (or adjacency)
    :param labeled_edges: edge_flow_map or similar dict of edge labels
    :param root_window: the Tk root window
    """
    figure, ax = plt.subplots(figsize=(5, 4))
    plt.axis('off')

    # Construct a directed graph from the matrix
    graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    positions = nx.spring_layout(graph)  # Use spring layout for better visualization

    # Draw the graph initially
    nx.draw(graph, positions, with_labels=True, ax=ax, node_color='lightblue', edge_color='gray')
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=labeled_edges, ax=ax)

    # Store positions to update on dragging
    node_positions = {node: positions[node] for node in graph.nodes}
    dragged_node = None

    def on_press(event):
        """Detect when a node is clicked."""
        nonlocal dragged_node
        for node, (x, y) in positions.items():
            if np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2) < 0.05:
                dragged_node = node
                break

    def on_motion(event):
        """Move the dragged node."""
        if dragged_node is not None and event.xdata is not None and event.ydata is not None:
            positions[dragged_node] = (event.xdata, event.ydata)
            ax.clear()
            nx.draw(graph, positions, with_labels=True, ax=ax, node_color='lightblue', edge_color='gray')
            nx.draw_networkx_edge_labels(graph, positions, edge_labels=labeled_edges, ax=ax)
            canvas.draw()

    def on_release(event):
        """Release the node after dragging."""
        nonlocal dragged_node
        dragged_node = None

    # Connect event handlers
    canvas = FigureCanvasTkAgg(figure, master=root_window)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("button_release_event", on_release)

    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)


if __name__ == '__main__':
    # Retrieve algorithm flag from command-line arguments
    # Usage example: python script.py FF or python script.py EK
    args = sys.argv[1:]
    if not args:
        sys.exit("No algorithm specified. Use 'FF' or 'EK'.")

    # Default to BFS if 'FF' (Ford-Fulkerson), else DFS if 'EK' (Edmonds-Karp)
    flow_method = 1 if args[0] == 'FF' else 0

    root = tk.Tk()

    # Read the input graph from 'input.txt'
    with open('input.txt', 'r') as file:
        line = file.readline().split(",")
        start_node = int(line[0])
        end_node = int(line[1])
        capacity_list = []
        for matrix_line in file:
            row_values = [int(num) for num in matrix_line.split(',')]
            capacity_list.append(row_values)

    capacity_array = np.array(capacity_list)
    num_nodes = len(capacity_list)

    # Build a NetworkX graph to retrieve initial capacities for labeling
    directed_graph = nx.DiGraph(capacity_array)
    initial_labels = nx.get_edge_attributes(directed_graph, "weight")

    # Initialize edge_flow_map for each edge with {capacity: 0}
    for edge in initial_labels:
        edge_flow_map[edge] = {initial_labels[edge]: 0}

    # Compute the max flow
    max_flow_value = edmonds_karp_or_ford_fulkerson(
        capacity_array.tolist(),
        start_node,
        end_node,
        flow_method
    )

    # Print which algorithm was used
    if flow_method == 1:
        print("Using Ford-Fulkerson (BFS) approach")
    else:
        print("Using Edmonds-Karp (DFS) approach")

    print("Max flow value:", max_flow_value)

    # Visualize the resulting graph
    build_graph_visual(capacity_array, edge_flow_map, root)
    root.mainloop()
