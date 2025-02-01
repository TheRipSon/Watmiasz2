import json
import networkx as nx
import matplotlib.pyplot as plt
import concurrent.futures
import time
from collections import defaultdict
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

###############################################################################
# Load data
###############################################################################
with open('cities_data.json') as f:
    city_data = json.load(f)

with open('cities_coordinates.json') as f:
    city_coords = json.load(f)

###############################################################################
# Build directed graph "G" with base_cost and capacity
###############################################################################
G = nx.DiGraph()

for city, info in city_data["cites"].items():
    G.add_node(city, pos=(city_coords[city][0], city_coords[city][1]))
    for route in info["routes"]:
        end_city = route["end_node_name"]
        road_length = route["road_length"]
        speed_lim = route["speed_limit"]
        lanes = route["number_of_lanes"]

        # base_cost example
        base_cost = road_length / (speed_lim * lanes)

        # capacity example
        capacity_val = speed_lim * lanes

        # If roads are bidirectional, add both directions:
        G.add_edge(
            city,
            end_city,
            base_cost=base_cost,
            road_length=road_length,
            speed_limit=speed_lim,
            number_of_lanes=lanes,
            capacity=capacity_val
        )
        G.add_edge(
            end_city,
            city,
            base_cost=base_cost,
            road_length=road_length,
            speed_limit=speed_lim,
            number_of_lanes=lanes,
            capacity=capacity_val
        )

###############################################################################
# Draggable subgraph for multiple routes (SECOND GRAPH)
###############################################################################
def make_draggable_subgraph(H, routes, pos, colors):
    """
    Make a draggable graph for the subgraph with routes shown in different colors.
    Off-route edges are in gray.
    """

    root = tk.Tk()
    root.title("Routes Visualization (Draggable)")

    figure, ax = plt.subplots(figsize=(6, 5))
    ax.axis('off')

    # Gather all edges used by the routes so we can color them differently
    route_edges_set = set()
    for idx, path in enumerate(routes):
        for i in range(len(path) - 1):
            route_edges_set.add((path[i], path[i + 1]))

    # Draw all nodes
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color='lightblue')
    nx.draw_networkx_labels(H, pos, ax=ax)

    # Draw each route with a different color
    for idx, path in enumerate(routes):
        edge_pairs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=edge_pairs,
            edge_color=[colors(idx)],
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            ax=ax,
        )

    # Draw edges not in the route set in gray
    off_route_edges = [(u, v) for (u, v) in H.edges() if (u, v) not in route_edges_set]
    nx.draw_networkx_edges(
        H,
        pos,
        edgelist=off_route_edges,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        ax=ax
    )

    dragged_node = None

    def on_press(event):
        nonlocal dragged_node
        if event.inaxes != ax:
            return
        x_click, y_click = event.xdata, event.ydata
        # Find the closest node
        for n, (xx, yy) in pos.items():
            dist = np.sqrt((x_click - xx) ** 2 + (y_click - yy) ** 2)
            if dist < 0.1:
                dragged_node = n
                break

    def on_motion(event):
        if event.inaxes != ax:
            return
        if dragged_node is not None:
            # Update node position
            pos[dragged_node] = (event.xdata, event.ydata)
            ax.clear()
            ax.axis('off')

            # Redraw nodes
            nx.draw_networkx_nodes(H, pos, ax=ax, node_color='lightblue')
            nx.draw_networkx_labels(H, pos, ax=ax)

            # Redraw route edges
            for idx, path in enumerate(routes):
                edge_pairs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                nx.draw_networkx_edges(
                    H,
                    pos,
                    edgelist=edge_pairs,
                    edge_color=[colors(idx)],
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=15,
                    ax=ax,
                )

            # Redraw off-route edges
            off_route_edges = [(u, v) for (u, v) in H.edges() if (u, v) not in route_edges_set]
            nx.draw_networkx_edges(
                H,
                pos,
                edgelist=off_route_edges,
                edge_color='gray',
                arrows=True,
                arrowstyle='-|>',
                arrowsize=15,
                ax=ax
            )

            canvas.draw()

    def on_release(event):
        nonlocal dragged_node
        dragged_node = None

    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("button_release_event", on_release)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    return root, figure

###############################################################################
# Penalty-based iterative approach (unique routes)
###############################################################################
def find_k_routes_with_penalties(graph, source, target, k=4, alpha=1.0, max_attempts=15):
    usage_count = defaultdict(int)
    routes = []
    unique_paths = set()
    attempts = 0

    while len(routes) < k and attempts < max_attempts:
        attempts += 1

        # Build a new penalized graph for path-finding (we'll treat it as undirected here)
        H = nx.Graph()
        for (u, v) in graph.edges():
            base = graph[u][v]['base_cost']
            penalty_factor = 1.0 + alpha * (usage_count[u] + usage_count[v])
            cost = base * penalty_factor

            H.add_node(u, pos=graph.nodes[u]['pos'])
            H.add_node(v, pos=graph.nodes[v]['pos'])
            H.add_edge(u, v, cost=cost)

        try:
            path = nx.shortest_path(H, source=source, target=target, weight='cost')
        except nx.NetworkXNoPath:
            break

        path_tuple = tuple(path)
        if path_tuple in unique_paths:
            continue  # skip duplicates

        routes.append(path)
        unique_paths.add(path_tuple)

        # Update usage count for nodes in this path
        for node in path:
            usage_count[node] += 1

    return routes

###############################################################################
# Helper: run a function with a timeout
###############################################################################
def run_with_timeout(func, timeout=5):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

###############################################################################
# Max Flow (Edmonds-Karp) using BFS
###############################################################################
def find_augmenting_path_bfs(cap_matrix, flow_matrix, source, sink):
    queue = [source]
    paths = {source: []}
    if source == sink:
        return paths[source]

    while queue:
        u = queue.pop(0)
        for v in range(len(cap_matrix)):
            residual = cap_matrix[u][v] - flow_matrix[u][v]
            if residual > 0 and v not in paths:
                paths[v] = paths[u] + [(u, v)]
                if v == sink:
                    return paths[v]
                queue.append(v)

    return None

def edmonds_karp(cap_matrix, source, sink):
    n = len(cap_matrix)
    flow_matrix = [[0] * n for _ in range(n)]
    path = find_augmenting_path_bfs(cap_matrix, flow_matrix, source, sink)

    path_count = 0
    while path is not None:
        path_count += 1
        # Find min residual along path
        flow = min(cap_matrix[u][v] - flow_matrix[u][v] for (u, v) in path)
        # Update flow
        for (u, v) in path:
            flow_matrix[u][v] += flow
            flow_matrix[v][u] -= flow

        print(f"Found augmenting path #{path_count} -> Flow added: {flow}")

        path = find_augmenting_path_bfs(cap_matrix, flow_matrix, source, sink)

    max_flow_val = sum(flow_matrix[source][i] for i in range(n))
    return max_flow_val, flow_matrix

###############################################################################
# Build capacity matrix from subgraph + node_list
###############################################################################
def build_capacity_matrix(graph, node_list):
    idx_map = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)
    matrix = [[0] * n for _ in range(n)]

    for (u, v) in graph.edges():
        cap = graph[u][v].get('capacity', 0)
        i = idx_map[u]
        j = idx_map[v]
        matrix[i][j] = cap

    return matrix

###############################################################################
# Draggable flow figure (THIRD GRAPH)
###############################################################################
def make_draggable_graph(H, edge_labels):
    """
    Build a Tkinter-based figure, let user drag nodes, show net-flow edge labels.
    The initial positions come from the 'pos' attribute on each node.
    """

    root = tk.Tk()
    root.title("Flow Visualization (Draggable)")

    figure, ax = plt.subplots(figsize=(6, 5))
    ax.axis('off')

    pos = nx.get_node_attributes(H, 'pos')

    nx.draw_networkx_nodes(H, pos, ax=ax, node_color='lightblue')
    nx.draw_networkx_labels(H, pos, ax=ax)
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color='green',
                           arrows=True, arrowstyle='-|>', arrowsize=15)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels,
                                 font_color='red', ax=ax)

    dragged_node = None

    def on_press(event):
        nonlocal dragged_node
        if event.inaxes != ax:
            return
        x_click, y_click = event.xdata, event.ydata
        # find closest node
        for n, (xx, yy) in pos.items():
            dist = np.sqrt((x_click - xx) ** 2 + (y_click - yy) ** 2)
            if dist < 0.1:
                dragged_node = n
                break

    def on_motion(event):
        if event.inaxes != ax:
            return
        if dragged_node is not None:
            pos[dragged_node] = (event.xdata, event.ydata)
            ax.clear()
            ax.axis('off')
            nx.draw_networkx_nodes(H, pos, ax=ax, node_color='lightblue')
            nx.draw_networkx_labels(H, pos, ax=ax)
            nx.draw_networkx_edges(H, pos, ax=ax, edge_color='green',
                                   arrows=True, arrowstyle='-|>', arrowsize=15)
            nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels,
                                         font_color='red', ax=ax)
            canvas.draw()

    def on_release(event):
        nonlocal dragged_node
        dragged_node = None

    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("button_release_event", on_release)
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    return root, figure

###############################################################################
# Interactive map (pick source & target, find routes, compute max flow)
###############################################################################
def interactive_map(G, alpha=1.0, timeout=5):
    fig, ax = plt.subplots()
    plt.title("Click two different cities to pick source & target")
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the entire directed graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                           arrows=True, arrowstyle='-|>', arrowsize=15)

    selected_cities = []

    def on_click(event):
        if event.inaxes != ax:
            return

        x_click, y_click = event.xdata, event.ydata
        min_dist = float('inf')
        chosen_city = None
        for city, (cx, cy) in pos.items():
            dist = np.hypot(cx - x_click, cy - y_click)
            if dist < min_dist:
                min_dist = dist
                chosen_city = city

        if chosen_city is None:
            return

        print(f"Selected city: {chosen_city}")
        selected_cities.append(chosen_city)

        # Once two cities are selected
        if len(selected_cities) == 2:
            source, target = selected_cities
            selected_cities.clear()

            print(f"You selected source = {source}, target = {target}")
            k_input = input("How many routes do you want to find? (k) => ")
            try:
                k = int(k_input)
            except ValueError:
                print("Invalid integer, defaulting to k=1.")
                k = 1

            # define route-finding task
            def route_finding_task():
                return find_k_routes_with_penalties(G, source, target, k, alpha=alpha)

            print(f"Finding up to {k} unique routes between {source} and {target} "
                  f"with alpha={alpha}, timeout={timeout}s")
            routes = run_with_timeout(route_finding_task, timeout=timeout)

            if routes is None:
                print("Route-finding timed out.")
                return
            if not routes:
                print("No routes found.")
                return

            # Collect route nodes & edges
            route_nodes = set()
            for idx, path in enumerate(routes, start=1):
                print(f"Route {idx}: {path}")
                route_nodes.update(path)

            # Build subgraph of the route nodes
            H = nx.DiGraph()
            for n in route_nodes:
                H.add_node(n, pos=G.nodes[n]['pos'])
            for (u, v) in G.edges():
                if u in route_nodes and v in route_nodes:
                    H.add_edge(u, v, **G[u][v])

            # Make the second graph draggable, with each route in a different color
            pos_sub = nx.get_node_attributes(H, 'pos')
            colors = plt.cm.get_cmap('tab10', len(routes))
            root, figure = make_draggable_subgraph(H, routes, pos_sub, colors)
            figure.suptitle(f"Subgraph for up to {k} penalized routes\n"
                            f"{source} -> {target}", fontsize=12)
            root.mainloop()

            # Prepare max flow on subgraph
            node_list = list(route_nodes)
            if source not in node_list or target not in node_list:
                print("Source/target not found in subgraph. Skipping max flow.")
                return

            cap_matrix = build_capacity_matrix(H, node_list)
            s_idx = node_list.index(source)
            t_idx = node_list.index(target)

            print("Calculating max flow (Edmonds-Karp)...")
            max_flow_val, flow_matrix = edmonds_karp(cap_matrix, s_idx, t_idx)
            print("Max flow value:", max_flow_val)
            print("Final flow matrix (forward/backward) =>")
            for row in flow_matrix:
                print(row)

            # Build net flow for labeling
            net_flow_labels = {}
            for i in range(len(node_list)):
                for j in range(len(node_list)):
                    cap = cap_matrix[i][j]
                    if cap > 0:
                        fwd = flow_matrix[i][j]
                        rev = flow_matrix[j][i]
                        if fwd > rev:
                            actual_flow = fwd - rev
                            if actual_flow > 0:
                                # dividing by 2 because roads are bidirectional,
                                # and Edmonds-Karp effectively doubles them
                                net_flow_labels[(node_list[i], node_list[j])] = f"{actual_flow/2}/{cap}"
                        else:
                            actual_flow = rev - fwd
                            if actual_flow > 0:
                                net_flow_labels[(node_list[j], node_list[i])] = f"{actual_flow/2}/{cap}"

            # THIRD GRAPH: Draggable flow figure
            root_flow, figure_flow = make_draggable_graph(H, net_flow_labels)
            figure_flow.suptitle(f"Flow distribution {source} -> {target}"
                                 f"\n(MaxFlow={max_flow_val})", fontsize=12)
            root_flow.mainloop()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    # Start interactive picking on the main map
    interactive_map(G, alpha=1.0, timeout=5)
