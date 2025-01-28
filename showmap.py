import json
import networkx as nx
import matplotlib.pyplot as plt
import concurrent.futures
import time
from collections import defaultdict
import numpy as np

# install all dependencies based on your requirements.txt

###############################################################################
# Load data
###############################################################################
with open('cities_data.json') as f:
    city_data = json.load(f)

with open('cities_coordinates.json') as f:
    city_coords = json.load(f)

###############################################################################
# Build the original graph with "base_cost"
###############################################################################
G = nx.Graph()

for city, info in city_data["cites"].items():
    G.add_node(city, pos=(city_coords[city][0], city_coords[city][1]))
    for route in info["routes"]:
        end_city = route["end_node_name"]
        base_cost = route["road_length"] / (route["speed_limit"] * route["number_of_lanes"])
        G.add_edge(
            city,
            end_city,
            base_cost=base_cost,
            road_length=route["road_length"],
            speed_limit=route["speed_limit"],
            number_of_lanes=route["number_of_lanes"],
        )

###############################################################################
# Penalty-based iterative approach (unique routes)
###############################################################################
def find_k_routes_with_penalties(graph, source, target, k=4, alpha=1.0, max_attempts=15):
    """
    Iteratively find up to k unique routes. We penalize repeated city usage so the
    pathfinder is nudged to choose less-used cities:
        cost(u->v) = base_cost(u->v) * (1 + alpha*(usage[u] + usage[v]))

    We ensure each newly found route is not exactly the same as a previously found one.
    If we hit a duplicate route, we skip it and try again until max_attempts is reached
    or we have k unique routes.
    """
    usage_count = defaultdict(int)
    routes = []
    unique_paths = set()
    attempts = 0

    while len(routes) < k and attempts < max_attempts:
        attempts += 1

        # Build a new "penalized" graph with updated costs
        H = nx.Graph()
        for (u, v) in graph.edges():
            base_cost = graph[u][v]['base_cost']
            # usage-based penalty factor
            penalty_factor = 1.0 + alpha*(usage_count[u] + usage_count[v])
            cost = base_cost * penalty_factor

            H.add_node(u, pos=graph.nodes[u]['pos'])
            H.add_node(v, pos=graph.nodes[v]['pos'])
            H.add_edge(u, v, cost=cost)

        # Attempt to find a single shortest path
        try:
            path = nx.shortest_path(H, source=source, target=target, weight='cost')
        except nx.NetworkXNoPath:
            break  # No more paths can be found

        path_tuple = tuple(path)
        if path_tuple in unique_paths:
            # It's a duplicate, skip it (do not update usage_count)
            continue

        # It's new, accept it
        routes.append(path)
        unique_paths.add(path_tuple)

        # Update usage
        for node in path:
            usage_count[node] += 1

    return routes

###############################################################################
# Helper function to run route-finding with a timeout
###############################################################################
def run_with_timeout(func, timeout=5):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

###############################################################################
# Interactive picking of source and target
###############################################################################
def interactive_map(G, alpha=1.0, timeout=5):
    """
    1) Displays the full graph.
    2) Lets user click any two different cities.
    3) Asks how many routes (k) to find.
    4) Computes routes & displays a subgraph containing all edges that connect
       the nodes used by those routes.
    """
    fig, ax = plt.subplots()
    plt.title("Click two different cities to pick source & target")
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the initial full graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, ax=ax)

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
            # Clear selection to allow picking again later
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

            print(f"Finding up to {k} unique routes between {source} and {target} with alpha={alpha}, timeout={timeout}s")
            routes = run_with_timeout(route_finding_task, timeout=timeout)

            if routes is None:
                print("Route-finding timed out.")
                return
            if not routes:
                print("No routes found.")
                return

            # Collect all nodes used in these routes
            route_nodes = set()
            for idx, path in enumerate(routes, start=1):
                print(f"Route {idx}: {path}")
                route_nodes.update(path)

            # Build a subgraph *including all edges* between those route nodes
            H = nx.Graph()
            for n in route_nodes:
                H.add_node(n, pos=G.nodes[n]['pos'])
            # Add any edges that exist in G among those nodes
            for (u, v) in G.edges():
                if u in route_nodes and v in route_nodes:
                    # Copy relevant attributes to the subgraph
                    H.add_edge(u, v, **G[u][v])

            print("\nNodes included in routes:")
            print(route_nodes)
            print("Edges among these route nodes (in original graph):")
            for e in H.edges():
                print(e)

            # Draw subgraph in a new figure
            fig2, ax2 = plt.subplots()
            plt.title(f"Subgraph for up to {k} penalized routes between {source} and {target}")
            pos_sub = nx.get_node_attributes(H, 'pos')
            nx.draw(
                H,
                pos_sub,
                with_labels=True,
                node_color='lightblue',
                edge_color='red',
                node_size=500,
                ax=ax2
            )
            plt.show()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    # Run interactive map
    interactive_map(G, alpha=1.0, timeout=5)
