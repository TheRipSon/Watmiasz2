import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
import networkx as nx
import numpy as np
import sys as sys

newlabels = {}


def max_flow(c, s, t, metod):
    f = [[0] * n for i in range(n)]
    i = 1
    print(f"Sciezka: {i}")
    if metod == 1:
        path = bfs(c, f, s, t)
    else:
        path = dfs(c, f, s, t)
    while path is not None:
        i += 1
        flow = min(c[u][v] - f[u][v] for u, v in path)
        for u, v in path:
            f[u][v] += flow
            f[v][u] -= flow
            a = (u, v)
            if newlabels.get(a) is None:
                flow *= -1
                a = (v, u)
            z = list(newlabels[a].keys())[0]
            newlabels[a][z] += flow
            if flow < 0:
                flow *= -1
            print(f"{u},{v} : {newlabels[a]}")
        print(f"Sciezka: {i}")
        if bfs:
            path = bfs(c, f, s, t)
        else:
            path = dfs(c, f, s, t)
    return sum(f[s][i] for i in range(n))


def dfs(c, f, s, t):
    stack = [s]
    paths = {s: []}
    if s == t:
        return paths[s]
    while (stack):
        u = stack.pop()
        for v in range(len(C)):
            if (c[u][v] - f[u][v] > 0) and v not in paths:
                paths[v] = paths[u] + [(u, v)]
                if v == t:
                    return paths[v]
                stack.append(v)
    return None


# find path by using BFS
def bfs(c, f, s, t):
    queue = [s]
    paths = {s: []}
    if s == t:
        return paths[s]
    while queue:
        u = queue.pop(0)
        for v in range(len(c)):
            if (c[u][v] - f[u][v] > 0) and v not in paths:
                paths[v] = paths[u] + [(u, v)]
                if v == t:
                    return paths[v]
                queue.append(v)
    return None


def build_graph(matrix, labels):
    # global table
    matrix = np.array(matrix)
    f = plt.figure(figsize=(5, 4))
    plt.axis('off')
    G = nx.from_numpy_array(matrix)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        exit("No algoritm selected")
    metod = 1
    if args[0] == 'FF':
        metod = 1
    if args[0] == 'EK':
        metod = 0
    root = tk.Tk()

    with open('input.txt', 'r') as q:
        start, end = q.readline().split(",")
        C = [[int(num) for num in line.split(',')] for line in q]
    start = int(start)
    end = int(end)

    A = np.matrix(C)
    n = len(C)
    G = nx.DiGraph(A)
    labels = nx.get_edge_attributes(G, "weight")
    for i in labels:
        newlabels[i] = {labels[i]: 0}
    max_flow_value = max_flow(C, start, end, metod)
    if metod == 1:
        print("Ford-Fulkerson algorithm")
    else:
        print("Edmonds-Karp algorithm")
    print("max_flow_value is: ", max_flow_value)
    build_graph(A, newlabels)

    root.mainloop()
