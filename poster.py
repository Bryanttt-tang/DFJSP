import networkx as nx
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,2))
ax.set_axis_off()

nodes = ["IoT Sensors", "Big Data", "Digital Twin", "DRL Scheduler", "Shop Floor"]
xpos = range(len(nodes))

for i, n in enumerate(nodes):
    ax.text(i*2, 0, n, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#2E86AB", ec="black", alpha=0.8, color="white"),
            color="white", fontsize=12)
    if i < len(nodes)-1:
        ax.arrow(i*2+0.8, 0, 0.4, 0, head_width=0.2, head_length=0.2, fc="black", ec="black")

plt.savefig("Smart_Manufacturing.png", transparent=True, dpi=300)


fig, ax = plt.subplots(figsize=(8,3))
machines = ["M1","M2","M3"]
jobs = {"J1":[("M1",0,2),("M2",2,4)], "J2":[("M2",0,3),("M3",3,5)], "J3":[("M1",1,4)]}

colors = {"J1":"#1f77b4","J2":"#2ca02c","J3":"#ff7f0e"}

for j, ops in jobs.items():
    for m, start, end in ops:
        y = machines.index(m)
        ax.barh(y, end-start, left=start, height=0.4, color=colors[j], edgecolor="black")
        ax.text((start+end)/2, y, j, ha="center", va="center", color="white")

ax.set_yticks(range(len(machines)))
ax.set_yticklabels(machines)
ax.set_xlabel("Time")
ax.set_title("Job Shop Scheduling Visualization")
plt.savefig("JobShop_Visualization.png", transparent=True, dpi=300)


# Disjunctive graph
G = nx.DiGraph()
ops = ["O11","O12","O21","O22"]
edges = [("O11","O12"),("O21","O22")]  # precedence
disj = [("O11","O21"),("O12","O22")]   # disjunctive

pos = {"O11":(0,1),"O12":(1,1),"O21":(0,0),"O22":(1,0)}

plt.figure(figsize=(6,3))
nx.draw(G, pos, with_labels=True, node_color="lightblue", edgecolors="black")
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="black", arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=disj, style="dashed", edge_color="red")
plt.title("Disjunctive Graph")
plt.savefig("DisjunctiveGraph.png", transparent=True, dpi=300)

# Mini Gantt
fig, ax = plt.subplots(figsize=(6,2))
machines = ["M1","M2"]
ax.barh(0,2,left=0,color="#1f77b4",edgecolor="black")
ax.text(1,0,"J1-O1",ha="center",va="center",color="white")
ax.barh(1,3,left=0,color="#2ca02c",edgecolor="black")
ax.text(1.5,1,"J2-O1",ha="center",va="center",color="white")
ax.barh(0,2,left=2,color="#1f77b4",edgecolor="black")
ax.text(3,0,"J1-O2",ha="center",va="center",color="white")
ax.barh(1,2,left=3,color="#2ca02c",edgecolor="black")
ax.text(4,1,"J2-O2",ha="center",va="center",color="white")
ax.set_yticks([0,1]); ax.set_yticklabels(machines)
plt.title("Mini Gantt Chart")
plt.savefig("Gantt.png", transparent=True, dpi=300)

