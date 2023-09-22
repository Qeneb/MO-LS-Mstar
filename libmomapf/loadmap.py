import numpy as np

def read_roadmap_from_file(edge_cost_fnames):
    f_costs = [open(fname) for fname in edge_cost_fnames]
    cost_dim = len(edge_cost_fnames)
    out = dict()
    
    num_nodes = -1
    num_edges = -1
    
    for line in f_costs[0]:
        lines = [line.strip()]
        for f in f_costs[1:]:
            lines.append(f.readline().strip())
        
        if lines[0].startswith('p'):
            index = lines[0][5:].find(' ') + 5
            num_nodes = int(lines[0][5:index])
            num_edges = int(lines[0][index + 1:])
            print(f"p:num_nodes: {num_nodes}")
            print(f"p:num_edges: {num_edges}")
            
        elif lines[0].startswith('a'):
            index1 = lines[0].find(' ', 2)
            index2 = lines[0].find(' ', index1 + 1)
            u = int(lines[0][2:index1])
            v = int(lines[0][index1 + 1:index2])
            
            cv = np.zeros(cost_dim, dtype=int)
            for i, cost_line in enumerate(lines):
                cv[i] = int(cost_line[cost_line.rfind(' ') + 1:])
            key = (u,v)
            out[key] = cv
    return out

# fnames = ["data/1-c1.gr","data/1-c2.gr","data/1-c3.gr"]
# out = read_roadmap_from_file(fnames)
# print(out)