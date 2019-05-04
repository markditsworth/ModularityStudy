import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zen
from zen.algorithms.community import louvain

def katz(G,tol=0.01,max_iter=1000,alpha=0.001,beta=1):
    iteration = 0
    centrality = np.zeros(G.num_nodes)
    while iteration < max_iter:
        iteration += 1          # increment iteration count
        centrality_old = centrality.copy()

        for node in G.nodes_():
            Ax = 0
            for neighbor in G.neighbors_(node):
                weight = G.weight_(G.edge_idx_(neighbor,node))
                Ax += np.multiply(centrality[neighbor],weight)

                #Ax += centrality[neighbor]      #exclude weight due to overflow in multiplication

            centrality[node] = np.multiply(alpha,Ax)+beta

        if np.sum(np.abs(np.subtract(centrality,centrality_old))) < tol:
            return centrality

def modular_graph(Size1, Size2, edges1, edges2, common, katz_alpha=0.001):
    g1 = zen.generating.barabasi_albert(Size1,edges1)
    avgDeg1 = (2.0 * g1.num_edges)/g1.num_nodes
    #lcc1 = np.mean(zen.algorithms.clustering.lcc_(g1))

    g2 = zen.generating.barabasi_albert(Size2,edges2)
    avgDeg2 = (2.0 * g2.num_edges)/g2.num_nodes
    #lcc2 = np.mean(zen.algorithms.clustering.lcc_(g2))

    Size = Size1 + Size2
    G = zen.Graph()
    for i in range(Size):
        G.add_node(i)

    for edge in g1.edges_iter():
        u = edge[0]
        v = edge[1]
        G.add_edge(u,v)

    for edge in g2.edges_iter():
        u = edge[0]+Size1
        v = edge[1]+Size1
        G.add_edge(u,v)

    # Select random pairs of nodes to connect the subgraphs
    join_nodes = np.empty((common,2),dtype=np.int64)
    nodes1 = np.random.randint(0,Size1,size=common)
    nodes2 = np.random.randint(Size1,Size,size=common)
    join_nodes[:,0] = nodes1
    join_nodes[:,1] = nodes2

    for edge in join_nodes:
        if not G.has_edge(edge[0],edge[1]):
            G.add_edge(edge[0],edge[1])

    return G

def modularity(G,classDict,classList):
    Q = zen.algorithms.modularity(G,classDict)
    # Maximum Modularity
    count=0.0
    for e in G.edges():
        n1 = G.node_idx(e[0])
        n2 = G.node_idx(e[1])
        if classList[n1] == classList[n2]:
            count += 1
    same = count / G.num_edges
    rand = same - Q
    qmax = 1 - rand
    return Q, qmax

def main(exists=False):
    if not exists:
        G_synth = modular_graph(500,500,2,8,100,katz_alpha=1e-4)
        zen.io.gml.write(G_synth,'/opt/adhoc.gml')
    else:
        G_synth = zen.io.gml.read('/opt/adhoc.gml')

    print "Nodes: %d"%G_synth.num_nodes
    print "Edges: %d"%G_synth.num_edges

    cset = louvain(G_synth)
    comm_dict = {}
    comm_list = np.zeros(G_synth.num_nodes)
    for i, community in enumerate(cset.communities()):
        comm_dict[i] = community.nodes()
        comm_list[community.nodes_()]=i

    q,qmax = modularity(G_synth,comm_dict,comm_list)
    print '%d communities found.'%(i+1)
    print 'Q:            %.3f'%q
    print 'Qmax:         %.3f'%qmax
    print 'Normalized Q: %.3f'%(q/qmax)

    evc = zen.algorithms.eigenvector_centrality_(G_synth)
    evc = evc - np.min(evc)
    evc = evc / np.max(evc)
    kc = katz(G_synth,alpha=1e-4)
    kc = kc - np.min(kc)
    kc = kc / np.max(kc)
    GROUP = [1,2,3,4]
    fig = plt.plot(figsize=(12,8))
    for i,com in enumerate(cset.communities()):
        if i+1 in GROUP:
            nodes = com.nodes_()
            plt.scatter(evc[nodes],kc[nodes],s=7,label='%d'%(i+1))
    plt.xlabel('Eigenvector centrality (normalized)',fontsize=14)
    plt.ylabel('Katz centrality (normalized)',fontsize=14)
    plt.legend()
    plt.xlim([-0.04,1])
    plt.ylim([-0.04,1])
    plt.savefig('/opt/louvain1.png')
    plt.close()
    fig = plt.plot(figsize=(12,8))
    for i,com in enumerate(cset.communities()):
        if i+1 not in GROUP:
            nodes = com.nodes_()
            plt.scatter(evc[nodes],kc[nodes],s=7,label='%d'%(i+1))
    plt.xlabel('Eigenvector centrality (normalized)',fontsize=14)
    plt.ylabel('Katz centrality (normalized)',fontsize=14)
    plt.legend()
    plt.savefig('/opt/louvain2.png')

    idx = np.where(evc>0.025)[0]

    nodes = G_synth.nodes_()
    c1 = nodes[idx]
    c2 = nodes[~idx]

    ClassDict = {}
    ClassDict[0] = [G_synth.node_object(x) for x in c1]
    ClassDict[1] = [G_synth.node_object(x) for x in c2]

    ClassList = np.zeros(G_synth.num_nodes)
    ClassList[c2] = 1

    q,qmax = modularity(G_synth,ClassDict,ClassList)
    print "manual community assignment:"
    print q
    print qmax
    print q/qmax

if __name__ == '__main__':
    main(exists=True)
