import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zen
import clusteringAlgo as CA

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

def main():
    G = zen.io.gml.read('amazon_reviews_health.gml',weight_fxn=lambda x: x['weight'])

    evc = zen.algorithms.eigenvector_centrality_(G,weighted=True)
    evc = evc - np.min(evc)
    evc = evc / np.max(evc)

    kc = katz(G,alpha=1e-6)
    kc = kc - np.min(kc)
    kc = kc / np.max(kc)

    clusters = CA.lineClustering(evc,kc,dx=0.1)

    for c in clusters:
        plt.scatter(evc[c],kc[c],s=8)
    plt.xlabel('Eigenvector centrality (normalized)',fontsize=14)
    plt.ylabel('Katz centrality (normalized)',fontsize=14)
    plt.savefig('figures/amzn_health_ke_plot.eps',bbox_inches='tight')

    ClassDict = {}
    for i,c in enumerate(clusters):
        ClassDict[i] = [G.node_object(x) for x in c]

    ClassList = np.zeros(G.num_nodes)
    for i,c in enumerate(clusters):
        ClassList[c]=i

    q,qmax = modularity(G,ClassDict,ClassList)
    print 'Q:            %.5f'%q
    print 'Qmax:         %.5f'%qmax
    print 'Normalized Q: %.5f'%(q/qmax)

if __name__ == '__main__':
    main()
