import time
import zen
import numpy as np
from zen.algorithms.community import louvain

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

def louvain_community_detection(G,network_type):
    cset = louvain(G)

    comm_dict = {}
    comm_list = np.zeros(G.num_nodes)
    for i,community in enumerate(cset.communities()):
        comm_dict[i] = community.nodes()
        comm_list[community.nodes_()] = i

    q,qmax = modularity(G,comm_dict,comm_list)

    with open(network_type+'_network_info.txt','a') as fObj:
        fObj.write('%d communities found.\n'%(i+1))
        fObj.write('Q:            %.3f\n'%q)
        fObj.write('Normalized Q: %.3f\n'%(q/qmax))

def main(network_type):
    G = zen.io.gml.read('amazon_reviews_'+network_type+'.gml',weight_fxn=lambda x: x['weight'])
    start_time = time.time()
    louvain_community_detection(G,network_type)
    stop_time = time.time()
    with open(network_type+'_network_info.txt','a') as fObj:
        fObj.write("Elapsed time (seconds): %.3f"%(stop_time - start_time))
