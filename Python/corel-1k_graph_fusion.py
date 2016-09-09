#######################################################################

# Description: Merge (voc and hsv) graphs and rerank the retrieval results by performing 
# a localized PageRank algorithm or find the weighted maximum density subgraph centered at the query image.
# Details in Section 3.3-3.5 of the following paper:
# Shaoting Zhang, Ming Yang, Timothee Cour, Kai Yu, Dimitris N. Metaxas: Query Specific Fusion for Image Retrieval. ECCV (2) 2012: 660-673

# Author: Shaoting Zhang, shaoting@cs.rutgers.edu

# Date: March-2012

#######################################################################

import numpy
import cPickle
import re
import copy

class GraphFusion:

    ########### Method 1: PageRank algorithm centered at the query image.###########
    def Fusion_Graph_Laplacian(self, graph_list, num_ranks, retri_amount, ground_truth):

        # merge all graphs
        initial_graph = graph_list[0]
        for i in range(1, num_ranks):
            cur_graph = graph_list[i]
            cur_keys = cur_graph.keys()
            initial_keys = initial_graph.keys()
            for j in cur_keys:
                if j not in initial_keys:
                    initial_graph[j] = cur_graph[j]
                else:
                    nodes = cur_graph[j]
                    for k in nodes:
                        initial_graph[j].append(k)

        # build Laplacian matrix
        all_keys = initial_graph.keys()
        sorted_keys = list(numpy.sort(all_keys))
        if -1 in sorted_keys:
            sorted_keys.remove(-1) # one key is -1
        matrix_size = numpy.size(sorted_keys) 
        Laplacian = numpy.zeros([matrix_size, matrix_size])
        for cur_key in sorted_keys:
            if cur_key == -1:
                continue
            cur_weights = initial_graph[cur_key]
            neighbors = []
            neighbor_weights = []
            for weight in cur_weights:
                neighbors.append(weight[0])
                neighbor_weights.append(weight[1])
            #neighbors = numpy.unique(neighbors)
            for each_neighbor, each_weight in zip(neighbors, neighbor_weights):
                if sorted_keys.index(cur_key) != sorted_keys.index(each_neighbor):
                    # M[i,j], i is current index, j is i's neighbor
                    Laplacian[sorted_keys.index(cur_key), sorted_keys.index(each_neighbor)] += each_weight #1

        # normalize Laplacian matrix
        for i in range(matrix_size):
            if sum(Laplacian[i,:]) != 0:
            	Laplacian[i,:] = Laplacian[i,:] / sum(Laplacian[i,:])
        rank_vector = numpy.matrix([1.0/matrix_size] * matrix_size)
        rank_vector = numpy.transpose(rank_vector)
        Laplacian = numpy.transpose(Laplacian)
        for i in range(10):
            rank_vector = Laplacian * rank_vector
        rank_set = {}
        idx = 0
        for value in rank_vector:
            rank_set[idx] = value
            idx += 1

        # Solve pagerank, graph Laplacian
        selected_images_tmp = []
        for idx in sorted(rank_set, key=rank_set.get, reverse=True):
            selected_images_tmp.append(sorted_keys[idx])

        selected_images = []
        selected_images.append(ground_truth)
        selected_images_tmp.remove(ground_truth)
        selected_images.extend(selected_images_tmp)

        # Post-process. Add extra results if there is no enough candidate. use voc since it is usually better
        if len(selected_images) < retri_amount:
            voc_candidate = graph_list[1]
            voc_candidate = voc_candidate[-1]
            for i in voc_candidate:
                selected_images.append(i)

        return selected_images[0:retri_amount]

        
    ########### Method 2: find the weighted maximum density subgraph ###########
    def Fusion_Density_Subgraph(self, graph_list, num_ranks, retri_amount):

        # merge all graphs
        initial_graph = copy.deepcopy(graph_list[0])
        for i in range(1, num_ranks):
            cur_graph = graph_list[i]
            cur_keys = cur_graph.keys()
            initial_keys = initial_graph.keys()
            for j in cur_keys:
                if j not in initial_keys:
                    initial_graph[j] = cur_graph[j]
                else:
                    nodes = cur_graph[j]
                    for k in nodes:
                        initial_graph[j].append(k)

        # compute the sum of weights for each vertex
        all_keys = initial_graph.keys()
        weight_sum = {}
        for cur_key in all_keys:
            if cur_key == -1:
                continue
            cur_weights = initial_graph[cur_key]
            for weight in cur_weights:
                if cur_key not in weight_sum.keys():
                    weight_sum[cur_key] = weight[1]
                else:
                    weight_sum[cur_key] += weight[1]

        # select vertices as per the sum of weights
        selected_images = []
        for vertex in sorted(weight_sum, key=weight_sum.get, reverse=True):
            selected_images.append(vertex)

        # Post-process. Add extra results if there is no enough candidate.
        if len(selected_images) < retri_amount:
            voc_candidate = graph_list[1] # 0: hsv or 1000d. 1: voc
            voc_candidate = voc_candidate[-1]
            for i in voc_candidate:
                if i not in selected_images:
                    selected_images.append(i)

        return selected_images[0:retri_amount]

                
#########################################################
    
if __name__ == "__main__":

    fn_graph_list = 'data/corel-1k_graph_list.txt'
    fd_stdin = open(fn_graph_list)

    fn_fusion_result = 'data/corel-1k_graph_fusion_results.txt'
    fd_stdin_fusion = open(fn_fusion_result, 'w')
    fn_label = 'data/corel-1k_list_images_labels.txt'

    graphfusion = GraphFusion()

    num_ranks = 2 # in this case, just 2 types of ranks, i.e., voc and hsv
    retri_amount = 25

    graph_list = []
    count = 1
    
    for line in fd_stdin:
        line = line.rstrip()
        line = line.split()
        fn_graph = line[0]

        if numpy.mod(count, num_ranks) != 0:
            graph_list.append(cPickle.load(open(fn_graph, 'rb')))
            count += 1
            continue
        else:
            graph_list.append(cPickle.load(open(fn_graph, 'rb')))
            count += 1
        graph_list_copy = copy.deepcopy(graph_list)
        selected_images = graphfusion.Fusion_Density_Subgraph(graph_list_copy, num_ranks, retri_amount)
        # Uncomment the next line to use PageRank-based method, but keep the above line as well. We need "selected_images[0]" to define the center of the graph
        #selected_images = graphfusion.Fusion_Graph_Laplacian(graph_list, num_ranks, retri_amount, selected_images[0]) 

        fd_stdin_fusion.write(line[0] + ' ')
        for img_id in selected_images:
            fd_stdin_fusion.write(str(img_id) + ' ')

        fd_stdin_fusion.write('\n')
        graph_list = [] 

    fd_stdin_fusion.close()
        
    import evaluate
    print "After fusing VOC and HSV graphs:"
    evaluate.Evaluate(fn_label, fn_fusion_result, retri_amount-2)
