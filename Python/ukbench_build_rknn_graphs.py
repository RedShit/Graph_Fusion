#######################################################################

# Description: Build a graph for the retrieval rank of each query image. 
# Reciprocal k-nearest neighbors and Jaccard similarity are used as rules to build these graphs. 
# Details in Section 3.2 of the following paper:
# Shaoting Zhang, Ming Yang, Timothee Cour, Kai Yu, Dimitris N. Metaxas: Query Specific Fusion for Image Retrieval. ECCV (2) 2012: 660-673
# (Also refer to the "Qin. et al., Hello neighbor: Accurate object retrieval with k-reciprocal nearest neighbors, CVPR'11" for reciprocal kNN)

# Author: Shaoting Zhang, shaoting@cs.rutgers.edu

# Date: March-2012

#######################################################################

import numpy
import cPickle
import re

############################## Load data ##############################

def load_data(fn_result):

    print "Load data"
    fd_stdin = open(fn_result)
    img_name = []
    result_idx = []
    for line in fd_stdin:
        line = line.rstrip()
        line = line.split()
        img_name.append(line[0])
        result_idx.append(map(int, line[1:]))

    print numpy.shape(img_name)
    #print numpy.shape(result_idx)
    result_length = len(result_idx)
    fd_stdin.close()
    
    return img_name, result_idx, result_length

############################## Reciprocal neighbors ###################

def find_reciprocal_neighbors(img_name, result_idx, result_length, fn_result_reranking, fn_folder_graph, search_region, kNN, retri_amount, feature_type):

    print "Find reciprocal neighbors"
    
    fd_stdin_result = open(fn_result_reranking, 'w')

    # build a reciprocal neighbor graph for each image
    for i in range(result_length): 
        result_graph = {}

        # 1st layer: choose reciprocal neighbors only
        qualified_list = []
        true_label = result_idx[i][0]
        qualified_list.append(true_label)
        result_graph[result_idx[i][0]] = [[result_idx[i][0], 1.0]]
        for j in range(search_region):
            cur_id = result_idx[i][j+1]
            cur_id_kNN = result_idx[cur_id][1:kNN]
            if result_idx[i][0] in cur_id_kNN:
                qualified_list.append(cur_id)
                
                (result_graph[result_idx[i][0]]).append([cur_id, 1.0])
                result_graph[cur_id] = [[result_idx[i][0], 1.0]]

        # 2nd layer: choose neighbors of reciprocal neighbors
        for j in range(1,len(qualified_list)):
            cur_id = qualified_list[j]
            cur_id_kNN = result_idx[cur_id][0:kNN]
            common_set = set(cur_id_kNN) & set(qualified_list)
            diff_set = set(cur_id_kNN) - set(qualified_list)
            union_set = set(cur_id_kNN) | set(qualified_list)
            # (rule 1: at least half of the close set) or (rule 2: bring less unknown)
            weight = float(len(common_set))/len(union_set)
            #if (len(common_set)-1) >= (len(qualified_list)-1)/2.0 or len(diff_set) <= (len(common_set)-1):
            if weight > 0.3: # works for 0.1-0.4
                for k in range(1,len(cur_id_kNN)):
                    if cur_id_kNN[k] not in result_graph.keys():
                        result_graph[cur_id_kNN[k]] = [[cur_id, weight]]
                    else:
                        result_graph[cur_id_kNN[k]].append([cur_id, weight])
                    if cur_id_kNN[k] not in qualified_list:
                        qualified_list.append(cur_id_kNN[k])

        # add more images if the candidates are less than the threshold. store them in results_graph[-1]
        if len(qualified_list) <= retri_amount:
            for j in range(1, retri_amount):
                cur_id = result_idx[i][j]
                if cur_id not in qualified_list:
                    qualified_list.append(cur_id)
                    if -1 not in result_graph.keys():
                        result_graph[-1] = [cur_id]
                    else:
                        result_graph[-1].append(cur_id)

        # save the reranking results, same format as the input
        fd_stdin_result.write(img_name[i] + ' ')
        for cur_id in qualified_list:
            fd_stdin_result.write(str(cur_id) + ' ')
        fd_stdin_result.write('\n')
        
        # save graph files for each image
        #img_name_tmp = (img_name[i].split('/'))[-1]
        img_name_tmp = str(true_label)
        fn_graph = fn_folder_graph + img_name_tmp + feature_type
        cPickle.dump(result_graph, open(fn_graph, 'wb'))
        

    fd_stdin_result.close()

if __name__=='__main__':

    #fn_result = 'data/ukbench_rank_voc.txt'
    fn_result = 'data/ukbench_rank_hsv3d.txt'
    #fn_result_reranking = 'data/ukbench_rerank_voc.txt'
    fn_result_reranking = 'data/ukbench_rerank_hsv3d.txt'
    fn_label = 'data/ukbench_list_images_labels.txt'
    fn_folder_graph = 'data/uk_bench_graphs/'
    
    # Parameter setting, 
    # smaller value for "tight graphs" such as ukbench (i.e., nearly-duplicate), 
    # larger for "loose graphs" such as corel (i.e., category classification)
    # okay to choose same values for search_range and kNN. very subtle difference
    search_region = 6
    kNN = 4
    retri_amount = 25    

    # Load data (retrieval results for all images)
    img_name, result_idx, result_length = load_data(fn_result)
    
    # Generate reciprocal graphs for all images
    if re.search('voc', fn_result):
        feature_type = '.voc'
    elif re.search('hsv', fn_result):
        feature_type = '.hsv'
    elif re.search('gist', fn_result):
        feature_type = '.gist'
    else:
        feature_type = '.unknown'

    find_reciprocal_neighbors(img_name, result_idx, result_length, fn_result_reranking, fn_folder_graph, search_region, kNN, retri_amount, feature_type)
    
    # Evaluate the accuracy	
    import evaluate
    print "Before building reciprocal kNN graphs:"
    evaluate.Evaluate(fn_label, fn_result, retri_amount-2)
    print "After building reciprocal kNN graphs:"
    evaluate.Evaluate(fn_label, fn_result_reranking, retri_amount-2)


