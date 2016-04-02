%% Loading files 
%
%  load data produced by hsv3d or voc method
%  data format:
%       img_name rank1 rank2 ...

fprintf('\nLoading Data...\n');

f_result = '../data/ukbench_rank_hsv3d.txt';
feature_type = '_hsv';
% f_result = '../data/ukbench_rank_voc.txt';
% feature_type = '_voc';

rank = importdata(f_result);

fprintf('\nThere are %d images\n',size(rank.rowheaders,1));

img_name = rank.rowheaders;
result_idx = int32(rank.data);
result_length = size(rank.rowheaders,1);

%% Build graph 
%
%

f_result_reranking = '../data/ukbench_rerank_hsv3d_mat.txt';
% f_result_reranking = '../data/ukbench_rerank_voc_mat.txt';
f_label = '../data/ukbench_list_images_labels.txt';
f_folder_graph = '../data/uk_bench_graphs_mat/';

search_region = 6;
kNN = 4;
retri_amount = 25;

find_reciprocal_neighbors(img_name, result_idx, result_length,... 
                          f_result_reranking, f_folder_graph,...
                          search_region, kNN, retri_amount, feature_type);
                      
fprintf('\nBefore building reciprocal kNN graphs\n');
% evaluate(f_label, f_result, retri_amount - 2);
fprintf('\nAfter building reciprocal kNN graphs\n');
% evaluate(f_label, f_result_reranking, retri_amount - 2);