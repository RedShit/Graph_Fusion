function  [] = find_reciprocal_neighbors( img_name, result_idx, result_length,... 
                                         f_result_reranking, f_folder_graph,...
                                         search_region, kNN, retri_amount,...
                                         feature_type )
%Summary: Find reciprocal neighbors
%   Detail:
fp = fopen(f_result_reranking, 'w');
fprintf('\nFind reciprocal neighbors\n');

% build a reciprocal neighbor graph for each image

result_idx = result_idx + 1;

for i = 1 : result_length
    fprintf('%d\n', i);
    true_label = result_idx(i, 1);
    % img name from 0
    % result_graph = zeros(result_length + 1, result_length + 1);
    
    result_graph = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    
    % 1st layer: choose reciprocal neighbors only
    % qualified_list = [true_label];
    qualified_list = zeros(1, retri_amount * 2);
    qualified_list(1) = true_label;
    qualified_list_len = 1;
    
    % result_graph(true_label, true_label) = 1;
    result_graph(true_label) = [true_label, 1.0]';
    for j = 1 : search_region
        cur_id = result_idx(i, j+1);
        cur_id_kNN = result_idx(cur_id, 1:kNN);
        if any(cur_id_kNN == result_idx(i, 1)) == 1
            qualified_list_len = qualified_list_len + 1;
            qualified_list(qualified_list_len) = cur_id;
            %qualified_list = [qualified_list cur_id];
            
            result_graph(result_idx(i,1)) = [result_graph(result_idx(i,1)), [cur_id, 1.0]'];
            result_graph(cur_id) = [result_idx(i,1), 1.0]';
            
            % result_graph(result_idx(i,1), cur_id) = 1;
            % result_graph(cur_id, result_idx(i,1)) = 1;
        end
    end
    
    % 2nd layer: choose neighbors of reciprocal neighbors
    tmp = qualified_list_len;
    for j = 2 : tmp
        cur_id = qualified_list(j);
        cur_id_kNN = result_idx(cur_id, 1:kNN);
        common_set = intersect(cur_id_kNN, qualified_list);
        % diff_set = setdiff(cur_id_kNN, qualified_list);
        union_set = union(cur_id_kNN, qualified_list);
        % (rule 1: at least half of the chose set) or (rule 2: bring less unknown)
        weight = size(common_set, 2) / size(union_set, 2);
        % if (size(common_set, 2) - 1) >= (size(qualified_list, 2) - 1)/2.0
        % || size(diff_set, 2) <= (size(common_set, 2) - 1)
        if weight > 0.3
            for k = 2 : size(cur_id_kNN, 2)
                if any(cell2mat(keys(result_graph)) == cur_id_kNN(k)) == 0
                    result_graph(cur_id_kNN(k)) = [cur_id, weight]';
                else
                    result_graph(cur_id_kNN(k)) = [result_graph(cur_id_kNN(k)), [cur_id, weight]'];
                end
                if any(qualified_list == cur_id_kNN(k)) == 0
                    qualified_list_len = qualified_list_len + 1;
                    qualified_list(qualified_list_len) = cur_id_kNN(k);
                    % qualified_list = [qualified_list cur_id_kNN(k)];
                end
            end
        end
    end
    
    % add more images if the candidates are less thea the threshold. store
    % them in results_graph[-1]
    
    tmp = qualified_list_len;
    if  tmp <= retri_amount
        for j = 2 : retri_amount
            cur_id = result_idx(i,j);
            if any(qualified_list == cur_id) == 0
                qualified_list_len = qualified_list_len + 1;
                qualified_list(qualified_list_len) = cur_id;
                % qualified_list = [qualified_list cur_id];
                % result_graph(true_label, cur_id) = -1;
                
                if any(cell2mat(keys(result_graph)) == -1) == 0
                    result_graph(-1) = cur_id;
                else
                    result_graph(-1) = [result_graph(-1), cur_id];
                end
            end
        end
    end
    
    % save the rerank results, same format as the input
    fprintf(fp, '%s', img_name{i});
    for j = 1 : qualified_list_len
        fprintf(fp, ' %s', num2str(qualified_list(j) - 1));
    end
    fprintf(fp, '\n');
    
    % save graph files for each image
    % result_graph = sparse(result_graph);
    img_name_tmp = num2str(true_label);
    f_graph_name = strcat(f_folder_graph, img_name_tmp, feature_type);
    save(f_graph_name, 'result_graph');
end

% fclose(fp);
end

