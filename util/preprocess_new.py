# preprocess data

def get_most_common_features(target, all_features, max = 3, min = 3):
    res = []
    main_keys = target.split('_')

    for feature in all_features:
        if target == feature:
            continue

        f_keys = feature.split('_')
        common_key_num = len(list(set(f_keys) & set(main_keys)))

        if common_key_num >= min and common_key_num <= max:
            res.append(feature)

    return res

def build_net(target, all_features):
    # get edge_indexes, and index_feature_map
    main_keys = target.split('_')
    edge_indexes = [
        [],
        []
    ]
    index_feature_map = [target]

    # find closest features(nodes):
    parent_list = [target]
    graph_map = {}
    depth = 2
    
    for i in range(depth):        
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)

            if feature not in graph_map:
                graph_map[feature] = []
            
            # exclude parent
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)

            graph_map[feature] = pure_children

            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)

                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)

        parent_list = pure_children

    return edge_indexes, index_feature_map

def construct_net_data(data, feature_map, labels=0):# characteristics of the network domain
    res_src = []
    res_dst = []
    res_size = []
    time = []
    map = {}
    with open('./data/swat/n_list.txt', 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')      
            ann = ann.split("-")
            map[ann[0].lstrip("HMI_")]=[ann[1]]  
    for feature in  feature_map:
        now_time = data.iloc[1].loc['datetime']
        src_list = []
        sor_list = []
        size_list = []
        if feature in map:
            dst_ip = map[feature]
        else:
            dst_ip = ''
        for index, row in data.iterrows():
            if row["datetime"] != now_time:
                dst_num =  data[(data['datetime'] == str(now_time)) & (data['dst'] == str(dst_ip).rstrip("']").lstrip("['"))]
                size = data[(data['datetime'] == str(now_time)) & (data['SCADA_Tag'] == "HMI_"+feature)]
                if dst_num.empty:
                    sor_list.append(0)
                else:
                    sor_list.append(dst_num.shape[0])
                if size.empty:
                    size_list.append(0)
                    src_list.append(0)
                else:
                    num = 0
                    for i ,r in size.iterrows():
                        num = num + len(str(size.loc[i]['Modbus_Value']))
                    num = num / size.shape[0]
                    size_list.append(num)
                    src_list.append(size.shape[0])
                now_time=row["datetime"]
                if feature == feature_map[0]:
                    time.append(now_time)
        res_src.append(src_list)   
        res_dst.append(sor_list)
        res_size.append(size_list)    
    # append labels as last
    sample_n = len(res_size[0])

    if type(labels) == int:
        res_src.append([labels]*sample_n)
        res_dst.append([labels]*sample_n)
        res_size.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res_size.append(labels)
        res_src.append(labels)
        res_dst.append(labels)
    return res_dst, res_size, res_src, time


def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    sample_n = len(res[0])

    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res

def build_loc_net(struc, all_features, feature_map=[]): 

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')

            c_index = index_feature_map.index(child)
            edge_indexes[0].append(p_index)
            edge_indexes[1].append(c_index)
        
    return edge_indexes