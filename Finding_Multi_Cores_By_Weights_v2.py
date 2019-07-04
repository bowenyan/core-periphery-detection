# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:05:23 2016

For the part of finding the average linkage hierarchical clusters, I modified the codes from "Programming collective intelligence", page 35-36.

@author: Bowen Yan

"""

import numpy as np
import networkx as nx
import pandas as pd
import os
import time
import sys

class struct_cluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None, cntofmembers=0):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance
        self.cntofmembers = cntofmembers
        
def change_edgelist_to_matrix(g, nodelist):
    narry = np.array(nx.to_numpy_matrix(g, nodelist=nodelist))
    return narry

def label_to_number(nodelist):
    i = 0 #all vertices start from 0
    idx_to_label = {}
    label_to_idx = {}
    for item in nodelist:
        idx_to_label[i] = item
        label_to_idx[item] = i
        i += 1
    return (idx_to_label, label_to_idx)
    
def average_weight(clu1, clu2, matrix):
    sum_w = 0.0
    avg = 0.0

    for i in range(len(clu1)):
        clu1_v = clu1[i]
        for j in range(len(clu2)):
            clu2_v = clu2[j]
            #print (clu1_v, clu2_v)
            sum_w= sum_w + matrix[clu1_v][clu2_v]    
    avg = sum_w / (len(clu1) * len(clu2))
    return avg 
    
def hcluster_average(matrix, idx_to_label, iteration, similarity=average_weight):    
    step = 1
    step_zscore = {}
    step_cores = {}
    step_periphery = {}
    
    ids = list(idx_to_label.keys())
    currentclusid = len(ids)
    similarities = {}  #store all distance values calculated during the clustering process
    
    #get the initial set - all nodes as the initial clusters
    cluster_set = [struct_cluster([ids[i]], id=i) for i in range(len(ids))]
    
    #initialise the cores and periphery pools
    cores_pool = []
    periphery_pool = list(idx_to_label.keys())
    linkage = []
    
    while len(cluster_set) > 1:
        #initial
        """
        find the cluters
        """
        child_pair = (0,1)
        closest_d = similarity(cluster_set[0].vec, cluster_set[1].vec, matrix)
        #print (closest_d)
        
        #print_cluster_set(cluster_set)
        #for the maximum average weight
        for i in range(len(cluster_set)):
            for j in range(i+1, len(cluster_set)):
                if (cluster_set[i].id, cluster_set[j].id) not in similarities:
                    similarities[(cluster_set[i].id, cluster_set[j].id)] = similarity(cluster_set[i].vec, cluster_set[j].vec, matrix)
                d = similarities[(cluster_set[i].id, cluster_set[j].id)]
                
                if (d > closest_d):
                    closest_d = d
                    child_pair=(i, j)
                    
        merged_clu = merge_clusters(child_pair, cluster_set)
        linkage.append(create_linkage(child_pair, cluster_set, closest_d, len(merged_clu)))
        newcluster = struct_cluster(merged_clu, left=cluster_set[child_pair[0]], 
                                    right=cluster_set[child_pair[1]],distance=closest_d, id=currentclusid, cntofmembers=len(merged_clu))
                                    
        currentclusid +=1
        del cluster_set[child_pair[1]]
        del cluster_set[child_pair[0]]
        cluster_set.append(newcluster)
        
        #print the clustering process at each step
        #print (currentclusid, closest_d)        
        #print_cluster_set(cluster_set, idx_to_label)
        
        """
        the statistical analysis: null models
        """
        
        partition_results = get_newpartition(merged_clu, cores_pool, periphery_pool)
        cores_pool = partition_results[0]
        periphery_pool = partition_results[1]
        step_cores[step] = cores_pool.copy()
        step_periphery[step] = periphery_pool.copy()  #without copy(), just the point
        #print (step, ':', cores_pool)
        #print (step, ':', periphery_pool)
        #print (step, '\t', step_cores[step])
        #print (step, '\t', step_periphery[step])
        
        if (len(periphery_pool)!=0):        
            zscore = compare_to_null_models(cores_pool, periphery_pool, matrix, iteration, ids)
            step_zscore[step] = zscore
            #print (step, ':', zscore)
        
        #print_partition(cores_pool, periphery_pool, idx_to_label)
        
        
        step +=1
        
    #print (step_periphery)  
    #print (step_zscore)
    
    linkage_df = pd.DataFrame(linkage)

    return {'cores':step_cores, 'periphery':step_periphery, 'zscore':step_zscore, 'linkage': linkage_df}

#compare to the null models
def compare_to_null_models(cores_pool, periphery_pool, matrix, iteration, ids):
    new_matrix = sort_matrix_by_partition(cores_pool, periphery_pool, matrix)
    #print (new_matrix)
    
    #get the density of partition in observed matrix
    densities_obs = calculate_density_partition(cores_pool, periphery_pool, new_matrix)
    
    #compare to null models
    zscore = run_null_models(densities_obs, cores_pool, periphery_pool, matrix, iteration, ids)
    return zscore
    
#generate random networks, and get zscore
def run_null_models(_densities_obs, cores_pool, periphery_pool, _matrix, iteration, ids):
    #calculate the ratio for the observed data
    d_cc_obs = _densities_obs['cc']
    d_cp_pp_obs = _densities_obs['cp_pp']
    if (d_cp_pp_obs!=0):
        d_ratio_cc_cp_pp_obs = d_cc_obs / d_cp_pp_obs
    else: 
        d_ratio_cc_cp_pp_obs = 0.0
        
    #random networks
    i = 0
    #step_density_cc_random = []
    #step_density_pp_random = []
    #step_density_cp_pp_random = []
    step_d_ratio_cc_cp_pp_random = []
    while (i < iteration):
        result_random = {}
        new_matrix = shuffle_matrix(_matrix, ids)
        result_random = calculate_density_partition(cores_pool, periphery_pool, new_matrix)
        d_cc_random = result_random['cc']
        d_cp_pp_random = result_random['cp_pp']
        
        if (d_cp_pp_random!=0):
            d_ratio_cc_cp_pp_random = d_cc_random / d_cp_pp_random
        else:
            d_ratio_cc_cp_pp_random = 0.0
        step_d_ratio_cc_cp_pp_random.append(d_ratio_cc_cp_pp_random)
        
        i +=1
    #print (step_d_ratio_cc_cp_pp_random)
    if (np.std(step_d_ratio_cc_cp_pp_random)!=0):
        ratio_cc_cp_pp_zscore = (d_ratio_cc_cp_pp_obs - np.mean(step_d_ratio_cc_cp_pp_random)) / np.std(step_d_ratio_cc_cp_pp_random)
    else:
        ratio_cc_cp_pp_zscore = 0.0
    return ratio_cc_cp_pp_zscore
    
#randomize the indices of matrix to keep the same weighted degree distribution
def shuffle_matrix(_matrix, ids):
    new_order_list = np.random.permutation(ids)
    #print (new_order_list)
    new_shuffle_matrix = _matrix[:, new_order_list][new_order_list]
    return new_shuffle_matrix

#calculate the density of different parts in the partition
def calculate_density_partition(cores_pool, periphery_pool, new_matrix):
    #print ('--')
    #print (new_matrix)
    #print ('--')
    start = 0
    pp_size = len(periphery_pool)
    
    #calculate the density of cores
    sum_matrix_cc = 0.0
    sum_base_cc = 0
    sum_core_nodes = 0
    for core in cores_pool:
        N = len(core)
        sum_core_nodes = sum_core_nodes + N
        sum_base_cc = (N-1) * N / 2 + sum_base_cc
        end = start + N
        matrix_cc = new_matrix[start:end, start:end]
        sum_matrix_cc = sum_matrix_cc + (matrix_cc.sum() / 2)
        start = end
    density_cc = sum_matrix_cc / sum_base_cc
    
    #calculate the density of the part between cores and periphery
    matrix_cp = new_matrix[end:,0:end]
    sum_matrix_cp = matrix_cp.sum()
    sum_base_cp = sum_core_nodes * pp_size
    if (sum_base_cp > 0):
        density_cp = sum_matrix_cp / sum_base_cp
    else:
        density_cp = 0.0
        
    #calculate the density of peripheral vertices
    matrix_pp = new_matrix[end:, end:]
    sum_matrix_pp = matrix_pp.sum() / 2
    if (pp_size>1):
        sum_base_pp = (pp_size-1) * pp_size / 2
        density_pp = sum_matrix_pp / sum_base_pp
    else:
        density_pp = 0.0
        
    #calculate the density of core_periphery together
    sum_matrix_cp_pp = sum_matrix_cp + sum_matrix_pp
    if (pp_size>0):
        sum_base_cp_pp = sum_base_cp + (pp_size-1) * pp_size / 2
        density_cp_pp = sum_matrix_cp_pp / sum_base_cp_pp
    else:
        density_cp_pp = 0.0
    
    #print (density_cc, density_pp, density_cp_pp)
    return {'cc': density_cc, 'pp': density_pp, 'cp_pp':density_cp_pp}
    
#sort the matrix according to the sequence of cores and periphery
def sort_matrix_by_partition(cc, pp, ori_matrix):
    new_order_list = []
    for core in cc:
        for v in range(len(core)):
            new_order_list.append(core[v])
        
    for v in range(len(pp)):
        new_order_list.append(pp[v])
    new_step_matrix = ori_matrix[:,new_order_list][new_order_list]
    return new_step_matrix

#merge the clusters
def merge_clusters(pair, cluster_set):
    merged_clu = []
    for i in range(len(pair)):
        items = cluster_set[pair[i]].vec
        for item in items:
            merged_clu.append(item)
    #print ('merged clu', merged_clu)
    return merged_clu
    
#create a linkage for the dendrogram
def create_linkage(_child_pair, cluster_set, _distance, _cntOfmembers):
    dict1 = {}
    dict1.update({'Cluster1':cluster_set[_child_pair[0]].id, 'Cluster2':cluster_set[_child_pair[1]].id, 'Distance':_distance, 'CntOfClusters':_cntOfmembers})
    #row = [, , _distance, _cntOfmembers]
    return dict1
    
#update the partition at each step of the clustering process
def get_newpartition(merged_clu, cores_pool, periphery_pool):   
    #print ('merged:', merged_clu)
    #print ('cores:', cores_pool)
    #print ('periphery:',periphery_pool)
    #print ('merged_cnt',len(merged_clu))
    if (len(merged_clu)>2):
        #print ('merged_clu', merged_clu)
        #print ('intersection',len(set(merged_clu).intersection(set(periphery_pool))))
        #add a vertex to the existing core OR merge two cores
        if (len(set(merged_clu).intersection(set(periphery_pool)))!=0):            
            #add vertices into the existing core
            for core in cores_pool:
                #print ('core:',core)
                if (len(set(core).intersection(set(merged_clu)))>1):
                    #print ('chosen core:', core)
                    cores_pool.remove(core)
                    cores_pool.append(merged_clu)
                    v_in_periphery = set(merged_clu).difference(core)
                    #print (v_in_periphery)
                    for v in v_in_periphery:
                        #print (v)
                        periphery_pool.remove(v)
                    break
        else:
            #merge two cores
            #for core in cores_pool:
            #    print (core)
            #    if (len(set(core).intersection(set(merged_clu)))>1):
            #        print ('chosen pair:', core)
            #        cores_pool.remove(core)
                    
            i = 0
            n = len(cores_pool)
            while i < n:
                core = cores_pool[i]
                #print (core)
                if (len(set(core).intersection(set(merged_clu)))>1):
                    #print ('chosen pair:', core)
                    del cores_pool[i]
                    n = n - 1
                else:
                    i = i + 1
            cores_pool.append(merged_clu)
    else:
        #add a core (pair of vertices) into cores_pool
        cores_pool.append(merged_clu)
        for v in merged_clu:
            periphery_pool.remove(v)
    return (cores_pool, periphery_pool)
    
#print the info of cluster_set
def print_cluster_set(cluster_set, idx_to_label):
    for i in range(len(cluster_set)):
        tmp = []
        for v in cluster_set[i].vec:
            tmp.append(idx_to_label[v])
        print (tmp)

def print_partition(cores_pool, periphery_cool, idx_to_label):
    #print ('print_partition:',cores_pool)
    cores_labels = []
    for core in cores_pool:
        tmp = []
        #print (core)
        for v in range(len(core)):
            tmp.append(idx_to_label[core[v]])
        cores_labels.append(tmp)
    print ('cores:', cores_labels)
    
    peripheral_labels = []
    for i in range(len(periphery_cool)):
        peripheral_labels.append(idx_to_label[periphery_cool[i]])
    print ('periphery:', peripheral_labels)
     
#create a file to store zscore, cores, periphery
def collect_results_to_dataframe(_partitions, idx_to_label):
    rows = []
    _step_cores = _partitions['cores']
    _step_periphery = _partitions['periphery']
    _step_zscore = _partitions['zscore']
    
    for step in range(1, len(_step_cores)):
        #print (step)
        _cores_pool = _step_cores[step]
        _periphery_pool = _step_periphery[step]

        cores_labels = []
        for core in _cores_pool:
            tmp = []
            for v in range(len(core)):
                tmp.append(idx_to_label[core[v]])
            cores_labels.append(tmp)
            
        peripheral_labels = []
        for i in range(len(_periphery_pool)):
            peripheral_labels.append(idx_to_label[_periphery_pool[i]])
            
        zscore = 0.0
        if (len(_step_zscore)>step):
            zscore = _step_zscore[step]

        dict1 = {}
        dict1.update({'step':step, 'zscore':zscore, 'cores':cores_labels, 'periphery': peripheral_labels})
        rows.append(dict1)
    #print (rows)
    return rows

def save_dict_to_file(dict_map, fn):
    f = open(fn,'w')
    f.write(str(dict_map))
    f.close()
    
if __name__ == '__main__':
    #iteration_random = 1000000
    #iteration_random = 10000
    
    #network_fn = './Core_Periphery/Network_Data/network_RTA_inventor_multiIPCs_3digitlevel_1976_2006.txt'
    #network_fn = './Core_Periphery/Network_Data/weighted_karate.txt'
    #network_fn = './Core_Periphery/Network_Data/train_bombing.txt'
    #network_fn = './Core_Periphery/Network_Data/test_tool_map.txt'
    
    network_fn = sys.argv[1]
    iteration_random = int(sys.argv[2])
    network_idx = os.path.splitext(os.path.basename(network_fn))[0]
    
    g = nx.read_weighted_edgelist(network_fn)
    
    #sort the nodelist
    nodelist = sorted(g.nodes())
    #print (nodelist)
    #get the matrix    
    matrix = change_edgelist_to_matrix(g, nodelist)
    #print (matrix)
    
    #get the node_idx, and idx_node list
    initial_results = label_to_number(nodelist)
    idx_to_label = initial_results[0]
    label_to_idx = initial_results[1]
    
    idx_to_label_fn = network_idx + '_idx_to_label.txt' 
    save_dict_to_file(idx_to_label, idx_to_label_fn)
    
    

    #print (idx_to_label)
    method = 'average'
    start_time = time.time()
    if (method == 'average'):
        _partitions = hcluster_average(matrix, idx_to_label, iteration_random)
    print ('--- %0.00f seconds ---' %(time.time()-start_time))
    
    partition_results_df = pd.DataFrame(collect_results_to_dataframe(_partitions, idx_to_label))
    df_results_fn = network_idx + '_' + str(method) + '_' + str(iteration_random) + '.csv'
    #print (df_results_fn)    
    partition_results_df.to_csv(df_results_fn)
    
    linkage_df = _partitions['linkage']
    df_linkage_fn = network_idx + '_' + str(method) + '_' + str(iteration_random) + '_linkage.csv'
    linkage_df.to_csv(df_linkage_fn, index=False)
    
    
    