f#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
import kmer_counter
import textwrap
import time
import argparse
import collections
import queue
from skopt import gp_minimize
import math
from argparse import RawTextHelpFormatter
import cuml
from numba import cuda 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
from collections import defaultdict


def enable_dyn_memory_growth_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def display_detected_gpus():
    # print("CUDA and TensorFlow Configuration:")
    # print("GenomeFace will utilize the Device(s) listed below:\n")
  
    cuda_gpus =  cuda.gpus
    if cuda_gpus:
        print("CUDA Devices (Clustering Acceleration):")
        for idx, gpu in enumerate(cuda_gpus):
            print("  - Device " + str(idx) + ": " + gpu.name.decode('utf-8'))
    else:
        print(bcolors.ERR+"    No CUDA Device detected" + bcolors.ENDC)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("")
    if gpus:
        print("Tensorflow Enabled Devices (Neural Network Acceleration):")
        for gpu in gpus:
            print( "  - " + gpu.name)
    else:
        print(bcolors.ERR+"  No GPUs detected by TensorFlow."+ bcolors.ENDC)
    print("")
class CustomHelpFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        lines = text.splitlines()
        formatted_lines = [textwrap.fill(line, width) for line in lines]
        return '\n'.join(formatted_lines)
# Custom Arg parser class for argument parsing to display GPUs in tensorflow
class CustomHelpParser(argparse.ArgumentParser):
    def print_help(self, *args, **kwargs):
        super(CustomHelpParser, self).print_help(*args, **kwargs)
        print()
        display_detected_gpus()
        print("Notes:")
        print("  - By default, coassembly is assumed. For concatenated single-sample assemblies, use the [-s] flag.\n")
        print("Examples:")
        print("  genomeface -i coassembly.fa.gz -a abundance.tsv -g markers.tsv -o ./output")
        print("  genomeface -i concatenated_single_sample_assemblies.fa.gz -a abundance.tsv -g markers.tsv -o ./output -s -m 1000\n")
hyperparam_range = [(0.0, 2.0)]

def calc_alpha(num_samples):
    c=0.3304441433347871
    d=4.853660700796253
    pre_alpha =  1/(c+d/num_samples)
    alpha = pre_alpha/(pre_alpha+0.8)
    return 2*alpha

# tool to supress Debug output From MST builder by redirecting stdout to /dev/null
def hdbscan_fit_gf_wrapper(cluster_instance,a,b,c):
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    os.dup2(devnull_fd, 1)

    cluster_instance.fit_GF(a,b,c)

    os.dup2(old_stdout_fd, 1)
    os.close(devnull_fd)
    os.close(old_stdout_fd)


from tensorflow.keras import  layers
from tensorflow import keras
import tensorflow as tf
# import tensorflow_addons as tfa

import numpy as np
import pandas as pd
import gc
import sys
import pandas as pd
from tqdm import tqdm



class EstimateTable():
    def __init__(self):
        self.table = np.zeros((8,8))
    def add(self, purity, completeness):
        if purity < .30 or completeness < .30:
            return
        idx_purity = 0
        idx_compl = 0
        if purity >= .95:
            idx_purity = -1
        else:
            idx_purity = (purity-.30) // 0.1
        if completeness >= .95:
            idx_compl = -1
        else:
            idx_compl = (completeness-.30) // 0.1
        self.table[int(idx_purity),int(idx_compl)] += 1

    def finalize(self):
        for i in range(7, 0, -1):
            self.table[i-1,:] += self.table[i,:]
        for i in range(7, 0, -1):
            self.table[:,i-1] += self.table[:,i]
    def display(self):
        tbl = pd.DataFrame(self.table, columns = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
        tbl.columns.name = "Recall"
        tbl.index = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        tbl.index.name = "Purity"
        print(f"{bcolors.BOLD}Outputted Bin counts by Purity / Recall (Estimated):{bcolors.ENDC}")
        print(tbl.to_string())


def calc_depths(depth_input,num_columns,numcontigs, min_contig_length):
    num_samples =  (num_columns - 3) // 2
    arr = np.zeros((numcontigs,num_samples),dtype='float') 
    used = 0
    names = []

    if depth_input[-3:] == "npz":
        return np.load(depth_input)['arr_0']
#    we only need every other (the ,mean depths,  starting at 4th column / index 3)
    with pd.read_csv(depth_input,sep='\t',lineterminator='\n',engine='c',chunksize = 10 ** 6) as reader:
        for chunk in tqdm(reader):
            ray = chunk[chunk['contigLen'] >= min_contig_length].iloc[:,range(3,num_columns,2)].to_numpy()
            arr[used:used+len(ray)] = ray
            used += len(ray)
            # names.extend(chunk[chunk['contigLen'] >= 1500].iloc[:,0])

            if used == numcontigs:
                break #FIXME 
            #df = pd.concat([df, chunk[chunk['contigLen'] >= 1500].iloc[:,range(3,num_columns,2)]], ignore_index=True)
    row_sums = arr.sum()
    return arr#*.1#row_sums/(num_samples*numcontigs))

class DisJointSets():
    def __init__(self,N):
        # Initially, all elements are single element subsets
        self._parents = [node for node in range(N)]
        self._ranks = [1 for _ in range(N)]

    def find(self, u):
        while u != self._parents[u]:
            # path compression technique
            self._parents[u] = self._parents[self._parents[u]]
            u = self._parents[u]
        return u

    def connected(self, u, v):
        return self.find(u) == self.find(v)

    def union(self, u, v):
        # Union by rank optimization
        root_u, root_v = self.find(u), self.find(v)
        if root_u == root_v:
            return True
        if self._ranks[root_u] > self._ranks[root_v]:
            self._parents[root_v] = root_u
        elif self._ranks[root_v] > self._ranks[root_u]:
            self._parents[root_u] = root_v
        else:
            self._parents[root_u] = root_v
            self._ranks[root_v] += 1
        return False


import uuid
class TreeNode():
    def __init__(self, markers, distance, index = -1):
        self.distance = distance
        self.birth_dist = distance
        self.markers = markers
        self.parent = None
        self.children = []
        self.index =index
        self.id = index
        self.id = uuid.uuid1()
        self.T =  1/self.distance if self.distance != 0 else 0
        self.cluster_size = 1
        self.is_merger = False
        self.stability = 0
        self.delta = False
    def calc_f1(self, beta,min_purity):
        tp = sum(1 for i  in self.markers if i != 0)
        fn = len(self.markers) - tp
        fp = sum(i -1 for i in self.markers if i > 1)
        b2 = beta*beta
        precision = tp/(fp+tp) if fp+tp > 0 else  1
        return (1+b2)*tp/((1+b2)*tp + b2*fn + fp)*(1 if precision >= min_purity else 0)
    def purity(self):
        tp = sum(1 for i  in self.markers if i != 0)
        fp = sum(i -1 for i in self.markers if i > 1)
        precision = tp/(fp+tp) if fp+tp > 0 else  1
        return precision
    def completeness(self):
        return  sum(1 for i  in self.markers if i != 0)/ len(self.markers)
    

    def calc_stability(self,beta,alpha, min_purity):
        bonus = self.T - self.cluster_size/(self.parent.distance) if self.parent is not None  and self.birth_dist != 0  else 0
        return self.calc_f1(beta, min_purity)**4# + (bonus*alpha)#*(1 if self.birth_dist< 0.6 else 0) #sum(self.markers)

    def merge(self, other, distance):
            merged = TreeNode([self.markers[i] + other.markers[i] for i in range(len(self.markers))],distance) if distance >= 0 else TreeNode([min(self.markers[i] + other.markers[i],1) for i in range(len(self.markers))],distance) 
            merged.children = [self,other]
            self.parent = merged
            other.parent = merged
            merged.is_merger = merged.distance >= 0
            if self.is_merger and not other.is_merger:
                merged.birth_dist = self.birth_dist
            else:
                if not self.is_merger and other.is_merger:
                    merged.birth_dist = other.birth_dist
                else:
                    merged.birth_dist = merged.distance
            return merged
    def connect(self, forest):
            q =  queue.Queue() 
            q.put(self)
            elements = []
            while not q.empty():
                node = q.get()
                if node.index != -1:
                    elements.append(node.index)
                for child in node.children:
                    q.put(child)
            for index in elements[1:]:
                forest.union(forest.find(elements[0]), index)
            for f in elements[1:]:
                assert forest.find(elements[0]) == forest.find(f)
class Acumen4:
    """A class that represents the Acumen3 model."""
    
    def __init__(self, beta=1, alpha=0, min_purity=85):
        """Initialize with beta and alpha values."""
        self.beta = beta
        self.alpha = alpha
        self.min_purity = min_purity / 100.0

    @staticmethod
    def normalize_marker(marker_name, trans_dict):
        """Normalize marker names based on a translation dictionary."""
        return trans_dict.get(marker_name, marker_name)
        
    def fit_predict(self, weights, src, dst, fetch_mg_folder, ids, pairs, disable_report=False):
        """Fit and predict the model."""
        src, dst, weights = self.convert_to_numpy(src, dst, weights)
        done = [False] * len(ids)
        markers, forest = self.initialize_containers(len(ids))
        idx_hm = {i: c for c, i in enumerate(ids)}
        
        self.populate_markers(fetch_mg_folder, markers, idx_hm)
        
        leaf_nodes, current_nodes = self.initialize_nodes(markers)
        root = self.build_tree(weights, src, dst, forest, current_nodes)
        
        nodes = self.breadth_first_traversal(root)
        self.calculate_node_properties(nodes)
        
        quality_table, component_forest = self.evaluate_quality(root, len(ids))
        
        if not disable_report:
            quality_table.display()
        
        return [component_forest.find(i) for i in range(len(ids))], root.stability

    def convert_to_numpy(self, src, dst, weights):
        """Convert to numpy arrays."""
        dtype_map = {'src': 'int', 'dst': 'int', 'weights': 'float'}
        src = src.to_output(output_type='numpy', output_dtype=dtype_map['src'])
        dst = dst.to_output(output_type='numpy', output_dtype=dtype_map['dst'])
        weights = weights.to_output(output_type='numpy', output_dtype=dtype_map['weights'])
        return src, dst, weights

    def initialize_containers(self, length):
        """Initialize marker and forest containers."""
        markers = [[0] * 40 for _ in range(length)]
        forest = DisJointSets(length)
        _ = [forest.find(i) for i in range(length)]
        return markers, forest

    def populate_markers(self, fetch_mg_folder, markers, idx_hm):
        """Populate marker information."""
        mg_df = pd.read_csv(fetch_mg_folder, sep='\t', header=None)
        normalize_marker_trans_dict = {
            'TIGR00388': 'TIGR00389',
            'TIGR00471': 'TIGR00472',
            'TIGR00408': 'TIGR00409',
            'TIGR02386': 'TIGR02387',
        }
        mg_enum = {}
        current_idx = 0
        for index, row in mg_df.iterrows():
            marker_name = self.normalize_marker(row[1], normalize_marker_trans_dict)
            if marker_name == "COG0086" or row[0] not in idx_hm:
                continue
            contig_index = idx_hm[row[0]]
            if marker_name in mg_enum:
                marker_num = mg_enum[marker_name]
            else:
                marker_num = current_idx
                mg_enum[marker_name] = current_idx
                current_idx += 1
            markers[contig_index][marker_num] = 1

    def initialize_nodes(self, markers):
        """Initialize leaf and current nodes."""
        leaf_nodes = [TreeNode(markers[i], 0, i) for i in range(len(markers))]
        current_nodes = leaf_nodes.copy()
        return leaf_nodes, current_nodes

    def build_tree(self, weights, src, dst, forest, current_nodes):
        """Build the hierarchical tree."""
        root = None
        for i in range(len(src)):
            src_set, dst_set = forest.find(src[i]), forest.find(dst[i])
            if src_set != dst_set:
                forest.union(src_set, dst_set)
                merged = forest.find(src[i])
                current_nodes[merged] = current_nodes[src_set].merge(current_nodes[dst_set], weights[i])
                root = current_nodes[merged]
        return root

    def breadth_first_traversal(self, root):
        """Perform breadth-first traversal and return the list of nodes."""
        nodes = []
        q = queue.Queue()
        q.put(root)
        while not q.empty():
            n = q.get()
            nodes.append(n)
            for child in n.children:
                q.put(child)
        return nodes

    def calculate_node_properties(self, nodes):
        """Calculate properties for each node."""
        for i in range(len(nodes) - 1, -1, -1):
            node = nodes[i]
            node.T += sum(child.T for child in node.children)
            node.cluster_size = sum(child.cluster_size for child in node.children)
            self_stability = node.calc_stability(self.beta, self.alpha, self.min_purity)
            children_stability = sum(child.stability for child in node.children)
            node.stability = max(self_stability, children_stability)
            node.delta = self_stability >= children_stability or len(node.children) == 0

    def evaluate_quality(self, root, length):
        """Evaluate and return the quality of the built tree."""
        quality_table = EstimateTable()
        component_forest = DisJointSets(length)
        q = queue.Queue()
        q.put(root)
        while not q.empty():
            node = q.get()
            if node.delta:
                quality_table.add(node.purity(), node.completeness())
                node.connect(component_forest)
            else:
                for child in node.children:
                    q.put(child)
        quality_table.finalize()
        return quality_table, component_forest


class PIBlock2(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim,input_dim=None):
        super(PIBlock2, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="tanh"), layers.Dense(embed_dim),]
        )
        self.bn = layers.BatchNormalization()
        self.s = embed_dim
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = inputs + attn_output
        ffn_output = self.ffn(out1)
        return out1 + ffn_output
    def compute_output_shape(self, input_shape):
        return self.ffn[-1].output.shape



def assemble_model(n_samples, weight_file='best_val_loss3.m'):
    this_file_loc = os.path.abspath(__file__)
    conda_prefix = os.path.dirname(os.path.dirname(this_file_loc))

    depth_input = Input(shape=(n_samples, 1))
    kmer_inputs = [Input(shape=(v,)) for v in [512,136,32,10,2,528,256,136,64,36]]
    
    # Abundance model
    y = layers.TimeDistributed(layers.Dense(16, activation='linear'))(depth_input)
    for _ in range(4):
        y = PIBlock2(16, 16, 512)(y)
    
    y = layers.Flatten()(y)
    y = tf.math.l2_normalize(y, axis=1)
    
    model_adb_eval = Model([Input(shape=(136,)),*kmer_inputs, depth_input], y)
    model_adb_eval.compile()
    path_weight = os.path.join(conda_prefix,"share","genomeface","weights","model9_eval.m")
    model_adb_eval.load_weights(path_weight)#"/pscratch/sd/r/richardl/camisets2/model/model9_air_oral_eval.m")
    
    # Compositional model
    x = layers.Concatenate()(kmer_inputs)
    x = layers.BatchNormalization()(x)
    
    for units in [1024 * 4, 1024 * 8 * 2]:
        x = layers.Dense(units, activation='tanh', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(512, use_bias=False, activation=None)(x)
    x = layers.BatchNormalization()(x)
    
    x = tf.math.l2_normalize(x, axis=1)
    
    model3 = Model([Input(shape=(136,)),*kmer_inputs, depth_input], x)
    model3.compile()
    path_weight = os.path.join(conda_prefix,"share","genomeface","weights","general_t2eval.m")
    model3.load_weights(path_weight)#"/pscratch/sd/r/richardl/trainings/genearl/general_t2eval.m")
    
    return model_adb_eval, model3
    
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def msg(*args):
    left =  bcolors.OKGREEN+"[" + str(datetime.now()) + "] "+ bcolors.ENDC
    print(left,*args)

sampled_points = []
function_values = []
def opt_function_generator(y20_cat,y21_cat,contig_names,marker_tsv, hdbscan_instance,min_purity,progress_bar=None):
    def opt_function(alpha_opt):
            global sampled_points
            global function_values
            sampled_points.append(alpha_opt[0])
            # print("running opt_function with alpha=", alpha_opt[0])
            hdbscan_fit_gf_wrapper( hdbscan_instance, y20_cat[:150000],y21_cat[:150000], alpha_opt[0])
            acumen = Acumen4(alpha=0.0,beta=1,min_purity=min_purity)
            acumen_labels2, obj_value = acumen.fit_predict(hdbscan_instance.mst_weights_, hdbscan_instance.mst_src_, hdbscan_instance.mst_dst_, marker_tsv,contig_names, ("a","b"),disable_report=True)
            del acumen
            function_values.append(obj_value)
            # print("reward", obj_value)
            if progress_bar is not None:
                progress_bar.update(1)
            return -obj_value

    return opt_function


def main(input_file, abundance_file, markers_file, output_folder, min_contig_length, is_multi_assembly, min_purity, optimize_balance):
    msg(f"{bcolors.BOLD}{bcolors.HEADER}GenomeFace{bcolors.ENDC}")

    enable_dyn_memory_growth_gpus()

    #Load Fasta Data (inpts is a list of contig x DIM_k matrices, each reprsenting kmer frequences for each contig, for various k)
    contig_names, contigs_lens_filtered, inpts = load_fasta_data(input_file, min_contig_length)
    num_contigs_work = len(contigs_lens_filtered)

    # assembly_counts is a  hashmap: assembly name ->  contigs per assembly in if
    # we are using multiple assemblies; assemblies names determined by contig prefix
    assembly_counts = count_assemblies(contig_names, is_multi_assembly)


    #  number of samples (int), numpy matrix of mean depths (contig x samples sized) 
    num_samples, pre_depths = load_depth_data(abundance_file, num_contigs_work, min_contig_length)
    
    gc.collect()

    start = time.time()
    msg("Assembling Model and Loading weights")
    model,model2 = assemble_model(num_samples)
    msg("Model Assembly took", "{:.3f}".format(time.time() -start), "seconds")

    #convert model input from numpy to tensorflow dataset format
    model_data_in, dataset = prepare_model_input(contigs_lens_filtered, inpts, pre_depths, num_samples, min_contig_length)
    del model_data_in

    msg("Data conversion time ", time.time()-start)

    start =time.time()
    gc.collect()

    msg("Embedding Contigs")
    # del aaq

    y20_cat, y21_cat = embed_contigs(model, model2, dataset, len(contig_names), num_samples)
    del dataset

    gc.collect()

    msg("Embedding Contigs took", "{:.3f}".format(time.time() -start), "seconds")
    start = time.time()
    msg("Freeing memory")
    del model
    del model2
    gc.collect()
    for device in cuda.gpus:
        device.reset()

    msg("Done resetting memory; took", time.time() - start)
    start = time.time()

    
    out_folder =  output_folder
    separator = 'C'

    if is_multi_assembly:
        # in case of concatenated assemblies, use sample with largest number of contigs to optimize balance of abundance / compositon
        max_prefix = max(assembly_counts, key=assembly_counts.get)

        not_nan_filter = np.char.startswith(contig_names,max_prefix+separator) &( ~np.any(np.isnan(y21_cat[contigs_lens_filtered>=min_contig_length]), axis=1)) &( ~np.any(np.isnan(y20_cat[contigs_lens_filtered>=min_contig_length]), axis=1))
        from cuml  import HDBSCAN

        clusterer = HDBSCAN(min_cluster_size=1500000,min_samples=1,cluster_selection_method='eom',allow_single_cluster=True)

        opt_alpha = calc_alpha(num_samples)

        if optimize_balance:
            n_calls=20
            msg("Tuning Balance between Composition and Abundance with Bayesian Optimization.")
            progress_bar = tqdm(total=n_calls)
            func_to_opt = opt_function_generator(y20_cat[contigs_lens_filtered>=min_contig_length][not_nan_filter],y21_cat[contigs_lens_filtered>=min_contig_length][not_nan_filter], list(np.asarray(contig_names)[contigs_lens_filtered>=min_contig_length][not_nan_filter]), markers_file, clusterer, min_purity,progress_bar)
            opt_output = gp_minimize(
                func_to_opt,  # the function to minimize
                hyperparam_range,  # the bounds on each dimension of x
                acq_func="EI",  # the acquisition function
                n_calls=n_calls,  # the number of evaluations of f
                n_initial_points=3,  # the number of random initialization points
                # noise=10,  # the noise level (optional)
                random_state=42,  # the random seed
                x0=[calc_alpha(num_samples)]
            )
            progress_bar.close()
            opt_alpha = opt_output.x[0]
            msg("Guess", calc_alpha(num_samples),"Optimized Alpha", opt_alpha)

        labels_out0 ,names_out0 =  [] , []
        num_clusters = 0
        len_asm_counts = len(assembly_counts)
        for sample_number, sample_name in enumerate(assembly_counts):
            msg(f"Clustering Assembly with Prefix {sample_name} ({1+sample_number}/{len_asm_counts})") 
            not_nan_filter = np.char.startswith(contig_names,sample_name+separator) &( ~np.any(np.isnan(y21_cat[contigs_lens_filtered>=min_contig_length]), axis=1)) &( ~np.any(np.isnan(y20_cat[contigs_lens_filtered>=min_contig_length]), axis=1))
            hdbscan_fit_gf_wrapper(clusterer, y20_cat[contigs_lens_filtered>=min_contig_length][not_nan_filter], y21_cat[contigs_lens_filtered>=min_contig_length][not_nan_filter],opt_alpha)
            msg("Constructing Hierarchy and Selecting Optimal Clusters")
            acumen = Acumen4(alpha=0.00000000001,beta=1,min_purity=min_purity)
            start = time.time()

            acumen_labels2, _ = acumen.fit_predict(clusterer.mst_weights_, clusterer.mst_src_, clusterer.mst_dst_, markers_file ,list(np.asarray(contig_names)[contigs_lens_filtered>=min_contig_length][not_nan_filter]), ("a","b"))
            msg("Hierarchy Construction and Optimal Clusters Selection  took", "{:.3f}".format(time.time() -start), "seconds")

            unique_labels = defaultdict(lambda: len(unique_labels))
            normed_labels = [unique_labels[label] for label in acumen_labels2]

            labels_out0.extend(lbel + num_clusters for lbel in normed_labels)
            names_out0.extend(list(np.asarray(contig_names)[contigs_lens_filtered>=min_contig_length][not_nan_filter]))
            num_clusters += max(normed_labels) +1 if len(normed_labels) > 0 else 0
        start = time.time()
        bases_binned = kmer_counter.write_fasta_bins(names_out0, labels_out0,input_file,  output_folder)
        df = pd.DataFrame(list(zip(labels_out0 ,names_out0)))
        df.to_csv(os.path.join( output_folder,"bins.tsv"),sep='\t',header = False,index=False)
        msg("Writing contigs took", "{:.3f}".format(time.time() -start), "seconds")
    else:
        import math

        not_nan_filter =  ~(np.any(np.isnan(y21_cat[contigs_lens_filtered>=min_contig_length]) , axis=1) + np.any(np.isnan(y20_cat[contigs_lens_filtered>=min_contig_length]) , axis=1))
        from cuml  import HDBSCAN

        clusterer = HDBSCAN(min_cluster_size=1500000,min_samples=1,cluster_selection_method='eom',allow_single_cluster=True)
        
        opt_alpha = calc_alpha(num_samples)

        if optimize_balance:
            n_calls=20
            msg("Tuning Balance between Composition and Abundance with Bayesian Optimization.")
            progress_bar = tqdm(total=n_calls)
            func_to_opt = opt_function_generator(y20_cat[contigs_lens_filtered>=min_contig_length][not_nan_filter],y21_cat[contigs_lens_filtered>=min_contig_length][not_nan_filter], list(np.asarray(contig_names)[contigs_lens_filtered>=min_contig_length][not_nan_filter]), markers_file, clusterer,min_purity, progress_bar)
            opt_output = gp_minimize(
                func_to_opt,  # the function to minimize
                hyperparam_range,  # the bounds on each dimension of x
                acq_func="EI",  # the acquisition function
                n_calls=n_calls,  # the number of evaluations of f
                n_initial_points=3,  # the number of random initialization points
                # noise=10,  # the noise level (optional)
                random_state=42,  # the random seed
                x0=[calc_alpha(num_samples)]
            )
            progress_bar.close()
            del progress_bar
            opt_alpha = opt_output.x[0]
            msg("Guess", calc_alpha(num_samples),"Optimized Alpha", opt_alpha)


        # for device in cuda.gpus:
        #     #device = cuda.select_device(i)
        #     device.reset()
            # print(cuda.peek_at_error())
        del clusterer



        clusterer = HDBSCAN(min_cluster_size=1500000,min_samples=1,cluster_selection_method='eom',allow_single_cluster=True)

        hdbscan_fit_gf_wrapper(clusterer, y20_cat[contigs_lens_filtered>=min_contig_length][not_nan_filter],y21_cat[contigs_lens_filtered>=min_contig_length][not_nan_filter], opt_alpha)


        msg("cluster time: ",time.time() -start)

        names_out = list(np.asarray(contig_names)[contigs_lens_filtered>=min_contig_length][not_nan_filter])

        del y20_cat
        del y21_cat
        gc.collect()

        out_folder =  output_folder
        start = time.time()

        msg("Constructing Hierarchy and Selecting Optimal Clusters")

        acumen = Acumen4(alpha=0.00000000001,beta=1,min_purity=min_purity)
        start = time.time()
        acumen_labels2, _ = acumen.fit_predict(clusterer.mst_weights_, clusterer.mst_src_, clusterer.mst_dst_, markers_file,list(np.asarray(contig_names)[contigs_lens_filtered>=min_contig_length][not_nan_filter]), ("a","b"))
        msg("Hierarchy Construction and Optimal Clusters Selection  took", "{:.3f}".format(time.time() -start), "seconds")
        start = time.time()
        bases_binned = kmer_counter.write_fasta_bins(names_out, list(acumen_labels2),input_file,  out_folder)
        
        df = pd.DataFrame(list(zip(acumen_labels2,names_out)))
        df.to_csv(os.path.join(out_folder,"bins.tsv"),sep='\t',header = False,index=False)

        msg("Writing contigs took", "{:.3f}".format(time.time() -start), "seconds")


def load_fasta_data(input_file, min_contig_length):
    msg("Loading fasta: " + input_file)
    start = time.time()
    aaq = kmer_counter.find_nMer_distributions(input_file, min_contig_length)
    inpts = [np.reshape(aaq[i], (-1, size)) for i, size in enumerate([512, 136, 32, 10, 2, 528, 256, 136, 64, 36], start=1)]
    
    contig_names = np.asarray(aaq[-1])
    contig_lens = np.asarray(aaq[0])
    contig_lens_filtered = contig_lens[contig_lens >= min_contig_length]
    contig_names = contig_names[contig_lens_filtered >= min_contig_length]
    
    msg(f"Loaded {len(contig_names)} contigs in {time.time() - start:.3f} seconds")
    return contig_names, contig_lens_filtered, inpts

def count_assemblies(contig_names, is_multi_assembly):
    if not is_multi_assembly:
        return
    separator = 'C'
    assembly_counts = defaultdict(int)
    for name in contig_names:
        prefix = name.split(separator, 1)[0]
        assembly_counts[prefix] += 1
    return assembly_counts

def load_depth_data(abundance_file, num_contigs, min_contig_length):
    msg("Loading depth file:", abundance_file)
    start = time.time()
    depth_file_header = pd.read_csv(abundance_file, sep='\t', lineterminator='\n', nrows=0)
    num_columns = len(depth_file_header.columns)
    pre_depths = calc_depths(abundance_file, num_columns, num_contigs, min_contig_length)
    msg(f"Loaded depth file in {time.time() - start:.3f} seconds")
    return (num_columns - 3) // 2, pre_depths


def prepare_model_input(contig_lens_filtered, inpts, pre_depths, num_samples,min_contig_length):
    model_data_in = [np.zeros((len(contig_lens_filtered), 136), dtype='float')]
    for i in range(len(inpts)):
        model_data_in.append(inpts[i][contig_lens_filtered >= min_contig_length])
    model_data_in.append(pre_depths.reshape((-1, num_samples, 1)))

    with tf.device('/cpu:0'):
        datasets = [tf.data.Dataset.from_tensor_slices(arr) for arr in model_data_in]
    dataset = tf.data.Dataset.zip(tuple(datasets))
    dataset = dataset.batch(8192 // 2)
    
    return model_data_in, dataset

def embed_contigs(model, model2, dataset, num_contigs,num_samples):
    y20_cat = np.zeros((num_contigs, 512))
    y21_cat = np.zeros((num_contigs, 16 * num_samples))
    done = 0
    
    progress_bar = tqdm(total=num_contigs // (8192 // 2) + 1 if num_contigs % (8192 // 2) else 0)
    for idx, b in enumerate(dataset):
        otmp = model.predict(x=b, verbose=0, batch_size=8192 // 2)
        y20_cat[done:done + len(otmp), :] = model2.predict(x=b, verbose=0, batch_size=8192 // 2)
        y21_cat[done:done + len(otmp), :] = otmp
        done += len(otmp)
        progress_bar.update(1)
    progress_bar.close()
    y20_cat /= np.linalg.norm(y20_cat, axis=1, keepdims=True)
    y21_cat /= np.linalg.norm(y21_cat, axis=1, keepdims=True)
    
    return y20_cat, y21_cat

if __name__ == "__main__":
    parser = CustomHelpParser(description=bcolors.BOLD+"GenomeFace Prerelease"+ bcolors.ENDC + "\n  - A next-generation tool for metagenomic binning, using deep learning and multi-GPU accelerated clustering. Ideal for large-scale, real-world data." +"\n\nThe Exabiome Project (Lawrence Berkeley National Laboratory) \n  - Contact rlettich@lbl.gov for issues or unexpected poor performance.", formatter_class=CustomHelpFormatter)
    
    parser.add_argument('-i', dest='input_file', required=True, help="Input FASTA file containing metagenome assembly (optionally gzipped).")
    parser.add_argument('-a', dest='abundance_file', required=True, help="MetaBAT 2 style TSV file containing abundance data. Typically produced by the `jgi_summarize_depths` or `coverm` programs.")
    parser.add_argument('-g', dest='markers_file', required=True, help="Input TSV file describing marker genes found on each contig. Can be produced by the included `markersgf` program.")
    parser.add_argument('-o', dest='output_folder', required=True, help="Output folder for writing bin FASTA files")
    parser.add_argument('-m', dest='min_contig_length', type=int, default=1500, help="Minimum contig length to be considered for binning (default: 1500).")
    parser.add_argument('-p', dest='min_purity', type=int, default=85, help="Minimum marker gene estimated %% purity for selecting clusters for output. Balances Precsion / Recall (default: 85)")
    #parser.add_argument('-b', dest='min_bin_size', type=int, default=200000, help="Minimum bases in bin for writing to FASTA file (Default: 200,000).")
    parser.add_argument('-s', dest='is_multiple_sample', action='store_true', help="Specifies that the input FASTA is multiple single sample assemblies, concatenated. Contig names should be Prefixed by per assembly name With ending in 'C'. e.g. asm_oneCsequence5 ")
    parser.add_argument('-b', dest='optimize_balance', action='store_true', help="Use Bayesian Optimizaiton to Optimize Balance Between Compositional and Abundance Distances (Default: False).")

    args = parser.parse_args()
    main(args.input_file, args.abundance_file, args.markers_file, args.output_folder, args.min_contig_length, args.is_multiple_sample, args.min_purity, args.optimize_balance)
