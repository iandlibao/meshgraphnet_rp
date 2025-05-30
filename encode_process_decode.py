# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Core learned graph net model."""

import collections
from math import ceil
from collections import OrderedDict
import functools
import torch
from torch import nn as nn
import torch_scatter
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets', 'global_features'])
MultiGraphWithPos = collections.namedtuple('Graph', ['node_features', 'edge_sets', 'target_feature', 'model_type', 'node_dynamic'])

import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')


class LazyMLP(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        input = input.to(device)
        y = self.layers(input)
        return y

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, dropout, bidirectional):
        super().__init__()

        self.gru = nn.GRU(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional)
        self.output_fc = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.output_fc(out[-1,:,:])
        return out


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.LazyLinear(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.to(device)

    def forward(self, input, index):

        latent = self.linear_layer(input)
        latent = self.leaky_relu(latent)
        result = torch.zeros(*latent.shape)

        result = scatter_softmax(latent.float(), index, dim=0)
        result = result.type(result.dtype)
        return result


class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size, message_passing_aggregator, has_global, global_model_in_processor, attention=False):
        super().__init__()
        self.mesh_edge_model = model_fn(output_size)
        self.world_edge_model = model_fn(output_size)
        self.node_model = model_fn(output_size)
        self.attention = attention
        self.has_global = has_global
        self.global_model_in_processor = global_model_in_processor

        if self.has_global == True and self.global_model_in_processor == True:
            self.global_model = model_fn(output_size)
        
        if attention:
            self.attention_model = AttentionModel()
        self.message_passing_aggregator = message_passing_aggregator

        self.linear_layer = nn.LazyLinear(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def _update_edge_features(self, node_features, edge_set, global_features, has_global):
        """Aggregrates node features, and applies edge function."""
        senders = edge_set.senders.to(device)
        receivers = edge_set.receivers.to(device)
        sender_features = torch.index_select(input=node_features, dim=0, index=senders)
        receiver_features = torch.index_select(input=node_features, dim=0, index=receivers)

        if has_global==True:
            cast_global = global_features.repeat(sender_features.shape[0],1)
            features = [sender_features, receiver_features, edge_set.features, cast_global]

        else:
            features = [sender_features, receiver_features, edge_set.features]

        #features = [sender_features, receiver_features, edge_set.features]
        features = torch.cat(features, dim=-1)
        if edge_set.name == "mesh_edges":
            return self.mesh_edge_model(features)
        else:
            return self.world_edge_model(features)

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        data = data.to(device)
        segment_ids = segment_ids.to(device)
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape).to(device)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'std':
            result = torch_scatter.scatter_std(data.float(), segment_ids, out=result, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def _update_node_features(self, node_features, edge_sets, global_features, has_global):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        features = [node_features]
        for edge_set in edge_sets:
            if self.attention and self.message_passing_aggregator == 'pna':
                attention_input = self.linear_layer(edge_set.features)
                attention_input = self.leaky_relu(attention_input)
                attention = F.softmax(attention_input, dim=0)
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation='sum'))
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation='mean'))
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation='max'))
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation='min'))
            elif self.attention:
                attention_input = self.linear_layer(edge_set.features)
                attention_input = self.leaky_relu(attention_input)
                attention = F.softmax(attention_input, dim=0)
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation=self.message_passing_aggregator))
            elif self.message_passing_aggregator == 'pna':
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='sum'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='mean'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='max'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='min'))
            else:
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers, num_nodes,
                                                    operation=self.message_passing_aggregator))
        
        if has_global==True:
            cast_global = global_features.repeat(num_nodes,1)
            features.append(cast_global)

        features = torch.cat(features, dim=-1)
        return self.node_model(features)

    def _update_global_features(self, node_features, edge_sets, global_features, has_global):
        features = []
        node_aggr = torch.mean(node_features, 0)
        features.append(node_aggr)

        for edge_set in edge_sets:
            edge_aggr = torch.mean(edge_set.features, 0)
            features.append(edge_aggr)
        
        features.append(global_features)

        features = torch.cat(features, dim=-1)

        return self.global_model(features)

    # def forward(self, graph, mask=None):
    #     """Applies GraphNetBlock and returns updated MultiGraph."""
    #     # apply edge functions
    #     new_edge_sets = []
    #     for edge_set in graph.edge_sets:
    #         updated_features = self._update_edge_features(graph.node_features, edge_set, graph.global_features, self.has_global)
    #         new_edge_sets.append(edge_set._replace(features=updated_features))
    #     new_edge_sets = [es._replace(features=es.features + old_es.features)
    #                      for es, old_es in zip(new_edge_sets, graph.edge_sets)]

    #     # apply node function
    #     new_node_features = self._update_node_features(graph.node_features, new_edge_sets, graph.global_features, self.has_global)
    #     # add residual connections
    #     new_node_features += graph.node_features
    #     if mask is not None:
    #         mask = mask.repeat(new_node_features.shape[-1])
    #         mask = mask.view(new_node_features.shape[0], new_node_features.shape[1])
    #         new_node_features = torch.where(mask, new_node_features, graph.node_features)


    #     #apply global function
        
    #     if(self.has_global==True):
    #         if self.global_model_in_processor ==  True:
    #             new_global_features = self._update_global_features(new_node_features, new_edge_sets, graph.global_features, self.has_global)
    #             #new_global_features = self._update_global_features(graph.node_features, graph.edge_sets, graph.global_features, self.has_global)
    #             new_global_features += graph.global_features #residual connection
    #         else:
    #             new_global_features = graph.global_features
    #     else:
    #         new_global_features = -1    
        
       
        
    #     return MultiGraph(new_node_features, new_edge_sets, new_global_features)

    def forward(self, graph, mask=None):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set, graph.global_features, self.has_global)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets, graph.global_features, self.has_global)

        #apply global function
        if(self.has_global==True):
            new_global_features = self._update_global_features(new_node_features, new_edge_sets, graph.global_features, self.has_global)
            new_global_features += graph.global_features #residual connection
        else:
            new_global_features = -1    
        
        # add residual connections
        new_node_features += graph.node_features
        if mask is not None:
            mask = mask.repeat(new_node_features.shape[-1])
            mask = mask.view(new_node_features.shape[0], new_node_features.shape[1])
            new_node_features = torch.where(mask, new_node_features, graph.node_features)
        new_edge_sets = [es._replace(features=es.features + old_es.features)
                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        return MultiGraph(new_node_features, new_edge_sets, new_global_features)


class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, latent_size, has_global, global_latent_size):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size
        self.has_global = has_global
        self.node_model = self._make_mlp(latent_size)
        self.mesh_edge_model = self._make_mlp(latent_size)
        self.world_edge_model = self._make_mlp(latent_size)

        if self.has_global == True:
            self.global_model = self._make_mlp(global_latent_size)

    def forward(self, graph):
        node_latents = self.node_model(graph.node_features)
        new_edges_sets = []

        if self.has_global == True:
            global_latents = self.global_model(graph.global_features)
        else:
            global_latents = -1

        for index, edge_set in enumerate(graph.edge_sets):
            if edge_set.name == "mesh_edges":
                feature = edge_set.features
                latent = self.mesh_edge_model(feature)
                new_edges_sets.append(edge_set._replace(features=latent))
            else:
                feature = edge_set.features
                latent = self.world_edge_model(feature)
                new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets, global_latents)


class Decoder(nn.Module):
    """Decodes node features from graph."""

    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.model = make_mlp(output_size)

    def forward(self, graph):
        return self.model(graph.node_features)

class Processor(nn.Module):
    '''
    This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection(features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent graph will be feed into original processor

    Option: choose whether to normalize the high rank node connection
    '''

    def __init__(self, make_mlp, output_size, message_passing_steps, message_passing_aggregator, has_global, global_model_in_processor, attention=False,
                 stochastic_message_passing_used=False):
        super().__init__()
        self.stochastic_message_passing_used = stochastic_message_passing_used
        self.graphnet_blocks = nn.ModuleList()
        for index in range(message_passing_steps):
            self.graphnet_blocks.append(GraphNetBlock(model_fn=make_mlp, output_size=output_size,
                                                      message_passing_aggregator=message_passing_aggregator,
                                                      has_global=has_global,
                                                      global_model_in_processor=global_model_in_processor,
                                                      attention=attention))

    def forward(self, latent_graph, normalized_adj_mat=None, mask=None):
        for graphnet_block in self.graphnet_blocks:
            if mask is not None:
                latent_graph = graphnet_block(latent_graph, mask)
            else:
                latent_graph = graphnet_block(latent_graph)
        return latent_graph
class RNNEncoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_gru, node_features_length, edge_features_length,latent_size, has_global, make_mlp, global_latent_size):
        super().__init__()
        self._make_gru = make_gru
        self._latent_size = latent_size
        self.node_mode = self._make_gru(node_features_length, latent_size)
        self.mesh_edge_model = self._make_gru(edge_features_length, latent_size)
        self.has_global = has_global

        if self.has_global == True:
            self._make_mlp = make_mlp
            self.global_model = self._make_mlp(global_latent_size)
        

    def forward(self, graph_list):
        gru_node_f_list = []
        gru_edge_f_list = []
        for graph in graph_list:
            gru_node_f_list.append(graph.node_features)
            gru_edge_f_list.append(graph.edge_sets[0].features)
        gru_node_input = torch.stack(gru_node_f_list)
        gru_edge_input = torch.stack(gru_edge_f_list)

        node_latents = self.node_mode(gru_node_input)
        edge_latents = self.mesh_edge_model(gru_edge_input)
        
        mesh_edges = EdgeSet(
            name='mesh_edges',
            features = edge_latents,
            receivers=graph_list[0].edge_sets[0].receivers,
            senders=graph_list[0].edge_sets[0].senders
        )

        if self.has_global == True:
            global_latents = self.global_model(graph_list[0].global_features)
        else:
            global_latents = -1

        return MultiGraph(node_latents, [mesh_edges], global_latents)

        #shape should be
        print("foo")
        # node_latents = self.node_model(graph.node_features)
        # new_edges_sets = []

        # if self.has_global == True:
        #     global_latents = self.global_model(graph.global_features)
        # else:
        #     global_latents = -1

        # for index, edge_set in enumerate(graph.edge_sets):
        #     if edge_set.name == "mesh_edges":
        #         feature = edge_set.features
        #         latent = self.mesh_edge_model(feature)
        #         new_edges_sets.append(edge_set._replace(features=latent))
            
        # return MultiGraph(node_latents, new_edges_sets, global_latents)

class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
                 output_size,
                 latent_size,
                 num_layers,
                 message_passing_aggregator, message_passing_steps, attention, has_global,
                 is_rnn_encoder,
                 node_features_length,
                 edge_features_length,
                 global_model_in_processor,
                 global_latent_size):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator

        self._attention = attention
        self.has_global = has_global

        if is_rnn_encoder == False:
            self.encoder = Encoder(make_mlp=self._make_mlp, latent_size=self._latent_size, has_global=self.has_global, global_latent_size=global_latent_size)
        else:
            self.encoder = RNNEncoder(make_gru=self._make_gru, node_features_length=node_features_length,
                    edge_features_length=edge_features_length, latent_size=self._latent_size,
                    has_global=self.has_global, make_mlp=self._make_mlp, global_latent_size=global_latent_size)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps,
                                   message_passing_aggregator=self._message_passing_aggregator, 
                                   has_global=self.has_global,
                                   global_model_in_processor=global_model_in_processor,
                                   attention=self._attention,
                                   stochastic_message_passing_used=False)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)

    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def _make_gru(self, input_size, latent_size, layer_norm=True):
        network = GRU(input_size=input_size,
            hidden_size=latent_size,
            num_layers=2,
            batch_first=False,
            dropout=0.2,
            bidirectional=False)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=latent_size))
        return network


    def forward(self, graph, is_training, world_edge_normalizer=None):
        """Encodes and processes a multigraph, and returns node features."""
        latent_graph = self.encoder(graph)
        latent_graph = self.processor(latent_graph)
        return self.decoder(latent_graph)

