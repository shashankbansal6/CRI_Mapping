#Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven.neuron import LIFNode
from quant_layer import *

class BN_Folder():
    def __init__(self):
        super().__init__()
        
    def fold(self, model):

        new_model = copy.deepcopy(model)

        module_names = list(new_model._modules)

        for k, name in enumerate(module_names):

            if len(list(new_model._modules[name]._modules)) > 0:
                
                new_model._modules[name] = self.fold(new_model._modules[name])

            else:
                if isinstance(new_model._modules[name], nn.BatchNorm2d) or isinstance(new_model._modules[name], nn.BatchNorm1d):
                    if isinstance(new_model._modules[module_names[k-1]], nn.Conv2d) or isinstance(new_model._modules[module_names[k-1]], nn.Linear):

                        # Folded BN
                        folded_conv = self._fold_conv_bn_eval(new_model._modules[module_names[k-1]], new_model._modules[name])

                        # Replace old weight values
                        #new_model._modules.pop(name) # Remove the BN layer
                        new_model._modules[module_names[k]] = nn.Identity()
                        new_model._modules[module_names[k-1]] = folded_conv # Replace the Convolutional Layer by the folded version

        return new_model


    def _bn_folding(self, prev_w, prev_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, model_2d):
        if prev_b is None:
            prev_b = bn_rm.new_zeros(bn_rm.shape)
            
        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
          
        if model_2d:
            w_fold = prev_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
        else:
            w_fold = prev_w * (bn_w * bn_var_rsqrt).view(-1, 1)
        b_fold = (prev_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

        return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)
        
    def _fold_conv_bn_eval(self, prev, bn):
        assert(not (prev.training or bn.training)), "Fusion only for eval!"
        fused_prev = copy.deepcopy(prev)
        
        if isinstance(bn, nn.BatchNorm2d):
            fused_prev.weight, fused_prev.bias = self._bn_folding(fused_prev.weight, fused_prev.bias,
                                 bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, True)
        else:
            fused_prev.weight, fused_prev.bias = self._bn_folding(fused_prev.weight, fused_prev.bias,
                                 bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, False)

        return fused_prev

class Quantize_Network():
    w_alpha = 1 #Range of the parameter
    w_bits = 16
    w_delta = w_alpha/(2**(w_bits-1)-1)
    w_delta_last_layer = None 
    weight_quant = weight_quantize_fn(w_bits)
    weight_quant.wgt_alpha = w_alpha
    
    def quantize(self, model):
        new_model = copy.deepcopy(model)
        module_names = list(new_model._modules)
        
        for k, name in enumerate(module_names):
            if len(list(new_model._modules[name]._modules)) > 0 and not isinstance(new_model._modules[name], MultiStepLIFNode):
                if name == 'patch_embed':
                    new_model._modules[name] = self.quantize(new_model._modules[name])
                elif name == 'block':
                    new_model._modules[name] = self.quantize_block(new_model._modules[name])
                else:
                    print('Unquantized: ', name)
            else:
                quantized_layer = self._quantize(new_model._modules[name])
                new_model._modules[name] = quantized_layer
        
        return new_model
    
    def quantize_block(self, model):
        new_model = copy.deepcopy(model)
        module_names = list(new_model._modules)
        
        for k, name in enumerate(module_names):
            
            if len(list(new_model._modules[name]._modules)) > 0 and not isinstance(new_model._modules[name], MultiStepLIFNode):
                if name.isnumeric() or name == 'attn' or name == 'mlp':
                    print('Quantized: ',name)
                    new_model._modules[name] = self.quantize_block(new_model._modules[name])
                else:
                    print('Unquantized: ', name)
            else:
                if name == 'attn_lif':
                    continue
                #     new_model._modules[name] = self._quantize_attn(new_model._modules[name])
                else:
                    new_model._modules[name] = self._quantize(new_model._modules[name])
                #     print('Single layer:',name, Quantize_Network.w_delta)
                # if isinstance(new_model._modules[name], MultiStepLIFNode):
                #     print('Threshold: ',new_model._modules[name].v_threshold)
        return new_model
    
    def _quantize(self, layer):
        if isinstance(layer, MultiStepLIFNode):
            return self._quantize_LIF(layer)

        elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            return self._quantize_layer(layer)
        
        else:
            return layer
        
#     def _quantize_block(self, layer):
        
#         if isinstance(layer, MultiStepLIFNode):
#             return self._quantize_LIF_block(layer)
        
#         elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
#             return self._quantize_layer_block(layer)
        
#         else:
#             return layer
        
    def _quantize_layer(self, layer):
        quantized_layer = copy.deepcopy(layer)
        
        weight_range = abs(max(layer.weight.flatten()) - min(layer.weight.flatten()))

        Quantize_Network.w_alpha = abs(max(layer.weight.flatten()) - min(layer.weight.flatten()))
        Quantize_Network.w_delta = Quantize_Network.w_alpha/(2**(Quantize_Network.w_bits-1)-1)
        Quantize_Network.weight_quant = weight_quantize_fn(Quantize_Network.w_bits) #reinitialize the weight_quan
        Quantize_Network.weight_quant.wgt_alpha = Quantize_Network.w_alpha
        
        layer.weight = nn.Parameter(Quantize_Network.weight_quant(layer.weight))
        quantized_layer.weight = nn.Parameter(layer.weight/Quantize_Network.w_delta)

        layer.bias = nn.Parameter(Quantize_Network.weight_quant(layer.bias))
        quantized_layer.bias = nn.Parameter(layer.bias/Quantize_Network.w_delta)
        
        
        return quantized_layer

    
    def _quantize_LIF(self,layer):
        
        layer.v_threshold = layer.v_threshold/Quantize_Network.w_delta
        
        return layer
    
#     def _quantize_layer_block(self, layer):
#         quantized_layer = copy.deepcopy(layer)
        
#         layer.weight = nn.Parameter(Quantize_Network.weight_quant(layer.weight))
#         quantized_layer.weight = nn.Parameter(layer.weight/Quantize_Network.w_delta)

#         layer.bias = nn.Parameter(Quantize_Network.weight_quant(layer.bias))
#         quantized_layer.bias = nn.Parameter(layer.bias/Quantize_Network.w_delta)
        
#         return quantized_layer

    
#     def _quantize_LIF_block(self,layer):
    
#         layer.v_threshold = layer.v_threshold/Quantize_Network.w_delta
        
#         return layer

    
# class CRI_Converter():
#     HIGH_SYNAPSE_WEIGHT = 1e6
#     def __init__(self):
#         pass
        
#     def input_converter(self, input_data):
#         #TODO: Convert input data into spikes
#         pass
        
    
#     def conv2d_converter(self):
#         #TODO: Convert conv2d layer 
#         pass

class CRIConverter:
    def __init__(self, scalar=1e6):
        self.axonsDict = {}
        self.neuronsDict = {}
        self.scalar = scalar
        
    @staticmethod 
    def _conv2dOutputSize(layer, inputSize):
        H_out = (inputSize[0] + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
        W_out = (inputSize[1] + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
        return [layer.out_channels, int(H_out), int(W_out)]
    
    @staticmethod
    def _maxPoolOutputSize(layer, inputSize):
        H_out = (inputSize[1] + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) // layer.stride + 1
        W_out = (inputSize[2] + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) // layer.stride + 1
        return [inputSize[0], int(H_out), int(W_out)]
    
    @staticmethod
    def _avgPoolOutputSize(layer, inputSize):
        H_out = (inputSize[1] + layer.padding*2 - (layer.kernel_size-1))/layer.stride +1
        W_out = (inputSize[2] + layer.padding*2 - (layer.kernel_size-1))/layer.stride +1
        return [inputSize[0],int(H_out),int(W_out)]   
    
    def conv2dToCRI(self, inputs, output, layer, layerIdx, axonsDict=None, neuronsDict=None):
        """
        Convert a convolutional layer to a CRI representation.
        
        Parameters:
            inputs (torch.Tensor): The input tensor to the convolutional layer.
            output (torch.Tensor): The output tensor of the convolutional layer.
            layer (torch.nn.Conv2d): The convolutional layer to convert.
            layerIdx (int): The index of the current convolutional layer within the model.
            axonsDict (dict, optional): A dictionary that maps axon IDs to their synapse information. Defaults to None.
            neuronsDict (dict, optional): A dictionary that maps neuron IDs to their synapse information. Defaults to None.
        
        Returns:
            None
        """
            
        Hk, Wk = layer.kernel_size
        filters = layer.weight.detach().cpu().numpy()
        pad_top, pad_left = Hk // 2, Wk // 2

        # Define a helper function to reduce code redundancy
        def process_patch(input_slice, output_slice, filters, axonsDict, neuronsDict, layerIdx, channel=None):
            """
            Process a given patch of input neurons or axons and update axonsDict or neuronsDict.
            """
            for filIdx, fil in enumerate(filters):
                postSynapticID = str(output_slice[filIdx])
                fil = fil[channel] if channel is not None else fil[0]
                for i, row in enumerate(input_slice):
                    for j, elem in enumerate(row):
                        key = str(elem) if layerIdx != 0 else elem
                        synapse_info = (postSynapticID, int(fil[i, j]))
                        (neuronsDict if layerIdx != 0 else axonsDict)[key].append(synapse_info)
        
        if layerIdx == 0:
            Hi, Wi = inputs.shape
            for row in range(pad_top, Hi - pad_top):
                for col in range(pad_left, Wi - pad_left):
                    patch = inputs[row - pad_top:row + pad_top + 1, col - pad_left:col + pad_left + 1]
                    output_slice = output[:, row - pad_top, col - pad_left]
                    process_patch(patch, output_slice, filters, axonsDict, neuronsDict, layerIdx)
        else:
            C, Hi, Wi = inputs.shape
            for channel in range(C):
                for row in range(pad_top, Hi - pad_top):
                    for col in range(pad_left, Wi - pad_left):
                        patch = inputs[channel, row - pad_top:row + pad_top + 1, col - pad_left:col + pad_left + 1]
                        output_slice = output[:, row - pad_top, col - pad_left]
                        process_patch(patch, output_slice, filters, axonsDict, neuronsDict, layerIdx, channel)


    def poolingToCRI(self, inputs, output, neuronsDict, layer, pool_type='max'):
        """
        Convert a pooling layer to a CRI representation.
        
        Parameters:
            inputs (torch.Tensor): The input tensor to the pooling layer.
            output (torch.Tensor): The output tensor of the pooling layer.
            neuronsDict (dict): A dictionary that maps neuron IDs to their synapse information.
            layer (torch.nn.MaxPool2d or torch.nn.AvgPool2d): The pooling layer to convert.
            pool_type (str, optional): The type of pooling operation, either 'max' or 'avg'. Defaults to 'max'.

        Returns:
            None
        """
        C, Hi, Wi = inputs.shape
        kernel_size = layer.kernel_size
        pad_size = kernel_size // 2
        stride = 2

        for channel in range(C):
            for row in range(0, Hi, stride):
                for col in range(0, Wi, stride):
                    patch = inputs[channel, row:row+kernel_size, col:col+kernel_size]
                    postSynapticID = str(output[channel, row//stride, col//stride])

                    # Determine the pooling value
                    if pool_type == 'max':
                        pooling_value = patch.max() * self.scalar
                    elif pool_type == 'avg':
                        pooling_value = patch.mean() * self.scalar

                    # Update neuronsDict with postSynapticID and pooling_value
                    for preSynNeuron in patch.flatten():
                        neuronsDict[str(preSynNeuron)].append((postSynapticID, pooling_value))
    
    # Wrapper functions
    def maxPoolToCRI(self, inputs, output, neuronsDict, layer):
        return self.poolingToCRI(inputs, output, neuronsDict, layer, pool_type='max')

    def avgPoolToCRI(self, inputs, output,neuronsDict, layer):
        return self.poolingToCRI(inputs, output, neuronsDict, layer, pool_type='avg')
    
    def linearToCRI(self, inputs, layer, neuronsDict, outputNeurons=None):
        """
        Convert a linear layer to a CRI representation.
        
        Parameters:
            inputs (torch.Tensor): The input tensor to the linear layer.
            output (torch.Tensor): The output tensor of the linear layer.
            layer (torch.nn.Linear): The linear layer to convert.
            layerIdx (int): The index of the current linear layer within the model.
            neuronsDict (dict): A dictionary that maps neuron IDs to their synapse information.
            outputNeurons (list, optional): A list to store the neuron IDs of the output neurons.
                                        Defaults to None, in which case a new list is created.

        Returns:
            None
        """
        if outputNeurons is None:
            outputNeurons = []

        # Flatten the input tensor
        inputs = inputs.flatten()

        # Detach the weight matrix from the computational graph and convert to NumPy array
        weight = layer.weight.detach().cpu().numpy()

        # Determine neuron index offsets for the current and next layers
        currLayerNeuronIdxOffset, nextLayerNeuronIdxOffset = inputs[0], inputs[-1] + 1

        # Iterate over neurons and weights in the transposed weight matrix
        for baseNeuronIdx, neuron in enumerate(weight.T):
            neuronID = str(baseNeuronIdx + currLayerNeuronIdxOffset)
            # Create a list of synapses with non-zero weights
            neuronEntry = [(str(basePostSynapticID + nextLayerNeuronIdxOffset), int(synapseWeight))
                        for basePostSynapticID, synapseWeight in enumerate(neuron) if synapseWeight != 0]
            neuronsDict[neuronID] = neuronEntry

        # Instantiate output neurons
        for baseNeuronIdx in range(layer.out_features):
            neuronID = str(baseNeuronIdx + nextLayerNeuronIdxOffset)
            neuronsDict[neuronID] = []
            outputNeurons.append(neuronID)
    
    def convBiasAxons(self, layer, axonsDict, axonOffset, outputs):
        """
        Convert biases of a convolutional layer to axon entries in the CRI representation.
        
        Parameters:
            layer (torch.nn.Conv2d): The convolutional layer to convert.
            axonsDict (dict): A dictionary that maps axon IDs to their synapse information.
            axonOffset (int): An offset value for axon IDs.
            outputs (torch.Tensor): The output tensor of the convolutional layer.

        Returns:
            None
        """
        # Detach the biases from the computational graph and convert to NumPy array
        biases = layer.bias.detach().cpu().numpy()

        # Iterate over biases and create axon entries
        for biasIdx, bias in enumerate(biases):
            biasID = f'a{biasIdx + axonOffset}'
            axonsDict[biasID] = [(str(neuronIdx), int(bias)) for neuronIdx in outputs[biasIdx].flatten()]

    def linearBiasAxons(self, layer, axonsDict, axonOffset, outputs):
        """
        Convert biases of a linear layer to axon entries in the CRI representation.
        
        Parameters:
            layer (torch.nn.Linear): The linear layer to convert.
            axonsDict (dict): A dictionary that maps axon IDs to their synapse information.
            axonOffset (int): An offset value for axon IDs.
            outputs (torch.Tensor): The output tensor of the linear layer.

        Returns:
            None
        """
        # Detach the biases from the computational graph and convert to NumPy array
        biases = layer.bias.detach().cpu().numpy()

        # Iterate over biases and create axon entries
        for biasIdx, bias in enumerate(biases):
            biasID = f'a{biasIdx + axonOffset}'
            axonsDict[biasID] = [(str(outputs[biasIdx]), int(bias))]