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

    
class CRI_Converter():
    HIGH_SYNAPSE_WEIGHT = 1e6
    def __init__(self):
        pass
        
    def input_converter(self, input_data):
        #TODO: Convert input data into spikes
        pass
        
    
    def conv2d_converter(self):
        #TODO: Convert conv2d layer 
        pass
        
    
    