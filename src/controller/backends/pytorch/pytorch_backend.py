# -*-coding:utf-8-*-
import torch
import collections
from torch import jit
from torch.autograd import Variable
from model.core import Model, Layer, Domain_Layer_Types
from controller.backends.backend import Backend


Operation_to_domain_layer_class_dict = {'aten::addmm': Domain_Layer_Types.Dense,
                                        'aten::dropout': Domain_Layer_Types.Dropout,
                                        'aten::relu_': Domain_Layer_Types.ReLu,
                                        'aten::relu': Domain_Layer_Types.ReLu,
                                        'aten::_convolution': Domain_Layer_Types.Conv2D,
                                        'aten::batch_norm': Domain_Layer_Types.BatchNormalization,
                                        'aten::max_pool2d': Domain_Layer_Types.MaxPooling2D,
                                        'aten::adaptive_avg_pool2d': Domain_Layer_Types.AveragePooling2D,
                                        'aten::add_': Domain_Layer_Types.Add,
                                        'aten::upsample_bilinear2d': Domain_Layer_Types.Default,
                                        'aten::cat': Domain_Layer_Types.Default,
                                        'aten::view': Domain_Layer_Types.View,
                                        'aten::reshape': Domain_Layer_Types.Reshape,
                                        'aten::sigmoid': Domain_Layer_Types.Default,
                                        'aten::constant_pad_nd': Domain_Layer_Types.Default,
                                        'aten::flatten': Domain_Layer_Types.Default,
                                        'aten::size': Domain_Layer_Types.Default,
                                        'aten::mul': Domain_Layer_Types.Default,
                                        'aten::unsqueeze': Domain_Layer_Types.Default,
                                        'aten::select': Domain_Layer_Types.Default,
                                        'aten::slice': Domain_Layer_Types.Default,
                                        'aten::clone': Domain_Layer_Types.Default
                                        }


class PytorchBackend(Backend):
    def load_model(self, file_path):
        model = Model()
        model.native_model = torch.load(file_path).cpu()
        input_shape = torch.unsqueeze(torch.randn(tuple(model.input_shape)), dim=0)
        self._pytorch_model_parser(model, input_shape)
        return model

    def load_weights(self, file_path):
        return torch.load(file_path)

    def predict(self, model, model_input):
        def init_output_activation(output_classes):
            if output_classes == 2:
                output_activation = torch.nn.Sigmoid()
            else:
                output_activation = torch.nn.Softmax()
            return output_activation
        model_input = Variable(torch.from_numpy(model_input).float(), requires_grad=True)
        model.native_model.eval()
        output = model.native_model(model_input)
        if isinstance(output, collections.OrderedDict):
            output = output['out']
        if model.add_output_activation:
            output_activation = init_output_activation(model)
            output = output_activation(output.detach()[0])
        return output.detach().numpy()

    def _pytorch_model_parser(self, model, model_input=torch.randn(1, 3, 224, 224)):
        model.native_model.eval()
        trace, _ = jit.get_trace_graph(model.native_model, args=(model_input,))
        node_name_to_node = dict()
        graph = trace.graph()
        for node in graph.nodes():
            if node.kind() == 'prim::ListConstruct':
                connect_to_layer = False
                for node_input in node.inputs():
                    if node_input.node().kind() in Operation_to_domain_layer_class_dict:
                        connect_to_layer = True
                        break
                if connect_to_layer:
                    node_name_to_node[next(node.outputs()).debugName()] = node
            if node.kind() in Operation_to_domain_layer_class_dict:
                node_name_to_node[next(node.outputs()).debugName()] = node

        seen = set()
        self.index = 0
        self.node_to_layer = dict()
        for node in graph.outputs():
            self._find_connection_nodes_from_output(node.debugName(), node_name_to_node, model, seen)
        model.layers.reverse()

    def _find_connection_nodes_from_output(self, input_name, node_name_to_node, model, seen):
        if input_name in seen:
            return self.node_to_layer[input_name]
        if input_name not in seen and input_name in node_name_to_node:
            self.index += 1
            n = node_name_to_node[input_name]
            if n.kind() != 'prim::ListConstruct':
                layer_class = Operation_to_domain_layer_class_dict[n.kind()].value
            else:
                layer_class = 'Default'
            layer = Layer(n.kind() + str(self.index), layer_class, index=self.index)
            attrs = {k: n[k] for k in n.attributeNames()}
            attrs = str(attrs).replace("'", ' ')
            layer.config = attrs
            layer.scope_name = n.scopeName()
            model.layers_append(layer)
            seen.add(input_name)
            self.node_to_layer[input_name] = layer
            for node_input in n.inputs():
                input_layer = self._find_connection_nodes_from_output(node_input.debugName(), node_name_to_node,
                                                                      model, seen)
                if input_layer:
                    layer.inbound_layers.append(input_layer)
            return layer
        return None

    def _find_module(self, model, target_layer_index):
        assert target_layer_index is not None, "Please select a layer"
        scope_name = model.get_layer_by_index(target_layer_index).scope_name
        if scope_name is None:
            scope_name = model.layers[1].scope_name
        subscope_str_list = scope_name.split('/')[1:]
        key_path = list()
        for subscope_str in subscope_str_list:
            key_path.append(subscope_str[subscope_str.find('[') + 1: subscope_str.find(']')])
        module = model.native_model
        for key in key_path:
            module = module._modules[key]
        return module

    def forward_pass_up_to_layer(self, model, model_input, target_layer_index):
        model.native_model.eval()
        input_pytorch_tensor = Variable(torch.from_numpy(model_input).float(), requires_grad=True)
        module = self._find_module(model, target_layer_index)
        hook = Hook(module)
        model.native_model(input_pytorch_tensor)
        hook.close()
        return hook.output.detach().numpy()[0]

    def gradients_of_output_wrt_input(self, model, model_input, model_input_layer_index, class_index):
        model.native_model.eval()
        input_tensor = Variable(torch.from_numpy(model_input).float(), requires_grad=True)
        if model_input_layer_index != 0:
            module = self._find_module(model, model_input_layer_index)
            if module is not None:
                hook = Hook(module, backward=True)
        model_output = model.native_model(input_tensor)
        if isinstance(model_output, collections.OrderedDict):
            model_output = model_output['out']
        model.native_model.zero_grad()
        one_hot_output = torch.FloatTensor(model_output.size()).zero_()
        one_hot_output[0][class_index] = 1
        model_output.backward(gradient=one_hot_output)
        if model_input_layer_index == 0:
            gradient = input_tensor.grad.detach().numpy()
            return gradient.transpose(0, 2, 3, 1)
        hook.close()
        gradient = hook.output[0].detach().numpy()
        return gradient.transpose(0, 2, 3, 1)

    def register_guided_relu_gradients(self, model):
        self.forward_relu_output = collections.OrderedDict()
        
        def forward_hook_fn(module, input, output):
            value=  self.forward_relu_output.get(str(module.__hash__())+str(output.shape))
            self.forward_relu_output[str(module.__hash__())+str(output.shape)] = output

        def backward_hook_fn(module, grad_in, grad_out):
            # output the grad of the model wrt. layer (only positive)
            corresponding_forward_output = self.forward_relu_output[str(module.__hash__())+str(grad_in[0].shape)]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            positive_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            return (positive_grad_out,)

        def register_relu_hooks(module):
            if len(module._modules) == 0:
                if isinstance(module, torch.nn.ReLU):
                    module.register_backward_hook(backward_hook_fn)
                    module.register_forward_hook(forward_hook_fn)
            else:
                for _, submodule_value in module._modules.items():
                    register_relu_hooks(submodule_value)
        register_relu_hooks(model.native_model)
        return model


# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook:
    def __init__(self, module, backward=False):
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        self.input = None
        self.output = None

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()
