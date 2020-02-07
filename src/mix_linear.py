##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Cheolhyoung Lee
## Department of Mathematical Sciences, KAIST
## Email: bloodwass@kaist.ac.kr
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch.autograd.function import InplaceFunction
from torch.nn import Parameter
import torch.nn._VF as _VF
from torch.nn.utils.rnn import PackedSequence
import math
import warnings
import numbers
import torch.nn.init as init

_rnn_impls = {
    'RNN_TANH': _VF.rnn_tanh,
    'RNN_RELU': _VF.rnn_relu,
}


def apply_permutation(tensor, permutation, dim=1):
    # type: (Tensor, Tensor, int) -> Tensor
    return tensor.index_select(dim, permutation)


class RNNBase(torch.nn.Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional']

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        self.flatten_parameters()
        self.reset_parameters()

    def __setattr__(self, attr, value):
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # keep self._flat_weights up to date if you do self.weight = ...
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        super(RNNBase, self).__setattr__(attr, value)

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.
        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not torch.is_tensor(w):
                return
        # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # or the tensors in _flat_weights are of different dtypes

        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not torch.is_tensor(fw.data) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on self._flat_weights
            with torch.no_grad():
                torch._cudnn_rnn_flatten_weight(
                    self._flat_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(RNNBase, self)._apply(fn)

        # Resets _flat_weights
        # Note: be v. careful before removing this, as 3rd party device types
        # likely rely on this behavior to properly .to() modules like LSTM.
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        # Flattens params (on CUDA)
        self.flatten_parameters()

        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def check_input(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> None
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> Tuple[int, int, int]
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        # type: (Tensor, Tuple[int, int, int], str) -> None
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tensor, Optional[Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx, permutation):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        _impl = _rnn_impls[self.mode]
        if batch_sizes is None:
            result = _impl(input, hx, self._flat_weights, self.bias, self.num_layers,
                           self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _impl(input, batch_sizes, hx, self._flat_weights, self.bias,
                           self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(RNNBase, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']

        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                    self._flat_weights_names.extend(weights)
                else:
                    self._all_weights += [weights[:2]]
                    self._flat_weights_names.extend(weights[:2])
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def _replicate_for_data_parallel(self):
        replica = super(RNNBase, self)._replicate_for_data_parallel()
        # Need to copy these caches, otherwise the replica will share the same
        # flat weights list.
        replica._flat_weights = replica._flat_weights[:]
        replica._flat_weights_names = replica._flat_weights_names[:]
        return replica

class MixtureArgument(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, start=None, neuron=False, noise_type="bernoulli", p=0.5, train=False, inplace=False):
        if noise_type not in ["bernoulli", "uniform"]:
            raise ValueError("noise has to be bernoulli or uniform, "
                             "but got {}".format(noise_type))
        if p < 0 or p > 1:
            raise ValueError("drop probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p    
        ctx.train = train
        
        if ctx.p == 0 or not ctx.train:
            return input
        
        if start is None:
            start = cls._make_noise(input)
            start.fill_(0)
        start = start.to(input.device)

        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        
        #identity = cls._make_noise(input)
        #identity.fill_(1)
        #identity.expand_as(input)
        
        ctx.noise = cls._make_noise(input)
        if noise_type == "bernoulli" and not neuron:
            ctx.noise.bernoulli_(1 - ctx.p)
        if noise_type == "bernoulli" and neuron:
            if len(ctx.noise.size()) == 1:
                ctx.noise.bernoulli_(1 - ctx.p)
            else:
                #incoming sync code
                #ctx.noise = ctx.noise[:, :1].bernoulli_(1 - ctx.p)
                #ctx.noise = ctx.noise.repeat(1, input.size()[1])
                #outgoing sync code
                ctx.noise.bernoulli_(1 - ctx.p)
                ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        if noise_type == "uniform" and not neuron:
            ctx.noise.uniform_(0, 1)
        if noise_type == "uniform" and neuron:
            if len(ctx.noise.size()) == 1:
                ctx.noise.uniform_(0, 1)
            else:
                ctx.noise.uniform_(0, 1)
                ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)
        
        if ctx.p == 1:
            output = start
        else:
            #output = ((identity - ctx.noise) * start + ctx.noise * output - ctx.p * start) / (1 - ctx.p)
            output = ((1 - ctx.noise) * start + ctx.noise * output - ctx.p * start) / (1 - ctx.p)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output * ctx.noise, None, None, None, None, None, None
        else:
            return grad_output, None, None, None, None, None, None

def mixarg(input, start=None, neuron=False, noise_type="bernoulli", p=0.5, training=False, inplace=False):
    return MixtureArgument.apply(input, start, neuron, noise_type, p, training, inplace)

class MixLinear(torch.nn.Module):
    __constants__ = ['bias']
    # If start is None, then this mixes the current weight to 0 (a.k.a. dropconncect/dropout).
    # neuron determines mixconnect or mixout
    def __init__(self, in_features, out_features, bias=True, start=None, neuron=False, noise_type="bernoulli", mix_prob=0.5):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.start = start
        self.neuron = neuron
        self.noise_type = noise_type
        self.mix_prob = mix_prob
    
    def reset_parameters(self):
        self.weight.data.normal_(mean=0.0, std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()
            
    def forward(self, input):
        return torch.nn.functional.linear(input, mixarg(self.weight, self.start, self.neuron, self.noise_type, self.mix_prob, self.training), self.bias)

    def extra_repr(self):
        mix_or_drop = 'drop' if self.start is None else 'mix' 
        out_or_connect = 'out' if self.neuron is True else 'connect'
        return 'in_features={}, out_features={}, bias={}, {}_prob={}'.format(self.in_features, self.out_features, self.bias is not None,mix_or_drop+out_or_connect, self.mix_prob)

class MixLSTM(RNNBase):
    def __init__(self, *args, start=None, neuron=False, noise_type="bernoulli", mix_prob=0.5, **kwargs):
        super(MixLSTM, self).__init__('LSTM', *args, **kwargs)
        self.start = start
        self.neuron = neuron
        self.noise_type = noise_type
        self.mix_prob = mix_prob
        self.flatten_parameters()
        self.reset_parameters() 
         
    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tuple[Tensor, Tensor], Optional[Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')

    def permute_hidden(self, hx, permutation):
        # type: (Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def forward(self, input, hx=None):  # noqa: F811
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        self._flat_weights = [(lambda wn: mixarg(getattr(self, wn), self.start[wn], self.neuron, self.noise_type, self.mix_prob, self.training) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        self.flatten_parameters()
        if batch_sizes is None:
            result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)
    def extra_repr(self):
        mix_or_drop = 'drop' if self.start is None else 'mix' 
        out_or_connect = 'out' if self.neuron is True else 'connect'
        return 'input_size={}, hidden_size={}, bias={}, dropout={}, num_layers={}, bidirectional={}, {}_prob={}'.format(self.input_size, self.hidden_size, self.bias is not None, self.dropout, self.num_layers, self.bidirectional is not None,mix_or_drop+out_or_connect, self.mix_prob)
'''            
class MixLSTM(torch.nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, start=None, neuron=False, noise_type="bernoulli", mix_prob=0.5):
        super(MixLSTM, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        self.start = start
        self.neuron = neuron
        self.noise_type = noise_type
        self.mix_prob = mix_prob
        self._flat_weights_names = []
        self._all_weights = []
        num_directions = 2 if bidirectional else 1
        gate_size = 4 * hidden_size
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        self.flatten_parameters()
        self.reset_parameters()
    
    def forward(self, input, hx=None):  # noqa: F811
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)
        self._flat_weights = [(lambda wn: mixarg(getattr(self, wn), self.start[wn], self.neuron, self.noise_type, self.mix_prob, self.training) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        
        #self._flat_weights = [mixarg(wn, self.start[s], self.neuron, self.noise_type, self.mix_prob, self.training) if 'weight' in s else wn for s, wn in zip(self.start, self._flat_weights)]
        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)   

    def extra_repr(self):
        mix_or_drop = 'drop' if self.start is None else 'mix' 
        out_or_connect = 'out' if self.neuron is True else 'connect'
        return 'input_size={}, hidden_size={}, bias={}, dropout={}, num_layers={}, bidirectional={}, {}_prob={}'.format(self.input_size, self.hidden_size, self.bias is not None, self.dropout, self.num_layers, self.bidirectional is not None,mix_or_drop+out_or_connect, self.mix_prob)
'''