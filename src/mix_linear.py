##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Cheolhyoung Lee
## Department of Mathematical Sciences, KAIST
## Email: bloodwass@kaist.ac.kr
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch.autograd.function import InplaceFunction
from torch.nn import Parameter

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