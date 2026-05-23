import random
from engine import Value

class Neuron:
    def __init__(self, inp):
        self.w = [Value(random.uniform(-1,1)) for _ in range(inp)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, inp, out):
        self.neurons = [Neuron(inp) for _ in range(out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, inp, outs):
        sz = [inp] + outs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        