import torch

def relu(x):
    return torch.maximum(x, torch.Tensor([0,]).cuda())

def drelu(x):
    return (x>=torch.Tensor([0,]).cuda()).float()

class PCLayer:
    """Basic predictive coding layer.

    Args:
        n (int): number of neurons in this layer.
        m (int, optional): number of neurons in the layer this layer predicts. Defaults to None.
        f (function, optional): activation function. Defaults to relu.
        df (function, optional): derivate of the activation function. Defaults to drelu.
        w_init_std (float, optional): standard deviation of the normal distribution used to initialize weights. Defaults to 0.001.
        
    Attributes:
        size (int): number of neurons in this layer.
        u (1d float tensor): representations / membrane potentials.
        e (1d float tensor): prediction errors.
        w (2d float tensor): prediction weight matrix.
        f (function): activation function. 
        df (function): derivate of the activation function.
    """
        
    def __init__(self, n, m=None, f=relu, df=drelu, w_init_std=0.001):
        self.size = n
        if m is not None:
            self.w = w_init_std * torch.randn(m, n).cuda()
        self.f = relu 
        self.df = drelu
        
    def set_repr(self, u):
        """set representations / membrane potentials.

        Args:
            u (1d float tensor): the value of representations to be set.
        """
        self.u = u
        self.e = torch.zeros(self.u.size()).cuda()
        
    def predict(self):
        """compute prediction.

        Returns:
            1d float tensor: the prediction.
        """
        return self.w @ self.f(self.u)
    
    @property
    def prediction(self):
        """prediction property.

        Returns:
            1d float tensor: the prediction.
        """
        return self.predict()
    
    def set_err(self, p):
        """compute errors.

        Args:
            p (1d float tensor): prediction of self activity from other layer
        """
        self.e = self.u - p
        
    def infer_step(self, e, ir):
        """inference step.

        Args:
            e (1d float tensor): errors from layer this layer sends prediction to
            ir (float): inference rate (euler scheme dt)
        """
        self.u += ir * (-self.e + self.df(self.u) * (self.w.mT @ e))
        
    def learn_step(self, e, lr):
        """prediction weight learning.

        Args:
            e (1d float tensor): errors from layer this layer sends prediction to
            lr (float): learning rate (euler scheme dt)
        """
        self.w += lr * torch.sum(torch.einsum('nx,ny->nxy', torch.squeeze(e), 
                                            torch.squeeze(self.f(self.u))), 0)


class DPCNet:
    """Discriminative predictive coding network.

    Args:
        args(list of int): architecture.
        ir (float, optional): inference rate (euler scheme dt). Defaults to 0.1.
        T (int, optional): number of inference step in the inference loop. Defaults to 20.
    
    Attributes:
        layers(list of PCLayer): the predictive coding network as a list of layers.
        ir (float, optional): inference rate (euler scheme dt).
        T (int, optional): number of inference step in the inference loop.
    """
        
    def __init__(self, *args, ir=0.1, T=20):
        
        self.layers = [PCLayer(s1, s2) for (s1, s2) in 
                       zip(args[:-1], args[1:])] + [PCLayer(args[-1]),]
        self.ir = ir 
        self.T = T
        
    def infer(self, data, target=None):
        """inference.
        If target is None, return the forward sweep prediction, else, initialize
        activity in hidden and top layer to predictions and perform T steps of 
        inference with inference rate ir with clamped top level to target.

        Args:
            data (2d or 3d float tensor): data to clamp bottom layer to.
            target (2d or 3d float tensor, optional): target to clamp top level to. Defaults to None.

        Returns:
            2d or 3d float tensor: if target is None, return the top level activity after the forward sweep.
        """
        N = len(self.layers)
        self.layers[0].set_repr(data)
        for i in range(1, N-1):
            self.layers[i].set_repr(self.layers[i-1].prediction)
        if target is not None:
            self.layers[-1].set_repr(target)
        else:
            self.layers[-1].set_repr(self.layers[-2].prediction)
            return self.layers[-1].u
        
        for _ in range(self.T):
            for i in range(1, N):
                self.layers[i].set_err(self.layers[i-1].prediction)
            for i in range(1, N-1): 
                self.layers[i].infer_step(self.layers[i+1].e, self.ir)

    def accuracy(self, data_loader):
        """compute accuracy of the network.

        Args:
            data_loader (torch.utils.data.DataLoader): torch dataset loader

        Returns:
            float: accuracy
        """
        ncorrect = 0
        for image, target in data_loader:
            l = self.infer(image.cuda()).to('cpu')
            ncorrect += torch.sum(torch.argmax(l, axis=1) == target)
        return ncorrect/len(data_loader.dataset)
        
        
class DPCHebbian:
    """Hebbian learner for discriminative predictive coding networks.

    Args:
        net (DPCNet): the networks
        lr (float, optional): learning rate (euler scheme dt). Defaults to 0.02.
            
    Attributes:
        net (DPCNet): the networks
        lr (float, optional): learning rate (euler scheme dt). Defaults to 0.02.
    """
    def __init__(self, net, lr=0.02):
        self.net = net
        self.lr = lr 
        
    def step(self):
        """perform a step of learning."""
        N = len(self.net.layers)
        for i in range(1, N):
            self.net.layers[i].set_err(self.net.layers[i-1].prediction)
            self.net.layers[i-1].learn_step(self.net.layers[i].e, self.lr)
         
            
class GPCNet:
    """Generative predictive coding network.

    Args:
        args(list of int): architecture.
        ir (float, optional): inference rate (euler scheme dt). Defaults to 0.1.
        T (int, optional): number of inference step in the inference loop. Defaults to 20.
    
    Attributes:
        layers(list of PCLayer): the predictive coding network as a list of layers.
        ir (float, optional): inference rate (euler scheme dt).
        T (int, optional): number of inference step in the inference loop.
    """
    def __init__(self, *args, ir=0.1, T=20):
        self.layers =  [PCLayer(args[0]),] + [PCLayer(s2, s1) for (s1, s2) in 
                                                zip(args[:-1], args[1:])]
        self.ir = ir 
        self.T = T
        
    def infer(self, target, data=None):
        """inference.
        If target is None, return the backward sweep prediction, else, initialize
        activity in hidden and top layer to predictions and perform T steps of 
        inference with inference rate ir with clamped top level to target.

        Args:
            data (2d or 3d float tensor): data to clamp bottom layer to.
            target (2d or 3d float tensor, optional): target to clamp top level to. Defaults to None.

        Returns:
            2d or 3d float tensor: if target is None, return the top level activity after the forward sweep.
        """
        N = len(self.layers)
        self.layers[-1].set_repr(target)
        for i in reversed(range(1, N-1)):
            self.layers[i].set_repr(self.layers[i+1].prediction)
        if data is not None:
            self.layers[0].set_repr(data)
        else:
            self.layers[0].set_repr(self.layers[1].prediction)
            return self.layers[0].u
        
        for _ in range(self.T):
            for i in range(N-1):
                self.layers[i].set_err(self.layers[i+1].prediction)
            for i in range(1, N-1): 
                self.layers[i].infer_step(self.layers[i-1].e, self.ir)
        
        
class GPCHebbian:
    """Hebbian learner for discriminative predictive coding networks.

    Args:
        net (DPCNet): the networks
        lr (float, optional): learning rate (euler scheme dt). Defaults to 0.02.
            
    Attributes:
        net (DPCNet): the networks
        lr (float, optional): learning rate (euler scheme dt). Defaults to 0.02.
    """
    def __init__(self, net, lr=0.02):
        self.net = net
        self.lr = lr 
        
    def step(self):
        """perform a step of learning."""
        N = len(self.net.layers)
        for i in range(N-1):
            self.net.layers[i].set_err(self.net.layers[i+1].prediction)
            self.net.layers[i+1].learn_step(self.net.layers[i].e, self.lr)