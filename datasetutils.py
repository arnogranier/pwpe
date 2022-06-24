import torchvision 
import torchvision.transforms as T
import torch 

def mnist(batch_size=1):
    """Helper function load the mnist dataset.

    Args:
        batch_size (int, optional): batch size. Defaults to 1.

    Returns:
        torch.utils.data.DataLoader, torch.utils.data.DataLoader: train dataset loader, test dataset loader
    """
    
    flatten_mnist_batch = lambda x : torch.unsqueeze(torch.flatten(x), -1)
    one_hot_label = lambda x : torch.unsqueeze(torch.Tensor([1. if i==x else 0.
                                                      for i in range(10)]), -1)
    int_label = lambda x : torch.Tensor([x]).float()
    img_transform = T.Compose([T.ToTensor(), T.Lambda(flatten_mnist_batch)])
    label_transform = T.Compose([T.Lambda(one_hot_label)])
    mnist_train = torchvision.datasets.MNIST('~/mnist', transform=img_transform,
                                             target_transform=label_transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, 
                                            batch_size=batch_size, shuffle=True)
    mnist_test = torchvision.datasets.MNIST('~/mnist', train=False,
                            transform=img_transform, target_transform=int_label)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                              shuffle=True) 
    return train_loader, test_loader 