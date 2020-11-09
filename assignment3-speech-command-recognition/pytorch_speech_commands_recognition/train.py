from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable


def train(loader, model, optimizer, epoch, cuda, log_interval, verbose=True):
    '''
    #############################################################################
    train the model, you can write this function partly refer to the "test" below
    Args:
        loader: torch dataloader
        model: model to train
        optimizer: torch optimizer
        epoch: number of epochs to train
        cuda: whether to use gpu
        log_interval: how many batches to wait before logging training status
        verbose: whether to print training log(such as epoch and loss)
    Return:
        the average loss of this epoch
    #############################################################################
    '''





def test(loader, model, cuda, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss
