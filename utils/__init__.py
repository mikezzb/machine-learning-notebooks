import torch
from torch import nn
import matplotlib.pyplot as plt


def __plot_learning_curve(train, test, label):
    plt.plot(train, label=f"Training {label}")
    plt.plot(test, label=f"Testing {label}")
    plt.title(f'Training and Testing {label}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()


def plot_learning_curves(acc_train, acc_test, loss_train, loss_test):
    __plot_learning_curve(acc_train, acc_test, "Accuracy")
    __plot_learning_curve(loss_train, loss_test, "Loss")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(dataloader, model, loss_fn, optimizer, scheduler=None, grad_clip=None, device="cuda"):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0

    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()

        # record loss
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if scheduler:
            scheduler.step()

    train_loss /= len(dataloader)
    correct /= size

    print(
        f" Train accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}")
    return correct, train_loss


def test(dataloader, model, loss_fn, device="cuda"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # record loss
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f" Test accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct, test_loss


def run(model, epochs, optimizer, loss_fn, train_dataloader, test_dataloader, scheduler=None, grad_clip=None):
    best_test_acc = 0
    acc_test_hist = []
    acc_train_hist = []
    loss_test_hist = []
    loss_train_hist = []
    for t in range(epochs):
        print('\n', "=" * 15, "Epoch", t + 1, "=" * 15)
        train_acc, train_loss = train(
            train_dataloader, model, loss_fn, optimizer, scheduler=scheduler, grad_clip=grad_clip)
        test_acc, test_loss = test(test_dataloader, model, loss_fn)
        acc_test_hist.append(test_acc)
        acc_train_hist.append(train_acc)
        loss_test_hist.append(test_loss)
        loss_train_hist.append(train_loss)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    plot_learning_curves(acc_train_hist, acc_test_hist,
                        loss_train_hist, loss_test_hist)
    print(f"Best test accuracy: {100*best_test_acc:0.2f}")
