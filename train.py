import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import sys,argparse

class FashionMnistData:
    def __init__(self) -> None:
        self._load_data()
        self._init_dataloader()

    def _load_data(self):
        # Download training data from open datasets.
        self.training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        self.test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

    def _init_dataloader(self, batch_size=64):
        # Create data loaders.
        self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size)

        for X, y in self.test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class TrainApp:
    def __init__(self, sys_argv=None) -> None:
        self._init_arguments(sys_argv)
        self._init_device()
        self._init_model()
        self._init_lossfn_optimizer()
        self.fashion_mnist = FashionMnistData()
        self.train_dataloader = self.fashion_mnist.train_dataloader
        self.test_dataloader = self.fashion_mnist.test_dataloader
        self.test_data = self.fashion_mnist.test_data

    def _init_arguments(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=64,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        parser.add_argument('--learning-rate',
            help='Learning rate',
            default=1e-3,
            type=float,
        )
        self.cli_args = parser.parse_args(sys_argv)



    def _init_device(self):
        # Get cpu, gpu or mps device for training.
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

    def _init_model(self):
        self.model = NeuralNetwork().to(self.device)
        print(self.model)

    def _init_lossfn_optimizer(self, lr=1e-3):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def _train(self):
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _test(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def do_training(self, epochs=2):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self._train()
            self._test()
        print("Done!")

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

    def load_model(self):
        self.model = NeuralNetwork().to(self.device)
        self.model.load_state_dict(torch.load("model.pth"))

    def do_serving(self):
        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
        self.model.eval()
        x, y = self.test_data[0][0], self.test_data[0][1]
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

if __name__ == '__main__':
    app = TrainApp()
    app.do_training()