from layers import LinearVariational, Latent_sample
from metric import kl_gaussian, eval_single_kl
import torch
from torch import nn
import torch.nn.functional as F

class Bayes_classifier(nn.Module):
    def __init__(self, dims):
        super(Bayes_classifier, self).__init__()
        [in_dim, out_dim] = dims
        self.var = LinearVariational(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_divergence = 0


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        z, mu, p = self.var(x)
        self.kl_divergence = kl_gaussian(mu, p, z.shape[-1])
        #out = self.softmax(z)
        out = self.log_softmax(z)
        return out

class MLP(nn.Module):
    def __init__(self, dims):
        # deterministic >1 layer classifier
        super(MLP, self).__init__()
        [in_dim, h_dim, out_dim] = dims
        neurons = [in_dim, *h_dim]

        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.out = nn.Linear(h_dim[-1], out_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = F.softmax(self.out(x), dim=-1)
        return x


class Bayes_MLP(nn.Module):
    def __init__(self, dims):
        # bayes last layer
        super(Bayes_MLP, self).__init__()
        [in_dim, h_dim, out_dim] = dims

        if isinstance(h_dim, list):
            neurons = [in_dim, *h_dim]
            linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)
            self.var = Bayes_classifier([h_dim[-1], out_dim])
            self.h_num = 1 # more than one hidden layer
        else:
            self.hidden = nn.Linear(in_dim, h_dim)
            self.var = Bayes_classifier([h_dim, out_dim])
            self.h_num = 0


        self.kl_divergence = 0


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        if (self.h_num == 0):
            x = F.relu(self.hidden(x))
        else:
            for layer in self.hidden:
                x = F.relu(layer(x))
        x = self.var(x)
        self.kl_divergence = self.var.kl_divergence
        return x


class Bayes_CNN(nn.Module):
    def __init__(self):
        super(Bayes_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 20)
        self.var = Bayes_classifier([20, 10])

    def forward(self, x):
        x = x.view(-1,1,28,28)
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = out.view(-1, 320)

        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))

        out = self.var(out)
        self.kl_divergence = self.var.kl_divergence

        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = out.view(-1, 320)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return F.softmax(out, dim=1)

class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.sample = Latent_sample(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            # x = F.relu(layer(x))
            x = F.softplus(layer(x))
        return self.sample(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class Decoder(nn.Module):
    def __init__(self, dims):
        # generate x' with input z
        super(Decoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims
        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            # x = F.relu(layer(x))
            x = F.softplus(layer(x))
        return self.output_activation(self.reconstruction(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, dims):
        # dims: x, z, [hidden dimensions]
        super(VariationalAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0


    def forward(self, x, y=None):
        # input x generates z and new_x, return new_x
        z, z_mu, z_p = self.encoder(x)
        self.kl_divergence = eval_single_kl(z_mu, z_p, z)
        x_mu = self.decoder(z)
        return x_mu

    def sample(self, z):
        # input z~N(0,I) generates x_new
        return self.decoder(z)


class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims):
        # M2 from Kingma 2014) in PyTorch.
        # dims: x_dim, y_dim, z_dim, [h_dim]
        [x_dim, self.y_dim, z_dim, h_dim] = dims

        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        #self.classifier = Bayes_CNN()
        self.classifier = CNN()
        #self.classifier = MLP([x_dim, [h_dim[0]], self.y_dim])
        # self.classifier = Perceptron([x_dim, self.y_dim])
        # self.classifier = Bayes_classifier(x_dim, self.y_dim)

    def forward(self, x, y):
        # y is one-hot encoded
        z, z_mu, z_p = self.encoder(torch.cat([x, y], dim=1))
        #self.kl_divergence = kl_gaussian(z_mu, z_p, len(x))
        self.kl_divergence = eval_single_kl(z_mu, z_p, z)
        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))
        return x_mu

    def classify(self, x):
        #logits = self.classifier(x).mean(axis = -1)
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        # generate x_new with inputs: z~N(0,1), y is desired one hot encoding class
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


