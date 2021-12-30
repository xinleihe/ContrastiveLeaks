import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        # self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, encoder_dim, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.net_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False),
            "alexnet": models.alexnet(pretrained=False),
            "vgg16": models.vgg16(pretrained=False),
            "vgg11": models.vgg11(pretrained=False),
            "mobilenet": models.mobilenet_v2(pretrained=False),
            "cnn": SimpleCNN()
            # "inception": models.inception(pretrained=False),
            # "shufflenet": models.shufflenet_v2_x1_0(pretrained=False),
        }

        net = self._get_basemodel(base_model)
        # num_ftrs = net.fc.in_features
        if base_model in ["resnet18", "resnet50"]:
            net = nn.Sequential(*list(net.children())[:-1])
        elif base_model in ["vgg11", "vgg16"]:
            net.classifier = nn.Sequential(
                *list(net.classifier.children())[:-3])
        elif base_model in ["alexnet", "mobilenet"]:
            net.classifier = nn.Sequential(
                *list(net.classifier.children())[:-2])

        self.features = net

        # projection MLP
        if base_model in ["resnet18", "resnet50"]:
            self.l1 = nn.Linear(encoder_dim, encoder_dim)
            self.l2 = nn.Linear(encoder_dim, out_dim)
        else:
            self.l1 = nn.Linear(encoder_dim, out_dim)
            self.l2 = nn.Linear(out_dim, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.net_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise (
                "Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        # h = h.squeeze()
        h = torch.flatten(h, 1)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x


class LinearClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LinearClassifier, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder_layer1 = nn.Linear(
            in_features=input_dim, out_features=256)
        self.encoder_layer2 = nn.Linear(
            in_features=256, out_features=128
        )
        # self.encoder_layer3 = nn.Linear(
        #     in_features=128, out_features=64
        # # )
        # self.decoder_layer1 = nn.Linear(
        #     in_features=64, out_features=128
        # )
        self.decoder_layer2 = nn.Linear(
            in_features=128, out_features=256
        )
        self.decoder_layer3 = nn.Linear(
            in_features=256, out_features=input_dim
        )

    def forward(self, features):
        activation = F.relu(self.encoder_layer1(features))
        activation = F.relu(self.encoder_layer2(activation))
        # activation = F.relu(self.encoder_layer3(activation))

        # activation = self.decoder_layer1(activation)
        activation = self.decoder_layer2(activation)
        reconstructed = self.decoder_layer3(activation)

        # activation = torch.relu(activation)
        # code = self.encoder_output_layer(activation)
        # code = torch.relu(code)
        # activation = self.decoder_hidden_layer(code)
        # activation = torch.relu(activation)
        # activation = self.decoder_output_layer(activation)
        # reconstructed = torch.relu(activation)
        return reconstructed


class CombineModel(nn.Module):
    def __init__(self, encoder_model, classifier_model, return_type="output+embedding"):
        super(CombineModel, self).__init__()
        self.encoder = encoder_model
        self.classifier = classifier_model
        self.return_type = return_type

    def forward(self, x):
        h = self.encoder.features(x)
        # h = h.squeeze()
        h = torch.flatten(h, 1)
        out = self.classifier(h)
        if self.return_type == "output+embedding":
            return h, out
        elif self.return_type == "output":
            return out  # use only in the label only attack
        else:
            raise ValueError()


class CombineModelOlympus(nn.Module):
    def __init__(self, encoder_model, autoencoder_model, classifier_model, return_type="output+embedding"):
        super(CombineModelOlympus, self).__init__()
        self.encoder = encoder_model
        self.autoencoder = autoencoder_model
        self.classifier = classifier_model
        self.return_type = return_type

    def forward(self, x):
        h = self.encoder.features(x)
        # h = h.squeeze()
        h = torch.flatten(h, 1)
        h = self.autoencoder(h)
        out = self.classifier(h)
        if self.return_type == "output+embedding":
            return h, out
        elif self.return_type == "output":
            return out  # use only in the label only attack
        else:
            raise ValueError()
