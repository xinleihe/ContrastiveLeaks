import torch
import numpy as np
import os
from models.resnet_simclr import ResNetSimCLR, LinearClassifier, CombineModel
import argparse
from utils.dataset_parser.dataset_loader import GetDataLoader
torch.manual_seed(0)
np.random.seed(0)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # adv training setting:
    parser.add_argument('--adv_training', type=str, default='yes',
                        choices=["yes", "no"], help='control whether using adv training')
    parser.add_argument('--adv_factor', type=int, default=5,
                        help='parameter for adv training')
    parser.add_argument('--adv_image', type=str,
                        default="augmented", help='original or augmented')
    parser.add_argument("--adv_location", type=str,
                        default="embedding", help='embedding or projection')
    # specific task
    parser.add_argument('--original_label', type=str, default='Gender')
    parser.add_argument("--aux_label", type=str, default='Race')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=10e-6,
                        help='temperature for loss function')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--task', type=str, default='mia',
                        help='specify the attack task, mia or ol')
    parser.add_argument('--dataset', type=str, default='UTKFace',
                        help='dataset')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='data_path')
    # Note: mode is set to ol when training overlearning model just to control the final save name
    parser.add_argument('--mode', type=str, default='target',
                        help='control using target dataset or shadow dataset (for membership inference attack)')
    # parser.add_argument('--n_class', type=int, default=100,
    #                     help='number of class')

    parser.add_argument('--mean', type=str,
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str,
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str,
                        default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32,
                        help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--projection_head_out_dim', type=int, default=256,
                        help='number of training epochs')

    # temperature
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument("--fp16_precision", type=bool, default=False)
    parser.add_argument('--log_every_n_steps', type=int,
                        default=50, help='log_every_n_steps')
    parser.add_argument('--save_every_n_epochs', type=int,
                        default=10, help='save_every_n_epochs')
    parser.add_argument('--single_label_dataset', type=list, default=["CIFAR10", "CIFAR100", "STL10"],
                        help="single_label_dataset")
    parser.add_argument('--multi_label_dataset', type=list, default=["UTKFace", "CelebA", "Place365", "Place100", "Place50", "Place20"],
                        help="multi_label_dataset")

    opt = parser.parse_args()

    model_encoder_dim_dict = {
        "resnet18": 512,
        "resnet50": 2048,
        "alexnet": 4096,
        "vgg16": 4096,
        "vgg11": 4096,
        "mobilenet": 1280,
        "cnn": 512,
    }
    dataset_class_dict = {
        "STL10": 10,
        "CIFAR10": 10,
        "CIFAR100": 100,
        "UTKFace": 2,
        "CelebA": 2,
        "Place365": 2,
        "Place100": 2,
        "Place50": 2,
        "Place20": 2,
    }
    opt.n_class = dataset_class_dict[opt.dataset]
    opt.encoder_dim = model_encoder_dim_dict[opt.model]

    return opt


def _load_encoder_model(opt, pretrained=""):
    model = ResNetSimCLR(
        base_model=opt.model, encoder_dim=opt.encoder_dim, out_dim=opt.projection_head_out_dim)

    checkpoints_folder = "./save/SimCLR/model_%s_bs_%d_dataset_%s/checkpoints/" % (
        opt.model, opt.batch_size, opt.dataset)
    model_name = 'model_%s_with_adv_factor_%s_advimage_%s_adv_location_%s_%s_%s.pth' \
        % (opt.mode, opt.adv_factor, opt.adv_image, opt.adv_location, opt.original_label, opt.aux_label)
    state_dict = torch.load(os.path.join(
        checkpoints_folder, model_name), map_location=torch.device('cpu'))
    print("load model from", checkpoints_folder + model_name)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    return model


def _load_classifier_model(opt):
    n_features = opt.encoder_dim
    n_classes = opt.n_class
    model = LinearClassifier(n_features, n_classes)
    model = model.to(device)
    return model


class SimCLR_linear_model_evaluator(object):
    def __init__(self, encoder, classifier, opt):
        self.encoder = encoder
        self.classifier = classifier
        self.encoder.eval()
        self.total_model = CombineModel(self.encoder, self.classifier)
        self.total_model = self.total_model.to(device)
        self.opt = opt
        self.save_path = "./save/SimCLR/model_%s_bs_%s_dataset_%s/" % (self.opt.model,
                                                                       self.opt.batch_size, self.opt.dataset)

    @staticmethod
    def _sample_weight_decay():
        # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
        weight_decay = np.logspace(-6, 5, num=45, base=10.0)
        weight_decay = np.random.choice(weight_decay)
        print("Sampled weight decay:", weight_decay)
        return weight_decay

    def get_label(self, label):
        if self.opt.dataset in self.opt.single_label_dataset:
            return label
        elif self.opt.dataset in self.opt.multi_label_dataset:
            return label[self.opt.original_label]
        else:
            raise ValueError("dataset not found")

    def eval(self, test_loader):
        correct = 0
        total = 0

        with torch.no_grad():
            self.total_model.classifier.eval()
            # cnt = 0
            for img, label in test_loader:
                # cnt +=1
                img, label = img.to(device), self.get_label(label).to(device)
                _, logits = self.total_model(img)

                predicted = torch.argmax(logits, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            final_acc = 100 * correct / total
            self.total_model.classifier.train()
            return final_acc

    def train(self, train_loader, test_loader):

        weight_decay = 1e-4
        #! only need to optimize the parameters of classifier part
        optimizer = torch.optim.Adam(
            self.total_model.classifier.parameters(), 1e-2, weight_decay=weight_decay)

        criterion = torch.nn.CrossEntropyLoss()

        best_accuracy = 0
        self.total_model.classifier.train()
        for e in range(1, self.opt.epochs + 1):
            batch_n = 0
            for img, label in train_loader:
                self.total_model.classifier.zero_grad()
                batch_n += 1

                img, label = img.to(device), self.get_label(label).to(device)

                _, logits = self.total_model(img)

                loss = criterion(logits, label)
                if batch_n % 10 == 0:
                    print("[Epoch %d][%d/%d] loss:%.3f" %
                          (e, batch_n, len(train_loader), loss))

                loss.backward()
                optimizer.step()
            if e % 50 == 0:
                train_acc = self.eval(train_loader)
                epoch_acc = self.eval(test_loader)
                print("epoch:%d, train acc:%.3f, test acc:%.3f" %
                      (e, train_acc, epoch_acc))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.opt.dataset in self.opt.single_label_dataset:
            torch.save(self.total_model.state_dict(), os.path.join(
                self.save_path + 'combined_model_%s.pth' % (self.opt.mode)))
        elif self.opt.dataset in self.opt.multi_label_dataset:
            model_name = 'combined_model_%s_with_adv_factor_%s_advimage_%s_adv_location_%s_%s_%s.pth' \
                % (opt.mode, opt.adv_factor, opt.adv_image, opt.adv_location, opt.original_label, opt.aux_label)
            torch.save(self.total_model.state_dict(),
                       os.path.join(self.save_path + model_name))

        print("--------------")
        print("Done training")
        print("Best accuracy:", best_accuracy)


opt = parse_option()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
torch.set_num_threads(1)
dataset = GetDataLoader(opt)
target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = dataset.get_data_supervised()
if opt.mode == "target":
    train_loader, test_loader = target_train_loader, target_test_loader,
elif opt.mode == "shadow":
    train_loader, test_loader = shadow_train_loader, shadow_test_loader

encoder_model = _load_encoder_model(opt)
classifier_model = _load_classifier_model(opt)
total_evaluator = SimCLR_linear_model_evaluator(
    encoder=encoder_model, classifier=classifier_model, opt=opt)

total_evaluator.train(train_loader, test_loader)
with open("log/adv_simclr_result.txt", "a") as wf:
    wf.write("finish adv SimCLR linear training dataset: %s, model:%s, mode: %s\n" % (
        opt.dataset, opt.model, opt.mode))

print("Finish")
