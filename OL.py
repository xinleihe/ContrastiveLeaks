import torch
import time
import numpy as np
import os
import argparse
from tqdm import tqdm
from models.resnet_simclr import ResNetSimCLR, LinearClassifier, CombineModel
from models.attack_model import MLP_OL
from utils.dataset_parser.dataset_loader import GetDataLoader
from utils.ol.attackTraining import attackTraining
torch.manual_seed(0)
torch.set_num_threads(1)
np.random.seed(0)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='UTKFace',
                        help='dataset')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='data_path')
    parser.add_argument('--mode', type=str, default='target',
                        help='control using target dataset or shadow dataset (for membership inference attack)')

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
                        choices=['SupCon', 'SimCLR', 'CE'], help='choose method')
    parser.add_argument('--projection_head_out_dim', type=int, default=256,
                        help='xxx')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
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

    parser.add_argument('--ratio', type=float, default=1.0,
                        help='how many data is used to train the adversarial classifier')
    # label
    parser.add_argument('--original_label', type=str, default='Gender')
    parser.add_argument("--aux_label", type=str, default='Race')
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

    aux_dataset_class_dict = {
        "UTKFace": 5,
        "CelebA": 4,
        "Place365": 365,
        "Place100": 100,
        "Place50": 50,
        "Place20": 20,
    }
    # print(dataset_class_dict)
    opt.n_class = dataset_class_dict[opt.dataset]
    opt.n_class_aux = aux_dataset_class_dict[opt.dataset]
    opt.encoder_dim = model_encoder_dim_dict[opt.model]

    return opt


def get_dataset_statistic(opt, targetTrainloader, targetTestloader, shadowTrainloader, shadowTestloader):
    target_train = [0 for i in range(opt.n_class)]
    target_test = [0 for i in range(opt.n_class)]
    shadow_train = [0 for i in range(opt.n_class)]
    shadow_test = [0 for i in range(opt.n_class)]
    for _, labels in tqdm(targetTrainloader):
        target_train[labels] += 1

    print(target_train)


def _load_encoder_model(opt):

    model = ResNetSimCLR(
        base_model=opt.model, encoder_dim=opt.encoder_dim, out_dim=opt.projection_head_out_dim)
    model = model.to(device)
    return model


def _load_classifier_model(opt):
    n_features = opt.encoder_dim
    n_classes = opt.n_class

    model = LinearClassifier(n_features, n_classes)
    model = model.to(device)
    return model


def _load_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


# def get_combine_model(checkpoint_path):

opt = parse_option()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

suffix = ""
if opt.dataset in opt.multi_label_dataset:
    suffix = "_%s" % (opt.original_label)


target_path = "./save/%s/model_%s_bs_%s_dataset_%s/combined_model_target%s.pth" % (
    opt.method, opt.model, opt.batch_size, opt.dataset, suffix)


# get aux dataset
s = GetDataLoader(opt)
target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = s.get_data_supervised()

target_encoder = _load_encoder_model(opt)
target_classifier = _load_classifier_model(opt)
target_combine_model = CombineModel(target_encoder, target_classifier)
target_combine_model = _load_model(target_combine_model, target_path)


attack_model = MLP_OL(dim_in=opt.encoder_dim, dim_out=opt.n_class_aux)
print("attack model dim_in dim_out: ", opt.encoder_dim, opt.n_class_aux)


attack = attackTraining(opt, target_train_loader, target_test_loader,
                        target_combine_model, attack_model, device)

start = time.process_time()

acc_train = 0
acc_test = 0
epoch_train = opt.epochs
train_acc, test_acc = attack.train(epoch_train)  # train 100 epoch
target_train_acc, target_test_acc = attack.original_performance
os.makedirs("log/model/exp_attack/", exist_ok=True)
with open("log/model/exp_attack/ol.txt", "a") as wf:
    wf.write("%s,%s,%s,%d,%s,%s,%.3f,%.3f,%.3f,%.3f\n" % (opt.method, opt.dataset, opt.model, epoch_train,
             opt.original_label, opt.aux_label, target_train_acc, target_test_acc, train_acc, test_acc))

print("Finish")
