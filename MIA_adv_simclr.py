import torch
from models.resnet_simclr import ResNetSimCLR, LinearClassifier, CombineModel
from models.attack_model import MLP_CE
from utils.dataset_parser.dataset_loader import GetDataLoader
from utils.mia.attackTraining import attackTraining
from utils.mia.metric_based_attack import AttackTrainingMetric
from utils.mia.label_only_attack import AttackLabelOnly
import numpy as np
import time
import argparse
import os
torch.manual_seed(0)
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
    parser.add_argument('--learning_rate', type=float, default=0.05,
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
                        choices=['SupCon', 'SimCLR', "CE"], help='choose method')
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
    # adv training setting:
    parser.add_argument('--adv_training', type=str, default='yes',
                        choices=["yes", "no"], help='control whether using adv training')
    parser.add_argument('--adv_factor', type=int, default=5,
                        help='parameter for adv training')
    parser.add_argument('--adv_image', type=str,
                        default="augmented", help='original or augmented')
    parser.add_argument("--adv_location", type=str,
                        default="embedding", help='embedding or projection')
    # label
    parser.add_argument('--original_label', type=str, default='Gender')
    parser.add_argument("--aux_label", type=str, default='Race')
    parser.add_argument('--single_label_dataset', type=list, default=["CIFAR10", "CIFAR100", "STL10"],
                        help="single_label_dataset")
    parser.add_argument('--multi_label_dataset', type=list, default=["UTKFace", "CelebA", "Place365", "Place100", "Place50", "Place20"],
                        help="multi_label_dataset")
    parser.add_argument('--mia_type', type=str, default="nn-based",
                        help="nn-based, lebel-only, metric-based")
    parser.add_argument('--select_posteriors', type=int, default=-1,
                        help='how many posteriors we select, if -1, we remains the original setting')

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


def write_res(opt, wf, attack_name, res):
    line = "%s,%s,%s,%s,%s," % (
        opt.dataset, opt.model, opt.method, opt.original_label, opt.aux_label)

    line += "%s," % attack_name

    line += ",".join(["%.3f" % (row) for row in res])
    line += "\n"
    wf.write(line)


opt = parse_option()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

suffix = ""
if opt.dataset in opt.multi_label_dataset:
    suffix = "_%s" % (opt.original_label)


save_path = "./save/%s/model_%s_bs_%s_dataset_%s/" % (
    opt.method, opt.model, opt.batch_size, opt.dataset)
target_model = 'combined_model_%s_with_adv_factor_%s_advimage_%s_adv_location_%s_%s_%s.pth' \
    % ("target", opt.adv_factor, opt.adv_image, opt.adv_location, opt.original_label, opt.aux_label)
shadow_model = 'combined_model_%s_with_adv_factor_%s_advimage_%s_adv_location_%s_%s_%s.pth' \
    % ("shadow", opt.adv_factor, opt.adv_image, opt.adv_location, opt.original_label, opt.aux_label)

target_path = save_path + target_model
shadow_path = save_path + shadow_model


s = GetDataLoader(opt)
target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = s.get_data_supervised()
target_encoder = _load_encoder_model(opt)
target_classifier = _load_classifier_model(opt)

if opt.mia_type == "label-only":
    return_type = "output"
else:
    return_type = "output+embedding"

target_combine_model = CombineModel(
    target_encoder, target_classifier, return_type=return_type)
target_combine_model = _load_model(target_combine_model, target_path)


shadow_encoder = _load_encoder_model(opt)
shadow_classifier = _load_classifier_model(opt)
shadow_combine_model = CombineModel(
    shadow_encoder, shadow_classifier, return_type=return_type)
shadow_combine_model = _load_model(shadow_combine_model, shadow_path)

attack_model = MLP_CE()

if opt.mia_type == "nn-based":
    attack = attackTraining(opt, target_train_loader, target_test_loader,
                            shadow_train_loader, shadow_test_loader, target_combine_model, shadow_combine_model, attack_model, device)

    attack.parse_dataset()

    acc_train = 0
    acc_test = 0
    epoch_train = opt.epochs
    train_acc, test_acc = attack.train(epoch_train)  # train 100 epoch
    target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance

    with open("log/model/exp_attack/mia_update_adv_simclr.txt", "a") as wf:
        res = [opt.adv_factor, epoch_train, target_train_acc, target_test_acc,
               shadow_train_acc, shadow_test_acc, train_acc, test_acc]
        write_res(opt, wf, "NN-based", res)


elif opt.mia_type == "metric-based":
    attack = AttackTrainingMetric(opt, target_train_loader, target_test_loader,
                                  shadow_train_loader, shadow_test_loader, target_combine_model, shadow_combine_model, attack_model, device)

    attack.parse_dataset()

    acc_train = 0
    acc_test = 0
    epoch_train = opt.epochs

    train_tuple0, test_tuple0, test_results0, train_tuple1, test_tuple1, test_results1, train_tuple2, test_tuple2, test_results2, train_tuple3, test_tuple3, test_results3 = attack.train()
    target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance
    with open("log/model/exp_attack/mia_update_adv_simclr.txt", "a") as wf:
        res0 = [opt.adv_factor, epoch_train, target_train_acc, target_test_acc,
                shadow_train_acc, shadow_test_acc, train_tuple0[0], test_tuple0[0]]
        res1 = [opt.adv_factor, epoch_train, target_train_acc, target_test_acc,
                shadow_train_acc, shadow_test_acc, train_tuple1[0], test_tuple1[0]]
        res2 = [opt.adv_factor, epoch_train, target_train_acc, target_test_acc,
                shadow_train_acc, shadow_test_acc, train_tuple2[0], test_tuple2[0]]
        res3 = [opt.adv_factor, epoch_train, target_train_acc, target_test_acc,
                shadow_train_acc, shadow_test_acc, train_tuple3[0], test_tuple3[0]]
        write_res(opt, wf, "Metric-corr", res0)
        write_res(opt, wf, "Metric-conf", res1)
        write_res(opt, wf, "Metric-ent", res2)
        write_res(opt, wf, "Metric-ment", res3)


elif opt.mia_type == "label-only":
    attack = AttackLabelOnly(opt, target_train_loader, target_test_loader,
                             shadow_train_loader, shadow_test_loader, target_combine_model, shadow_combine_model, attack_model, device)

    acc_train = 0
    acc_test = 0
    epoch_train = opt.epochs

    attack.searchThreshold(num_samples=-1)
    test_tuple = attack.test(num_samples=-1)
    target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc, threshold = attack.original_performance
    res = [opt.adv_factor, epoch_train, target_train_acc, target_test_acc, shadow_train_acc,
           shadow_test_acc, threshold, test_tuple[0]]
    os.makedirs("log/model/exp_attack/", exist_ok=True)
    with open("log/model/exp_attack/mia_update_adv_simclr.txt", "a") as wf:
        write_res(opt, wf, "Label-only", res)
print("Finish")
