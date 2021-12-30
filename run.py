from simclr_adv_update import SimCLR_adv
from simclr_update import SimCLR
from utils.dataset_parser.dataset_loader import GetDataLoader
import argparse
import torch
torch.set_num_threads(1)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--n_views', type=int, default=2,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # optimization
    # parser.add_argument('--learning_rate', type=float, default=3e-4,
    #                     help='learning rate')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='temperature for loss function')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--task', type=str, default='mia',
                        help='specify the attack task, mia or ol')
    parser.add_argument('--dataset', type=str, default='Place365',
                        help='dataset')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='data_path')
    # Note: mode is set to ol when training overlearning model just to control the final save name
    parser.add_argument('--mode', type=str, default='target',
                        help='control using target dataset or shadow dataset (for membership inference attack)')
    # parser.add_argument('--n_class', type=int, default=100,
    #                     help='number of class')
    # adv training setting:
    parser.add_argument('--adv_training', type=str, default='no',
                        choices=["yes", "no"], help='control whether using adv training')
    parser.add_argument('--adv_factor', type=int, default=10,
                        help='parameter for adv training')
    parser.add_argument('--adv_image', type=str,
                        default="augmented", help='original or augmented')
    parser.add_argument("--adv_location", type=str,
                        default="embedding", help='embedding or projection')

    parser.add_argument('--original_label', type=str, default='Gender')
    parser.add_argument("--aux_label", type=str, default='Race')

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
    parser.add_argument('--pretrain', type=str, default="no",
                        help='if yes, use STL10 unlabeled dataset')

    # other setting
    # parser.add_argument('--cosine', action='store_true',
    #                     help='using cosine annealing')
    # parser.add_argument('--syncBN', action='store_true',
    #                     help='using synchronized batch normalization')
    # parser.add_argument('--warm', action='store_true',
    #                     help='warm-up for large batch training')
    # parser.add_argument('--trial', type=str, default='0',
    #                     help='id for recording multiple runs')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--log_every_n_steps', type=int,
                        default=50, help='log_every_n_steps')
    parser.add_argument('--save_every_n_epochs', type=int,
                        default=5, help='save_every_n_epochs')
    parser.add_argument('--fine_tune_from', type=str, default="",
                        help='fine_tune_from')

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
    dataset_aux_class_dict = {
        "CIFAR10": 10,
        "STL10": 10,
        "CIFAR100": 100,
        "UTKFace": 5,
        "CelebA": 4,
        "Place365": 365,
        "Place100": 100,
        "Place50": 50,
        "Place20": 20,
    }
    opt.n_class = dataset_class_dict[opt.dataset]
    opt.aux_n_class = dataset_aux_class_dict[opt.dataset]
    opt.encoder_dim = model_encoder_dim_dict[opt.model]

    return opt


def main():

    opt = parse_option()
    print("exp setting (SimCLR), dataset:%s \t model:%s \t mode: %s" %
          (opt.dataset, opt.model, opt.mode))

    dataset = GetDataLoader(opt)
    if opt.adv_training == "no":
        simclr = SimCLR(dataset, opt)
        simclr.train()
    elif opt.adv_training == "yes":
        simclr = SimCLR_adv(dataset, opt)
        simclr.train()

    else:
        raise ValueError("wrong adv_training parameter!!!")
    with open("log/result/SimCLR_result.txt", "a") as wf:
        wf.write("finish SimCLR training dataset: %s, model:%s, mode: %s, adv_training: %s\n" % (
            opt.dataset, opt.model, opt.mode, opt.adv_training))

    print("Finish")


if __name__ == "__main__":
    main()
