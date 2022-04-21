import torch
import torch.utils.data as data
import argparse
from python.data.dataset import *
from python.models.generator import UNet
from python.models.discriminator import PatchGAN
from python.utils.images import *
from python.train.trainer import *
from python.models.utils import init_weights
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from fastai.vision.learner import create_body

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--R1', action="store_true") 
parser.add_argument('--pretrain', action="store_true") 
parser.add_argument('--real_label', type=float, default=1.0)
parser.add_argument('--fake_label', type=float, default=0.0)
parser.add_argument('--epochs', required=True, type=int) 
parser.add_argument('--dataset', required=True) 
parser.add_argument('--version', required=True) 
parser.add_argument('--load_generator')
parser.add_argument('--L1_weight', type=int, default=100)
parser.add_argument('--folders_name', required=True) 
parser.add_argument('--seed', type=int, default=42) 
args = parser.parse_args()

torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if __name__ == "__main__":
    print("\rLoading the dataset...", end="\r")
    dataset_train = CocoLab(args.dataset, split="train", version=args.version, size=256)
    train_loader = data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)

    dataset_test = CocoLab(args.dataset, split="val", version=args.version, size=256)
    test_loader = data.DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=4)

    print("\rSetup the networks...", end="\r")
        
    
    discriminator = PatchGAN(3).to(device)

    os.mkdir("saves/{}".format(args.folders_name))
    os.mkdir("saves/{}/figures".format(args.folders_name))
    os.mkdir("saves/{}/saved_models".format(args.folders_name))
    os.mkdir("saves/{}/logs".format(args.folders_name))

    noise = False

    if args.load_generator:
        generator = UNet(1, 2, stochastic=False).to(device)
        generator.load_state_dict(torch.load(args.load_generator, map_location=device))

    elif args.pretrain:
        resnet_body = create_body(resnet18, pretrained=True, n_in=2, cut=-2)
        generator = DynamicUnet(resnet_body, 2, (256, 256), y_range=(-1, 1)).to(device)
        for param in resnet_body.parameters():
            param.requires_grad = False

        noise = True

    else:
        generator = UNet(1, 2).to(device)
        generator.apply(init_weights) # init weights with a gaussian distribution centered at 0, and std=0.02

    discriminator.apply(init_weights) # init weights with a gaussian distribution centered at 0, and std=0.02

    print("\rTraining !                    \n")

    trainer = GanTrain(generator, discriminator, test_loader, train_loader, reg_R1=args.R1, real_label=args.real_label, fake_label=args.fake_label, gamma_1=args.L1_weight)
    trainer.train(args.epochs, models_path="saves/{}/saved_models/".format(args.folders_name), noise=noise, logs_path="saves/{}/logs/".format(args.folders_name), figures_path="saves/{}/figures/".format(args.folders_name))

    # TODO
    # trainer.make_plot("figures/{}/".format(args.folders_name))