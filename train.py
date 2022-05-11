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
parser.add_argument('--R1', action="store_true")  # Use R1 trick
parser.add_argument('--pretrain', action="store_true")  # Use pretrain model
# One side label smoothing trick
parser.add_argument('--real_label', type=float, default=1.0) 
parser.add_argument('--fake_label', type=float, default=0.0)
parser.add_argument('--epochs', required=True, type=int) # N° of epoch to perform
parser.add_argument('--early_stopping', type=int, default=-1)  # Apply early stopping on SSIM with a patience of `early_stopping`
parser.add_argument('--dataset', required=True)  # The dataset used ex: "data/Coco"
parser.add_argument('--version', required=True)  # Version of Coco ex: 2017
parser.add_argument('--load_generator') 
# Hyperparameters 
parser.add_argument('--L1_weight', type=float, default=100) # Lambda
parser.add_argument('--gamma_2', type=float, default=1) # Gamma
parser.add_argument('--folders_name', required=True) # save the figures/data and model in the file `folders_name`
parser.add_argument('--seed', type=int, default=42) 
parser.add_argument('--only_L1', action="store_true") 
args = parser.parse_args()

torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if __name__ == "__main__":
    print("\rLoading the dataset...", end="\r")
    dataset_train = CocoLab(args.dataset, splits=["train", "unlabeled"], version=args.version, size=256, train=True)
    train_loader = data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)

    dataset_test = CocoLab(args.dataset, splits="val", version=args.version, size=256, train=False)
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

        noise = True

    else:
        if args.only_L1:
            generator = UNet(1, 2, stochastic=False).to(device)

        else:
            generator = UNet(1, 2, stochastic=True).to(device)
        generator.apply(init_weights) # init weights with a gaussian distribution centered at 0, and std=0.02

    discriminator.apply(init_weights) # init weights with a gaussian distribution centered at 0, and std=0.02

    print("\rTraining !                    \n")

    trainer = GanTrain(generator, discriminator, test_loader, train_loader, reg_R1=args.R1, real_label=args.real_label, fake_label=args.fake_label, gamma_1=args.L1_weight, gan_weight= 0 if args.only_L1 else 1, gamma_2=args.gamma_2)
    trainer.train(args.epochs, models_path="saves/{}/saved_models/".format(args.folders_name), noise=noise, logs_path="saves/{}/logs/".format(args.folders_name), figures_path="saves/{}/figures/".format(args.folders_name), early_stopping=args.early_stopping)