import torch
import torch.utils.data as data
import argparse
from python.data.dataset import *
from python.models.generator import UNet
from python.models.discriminator import PatchGAN
from python.utils.images import *
from python.train.trainer import *
from python.models.utils import init_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--R1', action="store_true") 
parser.add_argument('--pretrain', action="store_true") 
parser.add_argument('--epochs', required=True, type=int) 
parser.add_argument('--dataset', required=True) 
parser.add_argument('--version', required=True) 
parser.add_argument('--generator', default="generator") 
parser.add_argument('--discriminator', default="discriminator") 
parser.add_argument('--load_generator')
parser.add_argument('--plot', default="plot") 
args = parser.parse_args()

if __name__ == "__main__":
    print("\rLoading the dataset...", end="\r")
    dataset_train = CocoLab(args.dataset, version=args.version, size=256, train=True)
    train_loader = data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)

    dataset_test = CocoLab(args.dataset, version=args.version, size=256, train=False)
    test_loader = data.DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=4)

    print("\rSetup the networks...", end="\r")
        
    
    discriminator = PatchGAN(3).to(device)

    if args.load_generator:
        generator = UNet(1, 2, stochastic=False).to(device)
        generator.load_state_dict(torch.load(args.load_generator, map_location=device))

    else:
        generator = UNet(1, 2).to(device)
        generator.apply(init_weights) # init weights with a gaussian distribution centered at 0, and std=0.02

    discriminator.apply(init_weights) # init weights with a gaussian distribution centered at 0, and std=0.02


    print("\rTraining !                    \n")
    if args.pretrain:
        trainer = Pretrain(generator, test_loader, train_loader)
        trainer.train(args.epochs, generator_file=args.generator, file_name_plot=args.plot)

    else:
        trainer = GanTrain(generator, discriminator, test_loader, train_loader, reg_R1=args.R1)
        trainer.train(args.epochs, generator_file=args.generator, discriminator_file=args.discriminator, file_name_plot=args.plot)

    trainer.make_plot(args.plot)