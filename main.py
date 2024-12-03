import torch
import argparse
import yaml
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from models.Unet import UNet
from dataset.Car import CarDataset
from torch.utils.data import DataLoader
from tools.function import train, validate




def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./config/config.yaml',
                        type=str)
    args = parser.parse_args()
    return args

def load_yaml_config(cfg_path):
    """Load configuration from YAML file."""
    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(device)
    print('torch_version:', torch.__version__)  # 查看torch版本
    print('cuda_version:', torch.version.cuda)  # 查看cuda版本

    args = parse_args()
    # load YAML
    cfg = load_yaml_config(args.cfg)

    transforms = T.Compose([T.ToTensor(),
                            T.Resize((cfg['DATASET']['IMG_HEIGHT'], cfg['DATASET']['IMG_WIDTH']))
                            ])
    # dataset 
    all_data  = CarDataset(cfg['DATASET']['TRAIN_SET'], cfg['DATASET']['LABEL_SET'], transforms)
    train_data, val_data = torch.utils.data.random_split(all_data, [0.7, 0.3])
    train_loader = DataLoader(train_data, batch_size=cfg['TRAIN']['BATCH_SIZE_PER_GPU'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg['TEST']['BATCH_SIZE_PER_GPU'], shuffle=False)

    # model set up
    
    model = UNet()
    model.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    print('model_set_up')

    # train
    print('Training')
    for epoch in range(cfg['TRAIN']['END_EPOCH']):
        train_acc = train(model, train_loader, loss_function, optimizer, device, cfg['PRINT_FREQ'], cfg)
        val_acc = validate(model, val_loader, device, cfg)
        print(f'Epoch[{epoch+1}] Train_Acc:{train_acc}, Val_Acc: {val_acc}')

if __name__ == '__main__':
    main()





    