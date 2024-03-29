from models.resnet_simclr import ResNetSimCLR
from data_aug.dataset_wrapper import DataSetWrapper
from torch.utils.data import DataLoader
from data_aug.cell_dataset import CellDataset
import torchvision.transforms as transforms
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm


def get_data(config):
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.7469, 0.7403, 0.7307), (0.1548, 0.1594, 0.1706))])
    dataset = CellDataset(config['eval_dataset']['path'],
                          config['eval_dataset']['root_dir'],
                          eval(config['dataset']['input_shape']),
                          config['dataset']['preload'],
                          num_workers=1,
                          transform=data_transforms)
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=1,
        shuffle=False)
    return loader

def convert_tensor_to_np(model, data_loader):
    train_feature_vector = []
    train_labels_vector = []
    model.eval()
    for batch_x, batch_y in tqdm(data_loader):
        batch_x = batch_x.to("cuda")
        train_labels_vector.extend(batch_y)
        features, full_features = model(batch_x)

        train_feature_vector.extend(features.cpu().detach().numpy())

    train_feature_vector = np.array(train_feature_vector)
    train_labels_vector = np.array(train_labels_vector)

    return train_feature_vector, train_labels_vector


def main():
    config = yaml.load(open("config.yaml", "r"))

    model = ResNetSimCLR(config["model"]["base_model"], config["model"]["out_dim"]).to("cuda")
    # update model path
    model_directory = "Dec28_19-51-34_c520871eabf7"
    checkpoint = torch.load(f"{config['run_dir']}/{model_directory}/checkpoints/model_latest.pth")
    model.load_state_dict(checkpoint["state_dict"])

    loader = get_data(config)
    X, Y = convert_tensor_to_np(model, loader)

    np.save(f"{config['run_dir']}/{model_directory}/features.npy", X)


def get_embeddings(config, model_path, loader):
    model = ResNetSimCLR(config["model"]["base_model"], config["model"]["out_dim"]).to("cuda")
    # update model path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])

    X, Y = convert_tensor_to_np(model, loader)

    return X

if __name__ == "__main__":
    main()
