import os

import cv2 
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn

from tqdm import tqdm


class DepthImageDataSet(Dataset):
    def __init__(self):
        self.gt_depth_dir = "./data/depth_gt/0048"
        self.input_depth_dir = "./data/depth_input/0048"
        self.filenames = self.get_filenames()

    
    def __len__(self):
        return len(os.listdir(self.gt_depth_dir))
    

    def get_filenames(self):
        files = os.listdir(self.input_depth_dir)
        f = []
        for file in files:
            f.append(file)

        f.sort()
        return f
    

    def __getitem__(self, idx):
        depth_input_image_path = os.path.join(self.gt_depth_dir, self.filenames[idx])
        depth_gt_image_path = os.path.join(self.input_depth_dir, self.filenames[idx])

        depth_input_image = cv2.imread(depth_input_image_path, cv2.IMREAD_ANYDEPTH)
        depth_gt_image = cv2.imread(depth_gt_image_path, cv2.IMREAD_ANYDEPTH)

        depth_input_image = torch.from_numpy(depth_input_image.astype(np.float32)).unsqueeze(0)
        depth_gt_image = torch.from_numpy(depth_gt_image.astype(np.float32)).unsqueeze(0)

        return depth_input_image, depth_gt_image


class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_net = nn.Sequential(
                            nn.Conv2d(1, 32, 3, padding='same'),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, 3, padding='same'),
                            nn.ReLU(),

                            nn.Conv2d(64, 64, 3, padding='same'),
                            nn.ReLU(),
                            nn.Conv2d(64, 32, 3, padding='same'),
                            nn.ReLU(),
                            nn.Conv2d(32, 1, 3, padding='same')
                        )
        
    def forward(self, depth_input):
        depth_pred = self.scale_net(depth_input)
        return depth_pred

        
def create_dataloader():
    dataset = DepthImageDataSet()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader


def train_step(model, train_loader, optimizer, device) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0.

    model.train()
    
    for batch_idx, data in enumerate(train_loader):
        # --- Your code here
        depth_input_image, depth_gt_image = data
        depth_gt_image = depth_gt_image.to(device)
        depth_input_image = depth_input_image.to(device)

        optimizer.zero_grad()
        out = model(depth_input_image)

        loss = nn.MSELoss()(out, depth_gt_image)
        loss.backward()
        optimizer.step()
        # ---
        train_loss += loss.item()
    return train_loss/len(train_loader)



def train_model(model, train_dataloader, num_epochs=5, lr=1e-3, device='cpu'):
    """
    Trains the given model for `num_epochs` epochs. Use Adam as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    optimizer = None
    # Initialize the optimizer
    # --- Your code here
    # optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr)


    # ---
    pbar = tqdm(range(num_epochs))
    train_losses = []

    for epoch_i in pbar:
        train_loss_i = None
        train_loss_i = train_step(model=model, train_loader=train_dataloader, optimizer=optimizer, device=device)
        train_losses.append(train_loss_i)
        pbar.set_description("Training loss: %f" % train_loss_i)

    return train_losses


def run():
    num_epochs = 50
    device = "cuda:0"

    train_dataloader, test_dataloader = create_dataloader()
    scale_net_model = ScaleNet().to(device)

    train_losses = train_model(model=scale_net_model,
                               train_dataloader=train_dataloader,
                               num_epochs=num_epochs,
                               device=device)
    

    # plot train loss
    plt.plot(figsize=(12, 3))
    plt.plot(train_losses)
    plt.grid()
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.yscale('log')

    plt.show()
    plt.savefig(os.path.join("./results", "training_loss.png"))

    # ---

    # save model:
    save_path = os.path.join("./model", "scan_net_model.pt")
    torch.save(scale_net_model.state_dict(), save_path)


if __name__ == "__main__":
    run()


    


