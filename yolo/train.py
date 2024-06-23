import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from packaging.version import Version
from model import YOLO
from data import CrackerBox
from loss import compute_loss, plot_losses

assert Version(np.__version__)<Version("2.0.0"), \
"This module was designed using NumPy v1.x.x, \
which is not compatible with NumPy v2. \
Please install the correct NumPy version (latest v.1.26.4)"


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    num_epochs = 100
    batch_size = 8
    learning_rate = 1e-4
    num_workers = 2

    dataset_train = CrackerBox('train', 'data.zip')  
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    epoch_size = len(train_loader)

    num_classes = 1
    num_boxes = 2
    network = YOLO(num_boxes, num_classes, device=device)
    image_size = network.image_size
    grid_size = network.grid_size
    network.train()

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    output_dir = 'checkpoints'
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    losses = np.zeros((num_epochs, epoch_size), dtype=np.float32)
    for epoch in range(num_epochs):
        for i, sample in enumerate(train_loader):
            image = sample['image'].to(device)
            gt_box = sample['gt_box'].to(device)
            gt_mask = sample['gt_mask'].to(device)

            output, pred_box = network(image)

            loss = compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size, network.device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[epoch, i] = loss

        epoch_loss = losses[epoch, :].sum() / epoch_size
        print('epoch %d/%d, lr %.6f, average epoch loss %.4f' % (epoch, num_epochs, learning_rate, epoch_loss))
        # save checkpoint for every fifth epoch
        if epoch % 5 == 0:
            state = network.state_dict()
            filename = 'yolo_epoch_{:d}'.format(epoch+1) + '.checkpoint.pth'
            torch.save(state, os.path.join(output_dir, filename))
            print(filename)
    state = network.state_dict()
    filename = 'yolo_final.checkpoint.pth'
    torch.save(state, os.path.join(output_dir, filename))
    print(filename)


    plot_losses(losses)