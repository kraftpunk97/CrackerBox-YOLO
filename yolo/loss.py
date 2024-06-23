import torch
import matplotlib.pyplot as plt

# compute Intersection over Union (IoU) of two bounding boxes
# the input bounding boxes are in (cx, cy, w, h) format
def compute_iou(pred, gt):
    x1p = pred[0] - pred[2] * 0.5
    x2p = pred[0] + pred[2] * 0.5
    y1p = pred[1] - pred[3] * 0.5
    y2p = pred[1] + pred[3] * 0.5
    areap = (x2p - x1p + 1) * (y2p - y1p + 1)    
    
    x1g = gt[0] - gt[2] * 0.5
    x2g = gt[0] + gt[2] * 0.5
    y1g = gt[1] - gt[3] * 0.5
    y2g = gt[1] + gt[3] * 0.5
    areag = (x2g - x1g + 1) * (y2g - y1g + 1)

    xx1 = max(x1p, x1g)
    yy1 = max(y1p, y1g)
    xx2 = min(x2p, x2g)
    yy2 = min(y2p, y2g)

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / (areap + areag - inter)    
    return iou


# output: (batch_size, num_boxes * 5 + num_classes, 7, 7), raw output from the network
# pred_box: (batch_size, num_boxes * 5 + num_classes, 7, 7), predicted bounding boxes from the network (see the forward() function)
# gt_box: (batch_size, 5, 7, 7), ground truth bounding box target from the dataloader
# gt_mask: (batch_size, 7, 7), ground truth bounding box mask from the dataloader
# num_boxes: number of bounding boxes per cell
# num_classes: number of object classes for detection
# grid_size: YOLO grid size, 64 in our case
# image_size: YOLO image size, 448 in our case
def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size, device):
    batch_size = output.shape[0]
    num_grids = output.shape[2]
    # compute mask with shape (batch_size, num_boxes, 7, 7) for box assignment
    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids)

    # compute assignment of predicted bounding boxes for ground truth bounding boxes
    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
 
                # if the gt mask is 1
                if gt_mask[i, j, k] > 0:
                    # transform gt box
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * grid_size
                    gt[1] = gt[1] * grid_size + j * grid_size
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size
                    # This is basically un-normalized ground truth
                    # print('gt in loss %.2f, %.2f, %.2f, %.2f' % (gt[0], gt[1], gt[2], gt[3]))

                    select = 0
                    max_iou = -1
                    # select the one with maximum IoU
                    for b in range(num_boxes):
                        # center x, y and width, height
                        pred = pred_box[i, 5*b:5*b+4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b
                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou
    box_confidence = box_confidence.to(device)
    box_mask = box_mask.to(device)
    # compute yolo loss
    weight_coord = 5.0
    weight_noobj = 0.5

    # according to the YOLO paper, we compute the following losses
    # loss_x: loss function on x coordinate (cx)
    # loss_y: loss function on y coordinate (cy)
    # loss_w: loss function on width 
    # loss_h: loss function on height
    # loss_obj: loss function on confidence for objects
    # loss_noobj: loss function on confidence for non-objects
    # loss_cls: loss function for object class


    def sq_error(gt_, pred_):
        return torch.pow(gt_ - pred_, 2.0)

    loss_x = weight_coord * torch.sum(box_mask * sq_error(gt_box[:, 0, :, :].unsqueeze(dim=1), output[:, 0:5*num_boxes:5, :, :]))
    loss_y = weight_coord * torch.sum(box_mask * sq_error(gt_box[:, 1, :, :].unsqueeze(dim=1), output[:, 1:5*num_boxes:5, :, :]))
    loss_w = weight_coord * torch.sum(box_mask * sq_error(torch.sqrt(gt_box[:, 2, :, :].unsqueeze(dim=1)), torch.sqrt(output[:, 2:5*num_boxes:5, :, :])))
    loss_h = weight_coord * torch.sum(box_mask * sq_error(torch.sqrt(gt_box[:, 3, :, :].unsqueeze(dim=1)), torch.sqrt(output[:, 3:5*num_boxes:5, :, :])))
    loss_obj = torch.sum(box_mask * sq_error(box_confidence, output[:, 4:5*num_boxes:5, :, :]))
    loss_noobj = weight_noobj * torch.sum((1-box_mask) * sq_error(box_confidence, output[:, 4:5*num_boxes:5, :, :]))
    loss_cls = torch.sum(box_mask * sq_error(gt_mask.unsqueeze(dim=1), output[:, 5*num_boxes:, :, :]))

    # the total loss
    loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls
    return loss

# plot losses
def plot_losses(losses, filename='train_loss.png'):

    num_epoches = losses.shape[0]
    l = np.mean(losses, axis=1)

    plt.subplot(1, 1, 1)
    plt.plot(range(num_epoches), l, marker='o', alpha=0.5, ms=4)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.xlim()

    plt.gcf().set_size_inches(6, 4)
    plt.savefig(filename, bbox_inches='tight')
    print('save training loss plot to %s' % (filename))
    plt.clf()