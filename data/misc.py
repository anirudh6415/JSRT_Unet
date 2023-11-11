import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
def multiclass_masks(heart_mask,right_lung_mask,left_lung_mask,right_clavicle_mask,left_clavicle_mask):
    heart_label = 1
    right_lung_label = 2
    left_lung_label = 3
    right_clavicle_label = 4
    left_clavicle_label = 5
    
    # heart_mask = torch.from_numpy(heart_mask)
    # right_clavicle_mask = torch.from_numpy(right_clavicle_mask)
    # left_clavicle_mask = torch.from_numpy(left_clavicle_mask)
    # right_lung_mask = torch.from_numpy(right_lung_mask)
    # left_lung_mask = torch.from_numpy(left_lung_mask)
    
    multiclass_mask = torch.zeros((6, *heart_mask.shape), dtype=torch.float32)
    # multiclass_mask = torch.zeros_like(heart_mask)
    multiclass_mask[1][heart_mask == 255] = 1
    multiclass_mask[2][right_lung_mask == 255] = 1
    multiclass_mask[3][left_lung_mask == 255] = 1
    multiclass_mask[4][right_clavicle_mask == 255] = 1
    multiclass_mask[5][left_clavicle_mask == 255] = 1
    multiclass_mask[0] = 0
#     multiclass_mask = TF.to_pil_image(multiclass_mask.byte())
#     multiclass_mask = TF.resize(multiclass_mask, (244, 244))
#     multiclass_mask = TF.to_tensor(multiclass_mask)
#     multiclass_mask = Image.new('L', heart_mask.size)
#     multiclass_mask.paste(heart_label, heart_mask)
#     multiclass_mask.paste(right_lung_label, right_lung_mask)
#     multiclass_mask.paste(left_lung_label, left_lung_mask)
#     multiclass_mask.paste(right_clavicle_label, right_clavicle_mask)
#     multiclass_mask.paste(left_clavicle_label, left_clavicle_mask)

#     #multiclass_mask = np.array(multiclass_mask)
#     multiclass_mask = TF.resize(multiclass_mask, (244, 244))
#     multiclass_mask = TF.to_tensor(multiclass_mask)
    # multiclass_mask =multiclass_mask.squeeze(0).numpy()
    # print(multiclass_mask.shape)
    
    import torch.nn.functional as F

    # assuming multiclass_mask is a tensor of size (C, H, W)
    # where C is the number of classes, H is the height, and W is the width

    # define the new height and width
    new_h, new_w = 256, 256

    # resize the tensor using bilinear interpolation
    resized_mask = F.interpolate(multiclass_mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
    multiclass_mask = resized_mask.squeeze(0)

    return(multiclass_mask)
    