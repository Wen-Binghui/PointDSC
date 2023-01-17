from misc.cal_fcgf import process_3dmatch
from misc.fcgf import ResUNetBN2C as FCGF
import torch
import MinkowskiEngine as ME

model = FCGF(
        1,
    32,
    bn_momentum=0.05,
    conv1_kernel_size=7,
    normalize_feature=True
).cuda()
# 3DMatch: http://node2.chrischoy.org/data/projects/DGR/ResUNetBN2C-feat32-3dmatch-v0.05.pth
# KITTI: http://node2.chrischoy.org/data/projects/DGR/ResUNetBN2C-feat32-kitti-v0.3.pth
checkpoint = torch.load("misc/ResUNetBN2C-feat32-3dmatch-v0.05.pth")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
process_3dmatch(voxel_size=0.05)