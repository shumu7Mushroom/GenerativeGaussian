import math
import torch
import torch.nn.functional as F
from loss_utils import l1_loss, ssim

##################################################################################
## PGSR
def single_view_geo_loss_from_two_depths(depth1, depth2, ref_image, cam, weight=0.05):
    """
    使用两张深度图的几何一致性 loss：
    - depth1：用于计算 GT normal（参考深度）
    - depth2：用于计算 predicted normal（预测深度）
    """

    from utils.loss_utils import get_img_grad_weight

    # squeeze to [H,W]
    if depth1.ndim == 3:
        depth1 = depth1.squeeze()
    if depth2.ndim == 3:
        depth2 = depth2.squeeze()

    # ------------------------------
    # 1) 根据 depth1 计算 GT normal
    # ------------------------------
    pts1 = depth_to_points_cam(depth1, cam)
    gt_normal = points_to_normals(pts1)
    gt_normal = orient_normals_towards_camera(gt_normal, pts1)
    gt_normal_world = normal_c2w(gt_normal, cam)   # [H,W,3]

    # -------------------------------------
    # 2) 根据 depth2 计算 Predicted normal
    # -------------------------------------
    pts2 = depth_to_points_cam(depth2, cam)
    pred_normal = points_to_normals(pts2)
    pred_normal = orient_normals_towards_camera(pred_normal, pts2)
    pred_normal_world = normal_c2w(pred_normal, cam)  # [H,W,3]

    # ------------------------------
    # 3) 图像梯度权重
    # ------------------------------
    image_weight = (1.0 - get_img_grad_weight(ref_image))
    image_weight = image_weight.clamp(0, 1).detach() ** 2
    image_weight = image_weight.unsqueeze(-1).expand(-1, -1, 3)

    # ------------------------------
    # 4) normal 差异（L1）
    # ------------------------------
    diff = (gt_normal_world - pred_normal_world).abs().sum(-1)

    if image_weight.shape[-1] == 3:
        image_weight = image_weight.mean(-1)  # [H, W]
    
    return weight * (image_weight * diff).mean()


def depth_to_points_cam(depth, cam):
    if depth.ndim == 3:
        depth = depth.squeeze()

    H, W = depth.shape
    device = depth.device

    xs, ys = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    xs = xs.float()
    ys = ys.float()

    fx = 0.5 * W / math.tan(cam.FoVx * 0.5)
    fy = 0.5 * H / math.tan(cam.FoVy * 0.5)
    cx = (W - 1) / 2
    cy = (H - 1) / 2

    Z = depth
    X = (xs - cx) / fx * Z
    Y = (ys - cy) / fy * Z

    return torch.stack([X, Y, Z], dim=-1)   # [H,W,3]

def points_to_normals(points):
    H, W, _ = points.shape

    dx = points[:, 2:, :] - points[:, :-2, :]
    dy = points[2:, :, :] - points[:-2, :, :]

    # pad back to [H,W,3]
    dx = F.pad(dx, (0,0,1,1))  # pad width
    dy = F.pad(dy, (0,0,0,0,1,1))  # pad height

    normals = torch.cross(dx, dy, dim=-1)
    normals = F.normalize(normals, dim=-1, eps=1e-6)

    return normals.permute(2,0,1)


def orient_normals_towards_camera(normals, pts):
    H, W, _ = pts.shape
    view = -pts / (torch.norm(pts, dim=-1, keepdim=True) + 1e-6)

    n = normals.permute(1,2,0)
    dot = (n * view).sum(-1)
    flip = dot < 0
    n[flip] = -n[flip]

    return n.permute(2,0,1)

def normal_c2w(normal_cam, cam):
    if hasattr(cam, 'world_view_transform'):
        # c2w = (w2c.T).inverse()
        c2w = cam.world_view_transform.T.inverse()
        R_c2w = c2w[:3, :3]
        normal_world = normal_cam.permute(1, 2, 0) @ R_c2w.T
    else:
        normal_world = normal_cam.permute(1, 2, 0)
    return normal_world

def single_view_geo_loss(pred_normal_world, depth, ref_image, cam, weight=0.05):
    from utils.loss_utils import get_img_grad_weight

    if depth.ndim == 3:
        depth = depth.squeeze()

    # 1) GT Normal from depth in camera space
    pts_cam = depth_to_points_cam(depth, cam)
    gt_normal = points_to_normals(pts_cam)
    gt_normal = orient_normals_towards_camera(gt_normal, pts_cam)
    gt_normal_world = normal_c2w(gt_normal,cam)
    # 2) Image gradient weights from GT
    image_weight = (1.0 - get_img_grad_weight(ref_image))
    image_weight = image_weight.clamp(0,1).detach() ** 2

    image_weight = image_weight.unsqueeze(-1).expand(-1, -1, 3)


    diff = (gt_normal_world - pred_normal_world).abs().sum(-1)
    return weight * (image_weight * diff).mean()

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
         [0., fy, H/2.],
         [0., 0., 1.0]]
    ).float().cuda()

    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    pts = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1,3).float().cuda()

    if depthmap.ndim == 3 and depthmap.shape[2] == 3:
        # 取单通道（如深度图通常是灰度，可能在 [H, W, 3] 里每个通道都一样，或只用第一个通道）
        depthmap = depthmap[..., 0]
    # 现在 depthmap shape 应为 [1024, 1024]
    depthmap = depthmap.reshape(-1, 1)  # [262144, 1]


    rays_d = pts @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1,1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    points = depths_to_points(view, depth).reshape(*depth.shape, 3)
    output = torch.zeros_like(points)

    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]

    normal_map = torch.nn.functional.normalize(torch.cross(dy, dx, dim=-1), dim=-1)

    output[1:-1,1:-1] = normal_map
    return output


def ssim_loss(image_resized, closest_anchor):

    lambda_dssim = 0.2   

    Ll1 = l1_loss(image_resized, closest_anchor)
    ssimloss = 1.0 - ssim(image_resized, closest_anchor)

    ssimloss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * ssimloss

    return ssimloss