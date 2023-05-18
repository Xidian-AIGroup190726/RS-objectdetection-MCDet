import torch

from build_utils.layers import *
from build_utils.parse_config import *
from torch.nn import functional as f
from train_utils.draw_feachermap import draw_features

ONNX_EXPORT = False


def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表
    :param img_size:
    :return:
    """

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size

    modules_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()

    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    # 遍历搭建每个层结构
    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not

            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            if isinstance(k, int):
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type.")

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d":
            pass

        elif mdef["type"] == "maxpool":
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"] == "route":  # [-2],  [-1,-3,-5,-6], [-1, 61]
            layers = mdef["layers"]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            # routs.extend([i + l if l < 0 else l for l in layers])
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        elif mdef['type'] == 'Shallow_Clue_Refinement':
            modules.add_module('Shallow_Clue_Refinement',
                               Shallow_Clue_Refinement(kernel_size=int(mdef['kernel_size']),stride=int(mdef['stride']),
                                                  padding=int(mdef['padding']), in_channels=int(mdef['in_channels']),
                                                  out_channels=int(mdef['out_channels'])))
            filters = mdef['out_channels']

        elif mdef['type'] == 'self_dilating_Pooling':
            modules.add_module('self_dilating_Pooling',
                               self_dilating_Pooling(in_channels=int(mdef['in_channels']), out_channels=int(mdef['out_channels'])))


        elif mdef["type"] == "yolo":
            yolo_index += 1  # 记录是第几个yolo_layer [0, 1, 2]
            stride = [32, 16, 8]  # 预测特征层对应原图的缩放比例

            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = -1

                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
    对YOLO的输出进行处理
    """
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85: x, y, w, h, obj, cls1, ...)
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        # 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride
        # batch_size, na, grid_h, grid_w, wh,
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                self.create_grids((nx, ny), p.device)

        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny  # 3*
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)

            p[:, :2] = (torch.sigmoid(p[:, 0:2]) + grid) * ng  # x, y
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)

        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        mask = []
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ""

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:
                if i == 76 or i == 89 or i == 103:
                    x, att_mask = module(x)
                    mask.append(att_mask)
                else:
                    x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train
            return yolo_out, mask
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            return p
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=False):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)


def get_yolo_layers(self):
    """
    获取网络中三个"YOLOLayer"模块对应的索引
    :param self:
    :return:
    """
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]


# Shallow Clue Refinement (SCR)
class Shallow_Clue_Refinement(nn.Module):
    def __init__(self, kernel_size, stride, padding, in_channels, out_channels):
        super(Shallow_Clue_Refinement, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_out_channels = in_channels*2//16
        self.padding = padding
        self.K = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1,padding=0)
        self.G = nn.Conv2d(in_channels=self.in_channels, out_channels=self.down_out_channels, kernel_size=1, stride=1,
                           padding=0)

        self.MaxPololing = nn.MaxPool2d(self.kernel_size, self.stride, self.padding)
        self.BN = nn.BatchNorm2d(self.out_channels)
        self.downsample = 1
        self.BN1 = nn.BatchNorm2d(1)
        self.leakyRelu = nn.LeakyReLU()
        self.Relu = nn.ReLU()

        self.KSANetconv = nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.down_out_channels,
                                    kernel_size=1, stride=1,
                                    padding=0)
        self.QSANetconv = nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.down_out_channels, kernel_size=1, stride=1,
                                    padding=0)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        shallowFeaturemap = x[0]
        deepFeaturemap = x[1]

        Batch_size, Channel, Height, Width = shallowFeaturemap.size()

        max_out = self.KSANetconv(deepFeaturemap)
        h = int(Height / self.stride)
        w = int(Width / self.stride)
        max_out1, _ = torch.max(max_out, dim=-3)
        QSANetconv = torch.unsqueeze(max_out1.view(Batch_size, -1, 1), dim=2)
        KSANetconv = torch.unsqueeze(max_out.view(Batch_size, -1, self.down_out_channels), dim=2)
        lit_attentionmap = torch.matmul(QSANetconv, KSANetconv)
        lit_attentionmap = torch.mean(lit_attentionmap, dim=-3).view(Batch_size, 1, self.down_out_channels)
        VSANetconv = max_out.view(Batch_size, self.down_out_channels, -1)
        re_score = torch.matmul(lit_attentionmap, VSANetconv).view(Batch_size, 1, h, w)
        sigmoid_re_score = torch.sigmoid(re_score)

        Q_Unfold_window = torch.unsqueeze(
            max_out.view(Batch_size, self.down_out_channels, int(Height / self.stride * Width / self.stride)), 3)
        Q_Unfold_window = torch.unsqueeze(Q_Unfold_window,4).permute(0, 2, 3, 1, 4).contiguous()
        Q_Unfold_window = Q_Unfold_window.expand(Batch_size, int(Height*Width/4), self.out_channels, self.down_out_channels, 1)

        K_x = self.K(shallowFeaturemap)
        K_Unfold = f.unfold(K_x, kernel_size=self.kernel_size, dilation=1, padding=self.padding, stride=self.stride)
        K_Unfold = K_Unfold.view(Batch_size, self.out_channels, -1, int(Height / self.stride * Width / self.stride)).contiguous()
        K_Unfold_window = torch.unsqueeze(K_Unfold,4).permute(0, 3, 1, 4, 2).contiguous()

        attention_map = torch.matmul(Q_Unfold_window, K_Unfold_window)
        attention_map = torch.squeeze(torch.mean(attention_map, dim=-2), dim=-2)
        new_kernel = torch.softmax(attention_map,
                                   dim=-1)

        K_Unfold = K_Unfold.permute(0, 3, 1, 2).contiguous()
        attention_map = torch.sum((new_kernel* K_Unfold),dim=-1).permute(0, 2, 1)

        attention_map = attention_map.view(Batch_size, self.out_channels, int(Height / self.stride),
                                           int(Width / self.stride))

        re_score_attention_map = sigmoid_re_score * attention_map + attention_map
        
        if False:  # Visual feature maps
          savepath = './feature_map_save'
          draw_features(8, 8, K_x.cpu().numpy(), "{}/conv_input_52_52.png".format(savepath))
          draw_features(8, 8, attention_map.cpu().numpy(), "{}/ada_conv_output_52_52.png".format(savepath))
          draw_features(8, 8, re_score_attention_map.cpu().numpy(), "{}/rescore_conv_output_52_52.png".format(savepath))
          draw_features(1, 1, (sigmoid_re_score).cpu().numpy(), "{}/global_descriptors_map_output_52_52.png".format(savepath))

        return re_score_attention_map, re_score


class self_dilating_Pooling(nn.Module):  # 20220814-100131 93.4*40 #double route
    def __init__(self, in_channels, out_channels, initialkernel=1, maxkernel=13, clazz=20):
        super(self_dilating_Pooling, self).__init__()
        self.initialkernel = initialkernel  # 初始膨胀率=1
        self.maxkernel = maxkernel  # 最大膨胀率=13

        self.Category_score_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels//16, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        )

        self.kernel_sizes = torch.tensor([1, 3, 5, 7, 9, 11, 13])
        #self.kernel_sizes = torch.tensor([1, 5, 9, 13])

        self.pooling_ModuleList = nn.ModuleList()
        for i, maxpool_kernel in enumerate(self.kernel_sizes[0:]):
            self.pooling_ModuleList.append(
                nn.MaxPool2d(kernel_size=int(maxpool_kernel), stride=1, padding=int(maxpool_kernel) // 2))
        self.relu = nn.ReLU()

    def forward(self, x):

        b, c, h, w = x.size()
        Content_score = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        globle_score = self.Category_score_block(Content_score)
        
        expend_num = globle_score.view(b, c, -1)
        expend_num_relu = self.relu(expend_num)

        dtype = x.data.type()
        devices = x.data.device

        kernel_sizes = self.kernel_sizes  # tensor([ 1,  3,  5,  7,  9, 11, 13])
        q_s = expend_num_relu.detach().floor()
        q_s = torch.clamp(q_s, 0, len(kernel_sizes) - 2).long()
        q_b = q_s + 1
        small_patch = q_s.view(b * c, -1)
        index = torch.arange(0, b * c).unsqueeze(1).to(devices)

        mix_min_pool = torch.zeros(0, h, w).type(dtype).to(devices)
        mix_max_pool = torch.zeros(0, h, w).type(dtype).to(devices)

        s_pool_dict, s_mix_select = self._get_p(small_patch, index, x, dtype, devices)
        for i, maxpool in enumerate(self.kernel_sizes[0:]):
            recent_pool = self.pooling_ModuleList[i]
            recent_feature = s_pool_dict[str(int(maxpool))]
            if recent_feature.size(0) != 0 and i != 0:
                pool_output = recent_pool(recent_feature)
            else:
                pool_output = recent_feature
            mix_min_pool = torch.cat((mix_min_pool, pool_output), dim=0)

        sorted_logits, sorted_indices = torch.sort(s_mix_select, descending=False, dim=0)  # 对logits进行递减排序
        s_outpool = torch.index_select(mix_min_pool, dim=0, index=sorted_indices).view(b, c, h, w)  # 小的从组合输出

        big_patch = q_b.view(b * c, -1)
        b_pool_dict, b_mix_select = self._get_p(big_patch, index, x, dtype, devices)

        for i, maxpool in enumerate(self.kernel_sizes[0:]):
            recent_pool = self.pooling_ModuleList[i]
            recent_feature = b_pool_dict[str(int(maxpool))]
            if recent_feature.size(0) != 0:
                pool_output = recent_pool(recent_feature)
            else:
                pool_output = recent_feature
            mix_max_pool = torch.cat((mix_max_pool, pool_output), dim=0)

        sorted_logits, sorted_indices = torch.sort(b_mix_select, descending=False, dim=0)
        b_outpool = torch.index_select(mix_max_pool, dim=0, index=sorted_indices).view(b, c, h, w)

        # bilinear
        bilinear_output = (torch.unsqueeze((expend_num_relu - q_s), dim=2).repeat(1, 1, h, w) * b_outpool) + (
                    torch.unsqueeze((q_b - expend_num_relu), dim=2).repeat(1, 1, h, w) * s_outpool)
        ssp_output =  bilinear_output + x
        return ssp_output

    def _get_p(self, patch, index, x, dtype, devices):
        b, c, h, w = x.size()
        mix_select = torch.zeros(0).type(dtype)
        mix_select = torch.as_tensor(mix_select, dtype=torch.int).to(devices)
        tiny_dict = {}
        for i, maxpool in enumerate(self.kernel_sizes[0:]):
            patch = patch
            mask = patch.eq(int(i))
            select = torch.masked_select(index, mask)
            mix_select = torch.cat((mix_select, select), dim=0)
            select_pooling = torch.index_select(x.view(b * c, h, w).float(), 0, select)
            tiny_dict.update({str(int(maxpool)): select_pooling})
        return tiny_dict, mix_select

class self_dilating_Pooling0(nn.Module):
    def __init__(self, in_channels, out_channels, initialkernel=1, maxkernel=13, clazz=10):
        super(self_dilating_Pooling, self).__init__()
    def forward(self, x):
        return x
