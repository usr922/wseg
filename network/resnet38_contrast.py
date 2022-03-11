import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self, num_classes=21):
        super().__init__()

        self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False)

        self.proj_fc = nn.Conv2d(4096, 128, 1, bias=False)

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.proj_fc.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8, self.proj_fc, self.f8_3, self.f8_4, self.f9]

    def forward(self, img):
        d = super().forward_as_dict(img)
        x = d['conv6']
        feats = F.relu(self.proj_fc(x), inplace=True)

        cam = self.fc8(x)

        n, c, h, w = cam.size()

        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)+1e-5
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, -1, :, :] = 1 - torch.max(cam_d_norm[:, :-1, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, :-1, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, :-1, :, :][cam_d_norm[:, :-1, :, :] < cam_max] = 0

        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
        x_s = F.interpolate(img, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)

        cam_rv = self.PCM(cam_d_norm, f)

        pred = F.avg_pool2d(cam, kernel_size=(h, w), padding=0)
        pred = pred.view(pred.size(0), -1)

        pred_rv = F.avg_pool2d(cam_rv, kernel_size=(h, w), padding=0)
        pred_rv = pred_rv.view(pred_rv.size(0), -1)

        return pred, cam, pred_rv, cam_rv, feats

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h*w)
        f = self.f9(f)
        f = f.view(n, -1, h*w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)
        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff/(torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv
