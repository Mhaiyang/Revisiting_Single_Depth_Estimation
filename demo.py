import argparse
import os
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")


def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    model.eval()

    imglist = os.listdir("/home/taylor/Revisiting_Single_Depth_Estimation/data/mirror")
    output_dir = "/home/taylor/Revisiting_Single_Depth_Estimation/data/mirror_depth/"
    for i, imgname in enumerate(imglist):
        print i
        output_path = output_dir + imgname
        nyu2_loader = loaddata.readNyu2(imgname)
        test(nyu2_loader, model, output_path)


def test(nyu2_loader, model, output_path):
    image = torch.autograd.Variable(nyu2_loader, volatile=True).cuda()
    out = model(image)

    matplotlib.image.imsave(output_path, out.view(out.size(2),out.size(3)).data.cpu().numpy())


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


if __name__ == '__main__':
    main()