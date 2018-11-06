import argparse
import os
import torch
import torch.nn.parallel

import sys
sys.path.append("/home/taylor/Revisiting_Single_Depth_Estimation/models")
import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb
import skimage.io
import skimage.transform

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")


def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    model.eval()

    image_dir = "/home/taylor/Mirror-Segmentation/data_640/train/image/"
    mask_dir = "/home/taylor/Mirror-Segmentation/data_640/train/mask/"
    imglist = os.listdir(image_dir)
    output_dir = "/media/taylor/mhy/depth2/train/"
    for i, imgname in enumerate(imglist):
        print i
        mask_path = mask_dir + imgname[:-4] + "_json/label8.png"
        output_path = output_dir + imgname[:-4] + ".png"
        nyu2_loader = loaddata.readNyu2(os.path.join(image_dir, imgname))
        test(nyu2_loader, model, mask_path, output_path)


def test(nyu2_loader, model, mask_path, output_path):
    for i, image in enumerate(nyu2_loader):
        image = torch.autograd.Variable(image, volatile=True).cuda()
        out = model(image)

        depth = out.view(out.size(2), out.size(3)).data.cpu().numpy()
        max = np.max(depth)
        min = np.min(depth)
        depth = (depth - min) / (max - min)
        depth = skimage.transform.resize(depth, [2*out.size(2), 2*out.size(3)], order=3)
        # depth = depth * (max - min) + min
        # depth = depth.astype(np.float32)
        # np.save(output_path, depth)

        # process mirror region
        mask = skimage.io.imread(mask_path)
        height = mask.shape[0]
        width = mask.shape[1]
        num_obj = np.max(mask)

        output_depth = depth
        for index in range(num_obj):

            mirror_depth = []
            for j in range(height):
                for i in range(width):
                    if mask[j, i] == index + 1:
                        mirror_depth.append(depth[j, i])

            mean_mirror_depth = sum(mirror_depth) / len(mirror_depth)
            print("mean depth of mirror {} is : {}".format(index, mean_mirror_depth))

            for j in range(height):
                for i in range(width):
                    if mask[j, i] == index + 1:
                        output_depth[j, i] = mean_mirror_depth

        skimage.io.imsave(output_path, output_depth)


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