import importlib

import torch


def get_model(args):
    model = getattr(importlib.import_module(args.network), 'Net')()

    if args.weights[-7:] == '.params':
        assert args.network in ["network.resnet38_cls",
                                "network.resnet38_eps",
                                "network.resnet38_eps_seam",
                                "network_with_PCM.resnet38_eps_seam_p"]
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    elif args.weights[-11:] == '.caffemodel':
        assert args.network == "network.vgg16_cls"
        import network.vgg16d
        weights_dict = network.vgg16d.convert_caffe_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    return model
