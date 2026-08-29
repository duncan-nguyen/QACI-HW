def get_network(network_name):
    network_name = network_name.lower()
    # Original GR-ConvNet
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == 'grconvnet2':
        from .grconvnet2 import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == 'grconvnet3':
        from .grconvnet3 import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
    # GR-ConvNet-3 + text-visual alignment ở mức part (Grasp-Anything++)
    elif network_name == 'grconvnet3_align':
        from .grconvnet3_align import GenerativeResnetAlign
        return GenerativeResnetAlign
    elif network_name == 'grconvnet4':
        from .grconvnet4 import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'ragt':
        from .ragt.ragt import RAGT
        return RAGT
    elif network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
