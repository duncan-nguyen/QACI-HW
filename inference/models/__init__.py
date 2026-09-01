def get_network(network_name):
    network_name = network_name.lower()
    # Original GR-ConvNet
    if network_name == "grconvnet":
        from .grconvnet import GenerativeResnet

        return GenerativeResnet
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == "grconvnet2":
        from .grconvnet2 import GenerativeResnet

        return GenerativeResnet
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == "grconvnet3":
        from .grconvnet3 import GenerativeResnet

        return GenerativeResnet
    # STAG: GR-ConvNet-3 + part-level text-visual alignment (Grasp-Anything++)
    elif network_name == "stag":
        from .stag import STAG

        return STAG
    elif network_name == "grconvnet4":
        from .grconvnet4 import GenerativeResnet

        return GenerativeResnet
    elif network_name == "ragt":
        from .ragt.ragt import RAGT

        return RAGT
    elif network_name == "ggcnn":
        from .ggcnn import GGCNN

        return GGCNN
    else:
        raise NotImplementedError(f"Network {network_name} is not implemented")
