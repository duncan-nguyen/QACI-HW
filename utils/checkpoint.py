"""Save / load checkpoints without dragging along the whole CLIP text tower.

The upstream repo saves with `torch.save(net)` -- pickling the entire object. For `stag` that
object also holds the CLIP text encoder (63M frozen parameters), so each checkpoint weighs
~265 MB even though the *learned* part is only ~2 MB. Training 100 epochs and saving on every
improvement means tens of GB for something HuggingFace re-downloads in seconds.

The new format is a plain dict:

    {'format': 2, 'network': 'stag', 'kwargs': {...}, 'state_dict': {...}}

`state_dict` drops every `text_encoder.*` key; on load, `CLIPTextEncoder` is rebuilt from the
HuggingFace cache. Models that cannot describe themselves (no `init_kwargs`) are still
pickled as before, so cornell/jacquard/ggcnn behaviour is unchanged.

`load_network` reads both formats.
"""

import os

import torch

FORMAT_VERSION = 2
_TEXT_PREFIX = "text_encoder."


def is_state_dict_checkpoint(obj):
    return isinstance(obj, dict) and "state_dict" in obj and "network" in obj


def checkpoint_state(net):
    """
    A self-describing dict for the model, or None if it declares no `init_kwargs`/`network_name`.
    """
    name = getattr(net, "network_name", None)
    kwargs = getattr(net, "init_kwargs", None)
    if not name or not kwargs:
        return None
    state = {
        k: v for k, v in net.state_dict().items() if not k.startswith(_TEXT_PREFIX)
    }
    return {
        "format": FORMAT_VERSION,
        "network": name,
        "kwargs": dict(kwargs),
        "state_dict": state,
    }


def save_checkpoint(net, path):
    """
    Write a checkpoint atomically (temp file then `os.replace`, so a Ctrl-C midway leaves no
    truncated file behind).

    :return: the path written
    """
    payload = checkpoint_state(net)
    if payload is None:
        payload = net  # fallback: pickle the whole module, as upstream does
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_network(path, map_location="cpu", text_encoder=None, strict=True):
    """
    Load a checkpoint in *any* of this repo's formats.

    :param text_encoder: reuse an existing CLIPTextEncoder instead of building a new one
                         (useful when evaluating several checkpoints in one process).
    :param strict: raise if the state_dict has missing/unexpected keys outside `text_encoder.*`.
    :return: an nn.Module in eval() mode
    """
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # Older checkpoints are pickled modules -> weights_only=False is required. We only
        # load files this repo produced itself, so that is acceptable.
        obj = torch.load(path, map_location=map_location, weights_only=False)

    if not is_state_dict_checkpoint(obj):
        return obj.eval()

    from inference.models import get_network

    kwargs = dict(obj["kwargs"])
    if text_encoder is not None and kwargs.get("use_text", False):
        kwargs["text_encoder"] = text_encoder
    net = get_network(obj["network"])(**kwargs)
    missing, unexpected = net.load_state_dict(obj["state_dict"], strict=False)

    missing = [k for k in missing if not k.startswith(_TEXT_PREFIX)]
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"state_dict does not match the model built from the kwargs of checkpoint {path}:\n"
            f"  missing:     {missing}\n  unexpected:  {list(unexpected)}"
        )
    if isinstance(map_location, (str, torch.device)):
        net = net.to(map_location)
    return net.eval()
