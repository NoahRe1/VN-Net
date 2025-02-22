from .lgvnfm import LGVNFM


def get_fusion_method(name):
    if name.lower() == "lgvnfm":
        return LGVNFM
    else:
        raise ValueError(f"Unknown fusion method: {name}")
