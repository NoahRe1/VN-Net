from .mlvfem import get_MLVFEM


def get_image_encoder(encoder_type, id):
    if encoder_type.lower() == "mlvfem":
        return get_MLVFEM()
    else:
        raise ValueError("Unknown image encoder type: {}".format(encoder_type))
