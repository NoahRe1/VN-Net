from .agcrn import AGCRN
from .degcn import DEGCN

def get_graph_encoder_decoder(type):
    if type.lower() == 'agcrn':
        return AGCRN
    elif type.lower() == 'degcn':
        return DEGCN
    else:
        raise ValueError('Unknown graph encoder type: {}'.format(type))