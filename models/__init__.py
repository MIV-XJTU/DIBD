from .DIBD import DIBD

def dibd():
    net = DIBD() 
    net.use_2dconv = False
    net.bandwise = False
    return net