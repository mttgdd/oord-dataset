from config.config import get_cfg_impl
from src.raplace import SinoFFT, RaPlace
from src.test import test

test(get_cfg_impl('config/raplace.yaml'), 
     SinoFFT(), RaPlace(), 'raplace')