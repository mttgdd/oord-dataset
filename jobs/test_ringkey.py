from config.config import get_cfg_impl
from src.ringkey import RingKey
from src.test import test

test(get_cfg_impl('config/ringkey.yaml'), 
     RingKey(), None, 'ringkey')