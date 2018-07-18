from SpeechModule import DeepSpeech
import unittest
import psutil
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
class DeepSpeechProfile(unittest.TestCase):
    def setUp(self):
        self.model = DeepSpeech()
        self.model.eval()
        self.input = Variable(torch.from_numpy(np.random.rand(6, 1, 161, 1275).astype(np.float32)))

    def test_gpu(self):
        print('testing gpu version')
        model = self.model.cuda() 
        embark = time.time()
        model(self.input.cuda())
        print('time cost: {}s'.format(time.time() - embark))
        print('#####################################')
        self.assertTrue(True)


    def test_cpu(self):
        print('testing cpu version')
        print('cores count: {}'.format(psutil.cpu_count()))
        model = self.model.cpu()
        embark = time.time()
        model(self.input)
        print('time cost: {}s'.format(time.time() - embark))
        print('#####################################')
        self.assertTrue(True)

if __name__ == '__main__':
        unittest.main()
