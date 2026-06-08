import unittest

import ray
import torch
from ray.util.state.api import list_actors

from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray


@ray.remote
class GpuMemHolder:
    def __init__(self):
        torch.cuda.set_device(0)
        print('haha')
        self.x = torch.zeros((4, 1024, 1024, 1024), device='cuda')


class RayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        ray.init()

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_num_gpus(self):
        assert ray.cluster_resources()["GPU"] >= 16

    '''
    def test_1(self):
        holder = GpuMemHolder.remote()
        all_actors = list_actors()
        n_alive = len([a for a in all_actors if a.state != 'DEAD'])
        assert n_alive == 1

    def test_2(self):
        holder = GpuMemHolder.remote()
        all_actors = list_actors()
        n_alive = len([a for a in all_actors if a.state != 'DEAD'])
        assert n_alive == 1

    def test_3(self):
        holder = GpuMemHolder.remote()
        all_actors = list_actors()
        n_alive = len([a for a in all_actors if a.state != 'DEAD'])
        assert n_alive == 1
    '''
