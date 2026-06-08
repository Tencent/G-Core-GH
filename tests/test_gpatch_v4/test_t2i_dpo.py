import os
import shutil
import unittest

import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose, initialize

from gpatch_v4.trainer.t2i_dpo_trainer import BaseDpoTrainer, T2iDpoConfig
from gpatch_v4.utils import dataclass_from_args


def load_config(config_path, config_name):
    with initialize(config_path=config_path, version_base=None):
        config = compose(config_name=config_name)
        return config


def main_worker(rank, world_size, cfg):
    mp.set_start_method("fork", force=True)
    # Setup for Distributed Data Parallel (DDP)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = f"{world_size}"
    os.environ["RANK"] = f"{rank}"
    import time
    trainer = hydra.utils.instantiate(cfg.trainer)
    assert isinstance(
        trainer, BaseDpoTrainer
    ), f"trainer_cls {type(trainer)} should be derived from BaseDpoTrainer"
    trainer.init()
    print("trainer.train_loop", flush=True)
    trainer.train_loop()
    trainer.finalize()
    time.sleep(2)
    dist.destroy_process_group()


class T2iDpoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # set proxy to load data from remote hub
        os.environ["http_proxy"] = "http://star-proxy.oa.com:3128"
        os.environ["https_proxy"] = "http://star-proxy.oa.com:3128"
        os.environ["ftp_proxy"] = "http://star-proxy.oa.com:3128"
        os.environ[
            "no_proxy"
        ] = ".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

    def test_t2i_dpo(self):
        cfg = load_config("configs/test_yaml", "test_t2i_dpo")
        try:
            world_size = 8  # Number of GPUs
            mp.spawn(
                main_worker,
                args=(world_size, cfg),
                nprocs=world_size,
                join=True  # Main process waits for all subprocesses to finish
            )
        finally:
            config = dataclass_from_args(cfg.trainer.args, T2iDpoConfig)
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
