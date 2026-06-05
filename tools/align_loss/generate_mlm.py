from types import SimpleNamespace
import argparse
import json
import os
import pprint
import re
import shutil
import sys
import gc
import shutil
import importlib
import time
from typing import Dict

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import transformers

from megatron.core import dist_checkpointing
from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.enums import ModelType
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from megatron.legacy import fused_kernels
from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args, set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.training.utils import unwrap_model
from megatron.training import get_tokenizer
from megatron.inference.text_generation.generation import ForwardStep
from megatron.inference.text_generation.generation import generate_tokens_probs_and_return_on_first_stage
from megatron.inference.text_generation.generation import score_and_return_on_first_stage
from megatron.core.transformer.module import Float16Module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.contexts import StaticInferenceContext

from gpatch.training.arguments import gpatch_extra_args
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig
from gpatch.patch_mcore import init_gpatch_for_mcore


def gen_args(parser):
    parser = gpatch_extra_args(parser)

    group = parser.add_argument_group(title='Gen')
    group.add_argument('--prompts', type=str, nargs='*', required=True)
    group.add_argument('--max_new_tokens', type=int, default=32)
    return parser


def load_model():
    args = get_args()
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args, GpatchTransformerConfig)

    mod = importlib.import_module(args.load_model_provider)
    model = mod.default_sft_model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    iteration, _ = load_checkpoint([model], None, None)

    model.eval()
    model = Float16Module(config, model).cuda()
    return model


def get_input_ids():
    args = get_args()
    tokenizer = get_tokenizer()

    # debugging purpose only
    '''
    args.prompts = [
        "问题：同房后有褐色分泌物，伴随腹痛是不是意味着流产的风险？\n\n搜索结果（开头方括号内的数字是搜索结果序号）：\n[1] 同房后有棕色分泌物，是阴道有少量的出血，而不是怀孕的表现。同房后有少量阴道出血，需结合年龄、月经史、避孕史以及孕产史等进行具体分析。外阴道炎症可能会引起少量出血，如果是月经中期，有排卵期的出血。如果是宫颈病变，宫颈CIN和早期的宫颈癌也会导致接触性出血。如果怀孕期间同房后有流产的表现，甚至伴有腹痛，可能是流产、先兆早产或者先兆临产的表现，怀孕需要做妊娠检查，比如HCG或B超检查。\n\n请在理解问题意图的基础上，根据上面的搜索结果，生成一段优质答案。答案需要满足以下要求：\n1.  答案清晰流畅，信息有用且充分完整，不缺失核心信息，没有重复冗余，不超过500字。\n2. 答案忠实于上文所给出的事实信息，避免杜撰和揣测引申。\n3. 答案条理分明有逻辑，适当使用换行，\"-\"列表，表格等格式组织。\n4. 如果搜索结果无法支撑当前问题的回答，请直接回答【当前问题无法给出有用答案】。",
        "问题：热水器有多重\n\n搜索结果（开头方括号内的数字是搜索结果序号）：\n[1] 电热水器即开即热，方便快捷，是生活中常见的舒适设备。然而，由于墙体承重的缘故，电热水器安装不当，很容易留下安全隐患。各位读者在购买电热水器前，不妨先听听编辑的讲解，做到心中有数吧！▲现在市场上常见的储水式电热水器，多采用挂墙式安装，安装时其挂墙安装的安全问题不容忽视。从悬挂的重量来看，安装后电热水器总重量等于热水器自重再加上容积中的储水重量。▲也就是说，以市面上常见的60L热水器为例，内胆中的储水60L其重量为60kg，热水器自身重量在30~40kg，所以两者相加后的总重量接近100kg，80L及以上热水器则超过100kg，这个重量可不小啊！▲目前很多楼房在选材时都采用空心砖等的轻质建材，该重量由安装墙体来承受，且长期处于悬挂状态，因此对安装墙壁的要求较高，否则容易出现电热水器掉落事故，轻则造成财产损失，重则危及人身安全。\n▲如果在非承重墙上挂这样的大水桶，显然无法承受。为避免安装后产生电热水器掉落危险，根据国家相关标准，以及壁挂式 电热水器的安装规范，安装墙体必须能承受热水器加满水后重量的4倍。▲可以说，面对多种多样的墙体进行实际安装时，安装人员并没有专门的测试设备或测试手段来对墙体质量进行检验和判定，到底悬挂安装热水器后是否可靠，就连专业安装人员也只能凭靠其直觉来判断而已。▲这样的墙面挂上大水桶，掉落砸伤人的几率很大。不少读者反映，目前很多楼房在选材时都采用空心砖等的轻质建材，这大大地影响了墙体的承重能力，传统的热水器装满了热水的水箱最轻也达到了100多kg，对墙体的承重能力要求非常高。▲通常家中的非承重墙是远远不能承受起这么重的重量，而如果安装在承重墙体上，对于传统热水器需要在墙体进行打孔，这对建筑本身就是一种伤害，对楼房的寿 命会产生一定的影响。\n▲而安装在非承重墙上，最直接的结果就是，经较长时间悬挂后两根膨胀螺栓在墙体内会出现松脱现象，此时热水器掉落就有可能会随时发生。▲因此，面对非承重墙，要解决挂墙电热水器掉落危险，目前常用方法之一是增加固定支架来分担热水器重量，二是用穿墙钉贯穿非承重墙，再用钢制的夹板将热水器固定。▲不过，上述的解决方法都存在弊端。其实，将热水器隐藏在浴柜之中，就不用担心挂在墙上的危险了。此外，更换采用集成热水器也是一种解决方法，且更为直接靠谱，因为集成热水器中的电热水器隐藏于浴室柜中，整体产品落地放置，无需挂墙安装。▲说到这里，不少读者又有了新的困惑：既然不能将热水器安装在承重墙上。很多地方并不是想拆就拆，想改就改的。那么，又该如何分辨承重墙与轻体墙呢？下面，编辑分享几个常见的方法。▲首先听声音：用手敲击墙体，有清脆回声的是轻墙体（非承重墙），而敲打承重墙听到的基本是比较沉闷的声音。\n▲其次看厚度：通常情况下，承重墙较轻体墙都比较厚，大家可以观测一下外墙的厚度，和它差不多厚度的基本都是承重墙。 一般新建公寓的承重墙厚度都在750px以上。▲最后辨部位：外墙以及和邻居共用的墙体都是承重墙，在这类墙体上大面积掏洞是很危险的，而家中的卫生间，储藏室，厨房及过道的分隔墙体多数是非承重墙。▲需要注意的是，以上3种仅仅是目前比较普遍的承重墙辨别方法，如果读者对墙体是否承重的问题拿捏不准，则不能随便改动结构，应 当请资深监理帮忙辨别，以免为装修后的入住造成影响。以上是热水器安装与承重墙的解析 相信你已经学到了不少知识 加入《装修情报》读者社群 了解更多内容吧！装修优品旧版 ， 交易担保 ， 放心买 ，     监理读者群 Mini Program \n编辑留言：不少老房易手之后，读者已经无法找到原先的建筑施工图或平面图了。这就使得承重墙和非承重墙无法轻易区分，给装修造成了一定的困难。而在承重墙上敲敲打打或无意损毁，都会留下隐患。如果你对此抱有疑虑，或是已经遇到了困难，不妨点击【阅读原文】留下你的联系方式，找寻更多专业的监理服务吧！\n\n请在理解问题意图的基础上，根据上面的搜索结果，生成一段优质答案。答案需要满足以下要求：\n1. 答案清晰流畅，信息有用且充分完整，不缺失核心信息，没有重复冗余，不超过500字。\n2. 答案忠实于上文所给出的事实信息，避免杜撰和揣测引申。\n3. 答案条理分明有逻辑，适当使用换行，\"-\"列表，表格等格式组织。\n4. 如果搜索结果无法支撑当前问题的回答，请直接回答【当前问题无法给出有用答案】。",
        "问题：苹果8plus的防水等级是多少\n\n搜索结果（开头方括号内的数字是搜索结果序号）：\n[1] iPhone8Plus的长、宽和厚度分别为158.4mm、78.1mm、7.5mm。iPhone8Plus的介绍：iPhone 8 Plus是Apple2017年9月13日在苹果园区（Apple Park）的史蒂夫·乔布斯剧院举行苹果新品发布会上发布的手机产品。iPhone8Plus的外观特色:机身设计:iPhone 8 Plus为太空级别铝质设计，前后均为玻璃镜面。颜色参数:iPhone 8 Plus有4种颜色，分别为银色、太空灰和金色、红色特别版。尺寸重量:尺寸为高度158.4毫米（6.24英寸），宽度78.1毫米（3.07英寸），厚度7.5毫米（0.30英寸），重量202克（7.13盎司）。\niPhone8Plus功能特点:相机:iPhone 8 Plus配备1200万像素双摄像头，F1.8超大光圈设计，并采用全新的镜头模组。视频拍摄支持4K 60帧。闪光灯加入了“慢速同步技术”，前置摄像头700万像素。iPhone 8 Plus后置双摄像头主打机器学习的人像背景虚化拍摄，支持60帧码流的4K视频拍摄，支持无线充电，图形传感器加入了对AR技术的支持。防护性能:iPhone 8 Plus可防溅、抗水、防尘，在受控实验室条件下经测试，其效果在IEC 60529 标准下达到IP67级别（在最深1米的水下停留时间最长可达30分钟）。防溅、抗水、防尘功能并非永久有效，防护性能可能会因日常磨损而下降。处理器:iPhone 8 Plus配备的处理器是苹果自研的A11仿生芯片，10纳米工艺，集成两大四小共六个CPU核心，频率分别为2.45吉赫、2.06吉赫，同时整合首次自研的三核心GPU。\n辅助功能:iPhone 8 Plus为了帮助残障人士更好地使用手机，配备有如下的辅助功能：旁白、缩放、放大器、Siri和听写、键入以使用 Siri、切换控制、隐藏式字幕、辅助触控、朗读屏幕。ipone的介绍：iPhone是苹果公司（AppleInc.）发布搭载iOS操作系统的系列手机。苹果公司（AppleInc.）已发布24款 手机产品，初代：iPhone，最新版本：iPhone13 mini，iPhone13，iPhone13 Pro， iPhone13Pro Max；iPhone系列产品静音键在设备正面的左侧；iPhone5之前机型使用30Pin（即30针）接口，iPhone5（包含）之后产品使用Lightning接口。iPhoneX之前机型配置Home键；iPhoneX（包含）之后机型取消了实体Home键。\n\n请在理解问题意图的基础上，根据上面的搜索结果，生成一段优质答案。答案需要满足以下要求：\n1. 答案清晰流畅，信息有用且充分完整，不缺失核心信息，没有重复冗余，不超过500字。\n2. 答案忠实于上文所给出的事实信息，避免杜撰和揣测引申。\n3. 答案条理分明有逻辑，适当使用换行，\"-\"列表，表格等格式组织。\n4. 如果搜索结果无法支撑当 前问题的回答，请直接回答【当前问题无法给出有用答案】。",
    ]
    args.prompts = [f"###{instruction}\n### Response:\n" for instruction in args.prompts]
    '''

    # welm 和 百川 都不需要 bos
    # for pi, prompt in enumerate(args.prompts):
    #     args.prompts[pi] = f'{tokenizer._tokenizer.bos_token}{prompt}'

    t_input_ids = []
    unpadded_lens = []
    for prompt in args.prompts:
        prompt_tokenized = tokenizer._tokenizer(
            prompt, add_special_tokens=False, return_tensors='pt'
        )
        input_ids = prompt_tokenized.input_ids
        t_input_ids.append(input_ids[0])
        unpadded_lens.append(input_ids.shape[1])

    num_lpads = None
    lpad_lens = None
    if args.gen_left_pad:
        l = max(unpadded_lens)
        lpad_lens = [l] * len(unpadded_lens)
        num_lpads = []
        for i, tii in enumerate(t_input_ids):
            t_input_ids[i] = torch.tensor(
                [tokenizer._tokenizer.pad_token_id] * (l - len(tii)) + tii.tolist()
            )
            num_lpads.append(l - len(tii))

    t_input_ids = pad_sequence(t_input_ids, batch_first=True).cuda()
    unpadded_lens = torch.tensor(unpadded_lens).cuda()
    if args.gen_left_pad:
        num_lpads = torch.tensor(num_lpads).cuda()
        lpad_lens = torch.tensor(lpad_lens).cuda()
    else:
        lpad_lens = unpadded_lens
    if torch.distributed.get_rank() == 0:
        print(f'input_ids 1 {t_input_ids} {lpad_lens} ')
    t_input_ids = torch.cat(
        [
            t_input_ids,
            torch.full((len(args.prompts), args.max_new_tokens),
                       tokenizer._tokenizer.pad_token_id).cuda()
        ],
        dim=1
    ).cuda()
    return t_input_ids, unpadded_lens, lpad_lens, num_lpads


def gen(model):
    args = get_args()
    tokenizer = get_tokenizer()
    torch.set_printoptions(precision=4, sci_mode=False)

    _input_ids, _, lpad_lens, num_lpads = get_input_ids()
    temperature = 1.
    top_k = 1
    top_p = 0.
    top_p_decay = 0.
    top_p_bound = 0.

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        # inference_max_requests=args.inference_max_requests,
        inference_max_seq_length=args.inference_max_seq_length,
    )

    inference_context = StaticInferenceContext.from_config(inference_wrapper_config)

    for tryi in range(1):
        input_ids = _input_ids.clone().detach()
        t0 = time.time()
        tokens, generated_sequence_lengths, output_log_probs, _ = generate_tokens_probs_and_return_on_first_stage(
            model,
            inference_context,
            ForwardStep,
            input_ids,
            lpad_lens,
            return_output_log_probs=False,
            top_k=top_k,
            top_p=top_p,
            top_p_decay=top_p_decay,
            top_p_bound=top_p_bound,
            temperature=temperature,
            use_eod_token_for_early_termination=True,
            stop_on_double_eol=False,
            stop_on_eol=False,
            prevent_newline_after_colon=False
        )
        t1 = time.time()
        if output_log_probs is not None:
            assert output_log_probs.shape == (tokens.shape[0], tokens.shape[1] - 1)
        assert tokens.shape[1] <= input_ids.shape[1]

        if torch.distributed.get_rank() == 0:
            print(f"generate tokens {tokens} {generated_sequence_lengths} {output_log_probs}")
        tokens = tokens.tolist()

        for i in range(len(tokens)):
            tokens[i] = tokens[i][:generated_sequence_lengths[i]]

        texts = tokenizer._tokenizer.batch_decode(tokens, skip_special_tokens=False)
        if torch.distributed.get_rank() == 0:
            # j = {
            #     'time elapsed': t1 - t0,
            #     'texts': texts,
            #     'input_ids': _input_ids,
            #     'tokens': tokens,
            # }
            for ti, text in enumerate(texts):
                print(f'TI {ti}\n' + '-' * 80 + f'\n^{text}$\n' + '-' * 80)
        torch.distributed.barrier()


def main():
    init_gpatch_for_mcore()
    args = parse_args(gen_args)
    args = validate_args(args)
    set_global_variables(args, build_tokenizer=True)
    args = get_args()
    _initialize_distributed(get_embedding_ranks=None, get_position_embedding_ranks=None)
    _set_random_seed(args.seed, args.data_parallel_random_init)
    fused_kernels.load(args)
    torch.distributed.barrier()

    model = load_model()
    torch.distributed.barrier()
    gen(model)


if __name__ == "__main__":
    main()
