<div align="center">

YATT: A Scalable, Simple, Efficient and Production Ready Training Library
=====

YATT is a scalable, simple, efficient, and production ready training library.

YATT 是一个 scalable、简单、高效且可用于生产环境的训练库。

内部名：`gcore`

<div align="left">


1. Scalalblity: YATT is built on Megatron, vLLM, and SGLang, providing good scalability.
2. Simplicity: If unfortunately in the worst-case scenario, YATT’s code remains sufficiently simple.
3. Efficiency: Training and inference engine integrations and SOTA RL throughput.
4. Production Ready: YATT has been validated in business scenarios.

---

1. 可扩展性：YATT 构建于 Megatron、vLLM 和 SGLang 之上，具备良好的可扩展性。
2. 简单性：即使在最坏的情况下，YATT 的代码依然足够简单。
3. 效率：训练、推理引擎良好集成，SOTA 吞吐。
4. 生产可用性：YATT 已在实际业务场景中得到验证。

# Keys Features

1. Megatron for LM and VQA RL training
2. vLLM and SGLang for rollout generation
3. Efficient generative rewarding RL training
4. Megatron LoRA support

# Documents

内部可访问：https://gcore.woa.com/docs

You can use Sphinx to build and view the documentation. It’s also fine to read the rst and md files in the docs folder directly.

你可以通过 sphinx build 文档，或者直接阅读 docs 文件夹下的 rst 和 md。

参考 [numpy docstring](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy)。

```bash
readonly WDIR=$PWD
git clone https://github.com/Tencent/Wechat-YATT
git clone git@github.com:NVIDIA/Megatron-LM.git -b core_v0.13.1

pip3 install sphinx-autobuild myst_parser
PYTHONPATH="$PWD:../Megatron-LM:$PYTHONPATH" sphinx-autobuild --port 8080 --host ${__HOST_IP__} docs/source docs/build

# or
PYTHONPATH="$PWD:../Megatron-LM:../mbridge:$PYTHONPATH" sphinx-build -M html docs/source docs/build
python -m http.server -d docs/build/html/ --bind $__HOST_IP__ 8080
```

### Code Formatting

Uses `yapf` (Facebook style, column limit 100) + `isort` via pre-commit hooks:

```bash
pip install pre-commit
git config --global --unset core.hooksPath
pre-commit install
```

Manual format:
```bash
bash gcore-dev/fmt.sh
```

# Env / Docker

We have tested it internally at WXG using tlinux, and it’s also possible to set up a similar environment on Ubuntu. In the docker section of the documentation, we have provided a sample Dockerfile for your reference.

我们在 WXG 内部使用 tlinux 测试过，ubuntu 也可以搭建类似的环境。我们在文档的 ***docker*** 部分贴了供参考的 dockerfile。

# 测试

```bash
# https://mirrors.tencent.com/#/private/generic2/detail?repo_name=wepsdl
bash tests/download.sh $your_rtx $token
bash tests/test.sh
```

# Getting Started

Please refer to the examples section of the documentation. It is recommended to start with Math SFT and Math GRPO.

请查看文档 ***examples*** 部分，建议从 Math SFT 与 Math GRPO 开始。

# Citation

```
@misc{wu2025wechatyattscalablesimpleefficient,
      title={WeChat-YATT: A Scalable, Simple, Efficient, and Production Ready Training Library}, 
      author={Junyu Wu and Weiming Chang and Xiaotao Liu and Guanyou He and Tingfeng Xian and Haoqiang Hong and Boqi Chen and Hongtao Tian and Tao Yang and Yunsheng Shi and Feng Lin and Ting Yao and Jiatao Xu},
      year={2025},
      eprint={2508.07970},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.07970}, 
}
```
