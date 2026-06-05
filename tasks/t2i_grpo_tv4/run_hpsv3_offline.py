import os
import os.path
import shutil
import uuid
from typing import Any, Dict, List

import torch
import numpy as np
from typing_extensions import override

from hpsv3 import HPSv3RewardInferencer

if __name__ == '__main__':
    model = HPSv3RewardInferencer(
        device='cuda',
        config_path='tasks/t2i_grpo_tv4/yaml/HPSv3_7B.yaml',
        checkpoint_path="hf-hub/MizzenAI/HPSv3/HPSv3.safetensors"
    )

    prompts = [
        'Colin Farrell depicts a realistic-looking Batman, posing in a masculine manner amidst a dark, fractal background.',
        'Littlest Pet Shop cat in a matte painting from Fantasia (1941).',
        'A digital painting of an Aztec empress in sharp focus, portrayed as a fantasy portrait in concept art style.',
        'a bath room with a stand up shower and a bath tub',
        'A painting of a monkey wearing gold headphones and sunglasses looking up at a starry night sky.',
        'Their is a toilet next to an opaque window.',
        'An artwork from Dan Mumford collection featuring a mage invoking divine gods during a storm with lightnings.',
        'A restroom with wood paneling on the wall',
    ]

    mean_rewards = []

    for idx, prompt in enumerate(prompts):
        # d = '../Dit_training_tool/output-oteam44-step-10000-cfg-1'
        # d = '../Dit_training_tool/output-oteam44-rl-step-512-cfg-1'
        d = '../Dit_training_tool/output-oteam44-step-10000-cfg-3.5'
        path = os.path.join(d, f'{idx:03d}_{prompt[:30]}.jpg')

        rewards = model.reward([path], [prompt])
        reward = rewards[0][0].item()
        mean_rewards.append(reward)
        print(f'{idx=} {prompt=} {path=} {reward}')

    print(np.mean(np.array(mean_rewards)))
