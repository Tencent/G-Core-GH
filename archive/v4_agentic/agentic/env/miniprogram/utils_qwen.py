import base64
import os
import sys
from datetime import datetime, timedelta, timezone

import openai
import requests

from .model_infer import model_infer, model_infer_gemini


def image_to_base64(image_path):
    try:
        if image_path.startswith(('http://', 'https://')):
            # 处理URL情况
            response = requests.get(image_path)
            response.raise_for_status()  # 检查请求是否成功
            encoded_string = base64.b64encode(response.content)
            return encoded_string.decode('utf-8')
        else:
            # 处理本地文件情况
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                return encoded_string.decode('utf-8')
    except FileNotFoundError:
        print(f"错误：未找到指定的图片文件: {image_path}", file=sys.stderr)
    except requests.exceptions.RequestException as e:
        print(f"错误：无法从URL获取图片: {str(e)}", file=sys.stderr)
        print(f"错误URL: {image_path}", file=sys.stderr)
    except Exception as e:
        print(f"错误：发生了未知错误: {str(e)}", file=sys.stderr)
        print(f"错误路径: {image_path}", file=sys.stderr)


def qwen72b_inference(input_img_paths, model_name, inst, subgoals, thinks):
    input_imgs = input_img_paths[-3:]
    try:
        prompt_sample = """
你是一位评估 GUI 代理任务轨迹的专家。你的任务是评估 GUI 操作任务轨迹的质量和有效性。一个轨迹包含以下组件：
1. **用户指令**：描述用户的预期任务，包括用户的目标及具体要求（包含时间、地点、数量、规格、价格等约束）。
    - 地点：例如“在北京市”，“广州，“附近”，“北京前门西街餐厅”。
    - 时间：例如“明天”，“今晚”，“周五”，“周六下午15:00-18:00”。
    - 数量：例如“1件”，“2个”，“3张”。
    - 规格：例如“冰”，“超大杯”。
    - 价格：例如“100元以下”，“50-100元”。
    - 其他：例如“最热”，“最新”。
2. **动作历史**：由代理执行的一系列动作.
3. **GUI代理的思考**：GUI代理执行最后一步动作的思考内容。
4. **GUI截图**：最近3张GUI截图：初始界面和在执行完每一步动作后的界面截图选取最近三张（从上到下依次排列）。

在评估轨迹时，请考虑以下关键方面：
当前日期：今天是{date}，在评估时注意判断用户指令中的时间要求是否被准确满足。

任务完成的判断标准：
- 如果是点单/下单/预定/订票/充值类任务，需同时满足以下条件方可视为完成：
1.最后一张页面为订单填写页面（且指令中提到的所有必填信息已完整填写）、订单确认页面、支付确认页面、订单结算页面、支付二维码页面，或包含支付按钮的最终界面；
2.注意：指令中未提到的信息不允许随意填写。
3.所下单的商品或服务内容必须与任务要求完全一致，数量不得多出或缺少（例如：任务要求购买1件商品A，实际下单为2件商品A则视为未完成；任务要求购买1件商品A，实际下单为1件商品A、1件商品B则视为未完成）；
- 如果是查询类任务，最后一张页面需要展示查询的信息，否则应视作未完成。
- 如果任务涉及日期，最后一张页面显示日期选择不正确，则任务视作未完成。
- 涉及起点与终点的任务（如公交查询、路线规划等）：最后一张页面呈现的起点、终点及行进方向必须与任务要求完全吻合。例如：任务要求查询“从嘉兴苑到苏州博物馆的换乘路线”，则最后一张页面必须包含“嘉兴苑”和“苏州博物馆”，且路线方向为从嘉兴苑出发前往苏州博物馆，反向则视为未完成。
- 特定任务示例
1.对于任务“用微博小程序看看现在微博热搜榜第一名是什么话题？”，需点击“更多热搜”按钮展开完整榜单，最终页面必须呈现热搜榜单。
- 需请求用户接管的情况（涉及关键信息缺失或敏感隐私）： 若当前页面必须填写或选择关键信息才能继续执行，但指令中未提及该具体信息，或者涉及用户个人敏感隐私，严禁随意填写、虚构数据或默认选择，此时必须判定为“请求用户接管”。具体场景包括但不限于：
1.缺失必要参数： 指令未提及姓名、身份证号、起点或终点等关键信息，但当前页面强制要求填写或选择。
2.涉及敏感隐私/鉴权： 涉及填写手机号、鉴权验密（输入密码/验证码）、具体房屋门牌号等个人隐私信息。

评估标准：
- 轨迹连贯性：低级步骤和相应动作是否遵循朝向目标指令靠近？动作是否清晰描述且具体？是否存在冗余或不必要的动作？
- 任务完成情况：轨迹是否成功完成了指令任务？是否完成了所有必要的交互？错误情况是否得到适当处理？

评分指南：根据评估标准，按 0 到 4 的等级对轨迹进行评分：
- 4: 任务完美完成，成功执行多项动作实现目标。序列逻辑清晰且没有明显冗余。
- 3: 任务完成，成功执行多项动作实现目标。但是完成过程存在效率低下，存在动作冗余、重复。
- 2: 任务部分完成，执行了一些成功动作。然而，由于任务或环境限制，目标未完全实现，或者序列以循环或错误结束。或者任务中途已经执行完成，但是执行了多余的操作，导致最后一张页不符合任务完成的标准。
- 1: 仅执行了少量动作。虽然有完成任务的尝试，但轨迹早期偏离目标或在执行和逻辑上表现出显著低效。
- 0: 任务完全失败，开始时没有执行有意义的动作。序列要么立即陷入死锁、重复循环，或在完成任务上没有价值。或者任务完全不可访问。

注意：
- 如果任务相对复杂，但轨迹表现出有价值的尝试，即使任务没有完全完成，也请考虑向上调整分数。然而，如果任务复杂但轨迹未能执行对任务完成有意义贡献的动作，则不应奖励额外分数。
- 请注意动作历史、GUI代理的思考可能会欺骗你，如果动作历史、GUI代理的思考和最近3张GUI截图冲突时，请以最近3张截图为准。
- 请仔细识别截图中订单上物品的内容型号、时间、数量、规格、价格信息。
- 用户指令中的地点，时间，数量，规格，价格等要求全部被满足才能视为任务完成。

您需要根据代理的动作、截图、思考过程综合评估得分。
输出格式：<reason>your thoughts and reasoningprocess for the score</reason><reward>your score from 0-4</reward>
一定不要输出多余内容，直接输出规定格式的答案。"""

        user_sample = """用户指令:{inst}
动作历史: 每一步的动作:{subgoals}
GUI代理的思考:{thinks},
GUI截图：
"""
        beijing_tz = timezone(timedelta(hours=8))
        now = datetime.now(beijing_tz)
        date_info = f"{now.year}年{now.month}月{now.day}日 {['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]}"

        input_imgs_content = [{
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{image_to_base64(img)}'
            }
        } for img in input_imgs]
        messages = [{
            'role': 'system',
            'content': prompt_sample.format(date=date_info)
        }, {
            'role':
            'user',
            'content': [{
                'type':
                'text',
                'text':
                user_sample.format(inst=inst, subgoals=subgoals, thinks=thinks)
            }] + input_imgs_content
        }]

        if model_name in ["gemini-3-flash-preview", "gemini-3-pro-preview"]:
            text = model_infer_gemini(messages, model_name)
        else:
            text = model_infer(messages, model_name)
        return text
    except Exception as e:
        print(f"捕获到异常: {str(e)}", file=sys.stderr)
        return ""
