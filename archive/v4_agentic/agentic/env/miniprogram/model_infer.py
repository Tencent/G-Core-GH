import hashlib
import hmac
import time

import openai

SOURCE = "miniprogram"
APPID = "appaotzt3ne25aag7bl"
APPKEY = "jtVLgvnpXPTuyhYikuvdMwTEEVFaXZvY"
BASE_URL = "http://ichat.woa.com/api/external"


def calcAuthorization(source, appkey):
    timestamp = int(time.time())
    signStr = "x-timestamp: %s\nx-source: %s" % (timestamp, source)
    sign = hmac.new(appkey.encode('utf-8'), signStr.encode('utf-8'),
                    hashlib.sha256).digest()
    return sign.hex(), timestamp


def get_auth_headers():
    auth, timestamp = calcAuthorization(SOURCE, APPKEY)
    headers = {
        "X-AppID": APPID,
        "X-Source": SOURCE,
        "X-Timestamp": str(timestamp),
        "X-Authorization": auth
    }
    return headers


def model_infer(messages, model_name):
    client = openai.OpenAI(
        api_key="EMPTY",
        # base_url="http://29.213.203.228:8000/v1",
        base_url="http://llmproxy-offline.lubanllm.polaris:9000/v1/test")
    response = client.chat.completions.create(
        # model="eval_kangyuqiao-Qwen2.5-VL-72B-Instruct-1123-11",
        model=model_name,
        messages=messages,
        temperature=0.0,
        seed=2025,
        max_tokens=2048,
        timeout=60,
    )
    return response.choices[0].message.content


def model_infer_gemini(messages, model_name):
    client = openai.OpenAI(api_key="ichat", base_url=BASE_URL)

    for attempt in range(3):
        headers = get_auth_headers()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                timeout=180,  # 180? 300?
                max_tokens=4096,
                temperature=0.7,
                seed=2025,
                extra_headers=headers,
                extra_body={"cid": "hess"}  #注意！！写自己的 rtx
            )
            text_content = response.choices[0].message.content
            if text_content != "":
                return text_content
            else:
                print(f"model infer err {attempt}")
                continue
        except Exception as e:
            print(f"model infer err {attempt}：{str(e)}")
            continue
    return ""
