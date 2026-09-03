# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""DSV4 chat-template constant.

Kept in its own module (no third-party imports) so that the alignment test
in ``tests/test_gfused/test_dsv4_chat_template.py`` can import the single
source of truth without dragging in ``megatron`` / ``transformers``.

The contract this template implements is documented inline below and is
enforced against the official DSV4 encoder by the alignment test.
"""

# DSV4-Flash/Pro tokenizer ships without a `chat_template` field, so we
# inject one that mirrors
# ``gpatch_v4.models.deepseek_v4.encoding_dsv4.encode_messages``. The full alignment
# contract (text + token ids, chat / thinking, single / multi-turn, with /
# without system) is exercised by
# ``tests/test_gfused/test_dsv4_chat_template.py`` -- if you touch this
# string, run that file before merging.
#
# Special tokens (full-width bars U+FF5C and U+2581 are mandatory -- ASCII
# '|' tokenizes differently):
#   BOS       = `<｜begin▁of▁sentence｜>` (id 0)
#   EOS       = `<｜end▁of▁sentence｜>`   (id 1)
#   <｜User｜>      = id 128803
#   <｜Assistant｜> = id 128804
#   <think>   = id 128821
#   </think>  = id 128822
#
# Per oracle (``drop_thinking=True``, the default):
#   * BOS at the very start of the conversation;
#   * system content prepended verbatim, no role wrapper;
#   * every user turn:       <｜User｜>{content}
#   * historical assistant:  <｜Assistant｜></think>{content}<｜EOS｜>
#   * trailing assistant in thinking mode:
#                            <｜Assistant｜><think></think>{content}<｜EOS｜>
#   * trailing assistant in chat mode:
#                            <｜Assistant｜></think>{content}<｜EOS｜>
#   * add_generation_prompt: <｜Assistant｜>{<think> | </think>}
DSV4_CHAT_TEMPLATE = (
    # --- BOS + optional system prefix ---
    "{{ '<\uff5cbegin\u2581of\u2581sentence\uff5c>' }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    # --- think token selector (chat vs thinking) ---
    "{% set think_open = '<think>' %}"
    "{% set think_close = '</think>' %}"
    "{% if enable_thinking is defined and enable_thinking %}"
    "{% set tail_think = think_open %}"
    "{% else %}"
    "{% set tail_think = think_close %}"
    "{% endif %}"
    # --- main loop ---
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<\uff5cUser\uff5c>' + message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    # historical assistant turns are always rendered with </think>
    # (drop_thinking=True); only the *trailing* assistant turn in thinking
    # mode keeps the leading <think> + </think> pair.
    "{% if loop.last and not add_generation_prompt"
    " and (enable_thinking is defined and enable_thinking) %}"
    "{{ '<\uff5cAssistant\uff5c>' + think_open + think_close"
    " + message['content']"
    " + '<\uff5cend\u2581of\u2581sentence\uff5c>' }}"
    "{% else %}"
    "{{ '<\uff5cAssistant\uff5c>' + think_close"
    " + message['content']"
    " + '<\uff5cend\u2581of\u2581sentence\uff5c>' }}"
    "{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    # --- generation prompt ---
    "{% if add_generation_prompt %}"
    "{{ '<\uff5cAssistant\uff5c>' + tail_think }}"
    "{% endif %}"
)
