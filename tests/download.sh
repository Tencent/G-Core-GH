export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

DFS=$PWD/hf-hub

repo_id=ma-xu/fine-t2i
hf download --repo-type dataset $repo_id synthetic_enhanced_prompt_random_resolution/train-000000.tar --local-dir $DFS/$repo_id

repo_id=google/byt5-small
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-Embedding-8B
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-Coder-30B-A3B-Instruct
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen2.5-7B
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=laion/CLIP-ViT-H-14-laion2B-s32B-b79K
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=xswu/HPSv2
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=MizzenAI/HPSv3
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=black-forest-labs/FLUX.1-dev
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-VL-30B-A3B-Instruct
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen2.5-VL-72B-Instruct
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen2.5-VL-3B-Instruct
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-VL-2B-Instruct
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-VL-4B-Instruct
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen2.5-Math-1.5B
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

# hubery 会删除这个的，先这样子。
repo_id=bghira/Glyph-SDXL-v2
hf download --repo-type space $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-VL-30B-A3B-Instruct
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-VL-235B-A22B-Instruct
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-0.6B
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-1.7B
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-8B
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=stable-diffusion-v1-5/stable-diffusion-v1-5
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=xzuyn/pickapic_v2_only_some
hf download --repo-type dataset $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-VL-8B-Instruct
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

# dpo multimodal data
repo_id=llamafactory/RLHF-V
save_dir=hf-hub/$repo_id
hf download --repo-type dataset $repo_id --local-dir $save_dir

# for r3 test
repo_id=Qwen/Qwen2.5-Math-7B
save_dir=hf-hub/$repo_id
hf download --repo-type model $repo_id --local-dir $save_dir

repo_id=Qwen/Qwen3-30B-A3B-Thinking-2507
save_dir=hf-hub/$repo_id
hf download --repo-type model $repo_id --local-dir $save_dir

repo_id=DaertML/gsm8k-jsonl
save_dir=hf-hub/$repo_id
hf download --repo-type dataset $repo_id --local-dir $save_dir

repo_id=openai/gsm8k
hf download --repo-type dataset $repo_id --local-dir $DFS/$repo_id

repo_id=AI-MO/NuminaMath-CoT
hf download --repo-type dataset $repo_id --local-dir $DFS/$repo_id

bash tasks/math_rl_v3/qwen/preprocess_data.sh

repo_id=inclusionAI/LLaDA-MoE-7B-A1B-Base
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3-30B-A3B
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=Qwen/Qwen3.6-35B-A3B
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=deepseek-ai/DeepSeek-V4-Flash
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=deepseek-ai/DeepSeek-V4-Pro
hf download --repo-type model $repo_id --local-dir $DFS/$repo_id

repo_id=tatsu-lab/alpaca
hf download --repo-type dataset $repo_id --local-dir $DFS/$repo_id

repo_id=yifengzhu-hf/LIBERO-datasets
hf download --repo-type dataset $repo_id --local-dir $DFS/$repo_id

# 闭源，从 private repo 拉取

# 从腾讯内部这两个文档获取 username / token：
# 1. https://mirrors.tencent.com/#/private/generic2/detail?repo_name=wepsdlpriv （点击“获取访问 token”）
# 2. https://iwiki.woa.com/p/17512393
username=$1
token=$2

if [[ $token == "" ]]; then
  echo "no tx generic repo token provided"
  exit 0
fi

rm generic-downloader
wget https://mirrors.tencent.com/repository/generic/mirrors/tools/generic-downloader
chmod a+x generic-downloader

# oteam4-4 rl data
tdir='hf-hub/wechat/data'
echo "downloading $tdir"
./generic-downloader -r wepsdlpriv/test_data/$tdir -d $tdir -u $username -p $token

# oteam4-3
oteam4_3_dir=hf-hub/wechat/oteam4_3
tdir=$oteam4_3_dir
echo "downloading $tdir"
./generic-downloader -r wepsdlpriv/test_data/$tdir -d $tdir -u $username -p $token


# oteam4-4
# 对应 /mnt/shenzhen2cephfs/xinhangleng/models/oteam_models/oteam44/1024/sft/1104_sft1024_cluster30w_qwennimageinfer_lr1e-5_gb128/step-10000
oteam4_4_dir=hf-hub/wechat/oteam4_4-step-10000
tdir=$oteam4_4_dir
echo "downloading $tdir"
./generic-downloader -r wepsdlpriv/test_data/$tdir -d $tdir -u $username -p $token
ln -s ${PWD}/hf-hub/wechat/oteam4_3/vae $oteam4_4_dir
ln -s ${PWD}/hf-hub/Qwen/Qwen2.5-7B $oteam4_4_dir
ln -s ${PWD}/hf-hub/Qwen/Qwen3-Embedding-8B $oteam4_4_dir
mkdir -p $oteam4_4_dir/glyph_byt5_v2/glyph
ln -s ${PWD}/hf-hub/google/byt5-small $oteam4_4_dir/glyph_byt5_v2
ln -s ${PWD}/hf-hub/bghira/Glyph-SDXL-v2/checkpoints/glyph-sdxl_multilingual_10-lang $oteam4_4_dir/glyph_byt5_v2/glyph
ln -s ${PWD}/hf-hub/bghira/Glyph-SDXL-v2/assets $oteam4_4_dir/glyph_byt5_v2/glyph

# oteam4-4
# 对应 /mnt/shenzhen2cephfs/xinhangleng/models/oteam_models/oteam44/1024/sft/1112_sft1024_cluster30w_text10w_human5w_geneval2w_seed4infer_lr1e-5_gb128/step-60000
oteam4_4_dir=hf-hub/xinhangleng/models/oteam_models/oteam44/1024/sft/1112_sft1024_cluster30w_text10w_human5w_geneval2w_seed4infer_lr1e-5_gb128/step-60000
tdir=$oteam4_4_dir
echo "downloading $tdir"
./generic-downloader -r wepsdlpriv/test_data/$tdir -d $tdir -u $username -p $token
ln -s ${PWD}/hf-hub/wechat/oteam4_3/vae $oteam4_4_dir
ln -s ${PWD}/hf-hub/Qwen/Qwen2.5-7B $oteam4_4_dir
ln -s ${PWD}/hf-hub/Qwen/Qwen3-Embedding-8B $oteam4_4_dir
mkdir -p $oteam4_4_dir/glyph_byt5_v2/glyph
ln -s ${PWD}/hf-hub/google/byt5-small $oteam4_4_dir/glyph_byt5_v2
ln -s ${PWD}/hf-hub/bghira/Glyph-SDXL-v2/checkpoints/glyph-sdxl_multilingual_10-lang $oteam4_4_dir/glyph_byt5_v2/glyph
ln -s ${PWD}/hf-hub/bghira/Glyph-SDXL-v2/assets $oteam4_4_dir/glyph_byt5_v2/glyph

# 特殊处理，应该只是 chenxu 或者 xinhang 为了方便改了路径，我给他绕回来吧。
mkdir $oteam4_4_dir/transformer
ln -s `realpath $oteam4_4_dir/diffusion_pytorch_model.bin` $oteam4_4_dir/transformer
cp $oteam4_4_dir/transformer_ema/config.json $oteam4_4_dir/transformer

# 多模态数据
geo3k_gen_dir=hf-hub/wechat/geo3k_gen_16_sample
tdir=$geo3k_gen_dir
./generic-downloader -r wepsdl/test_data/$tdir -d $tdir -u $username -p $token

geo3k_gen_dir=hf-hub/wechat/geometry3k_16_samples
tdir=$geo3k_gen_dir
./generic-downloader -r wepsdl/test_data/$tdir -d $tdir -u $username -p $token

# wegen hpsv3
wegen_hpsv3_dir=hf-hub/wechat/wegen_hpsv3
tdir=$wegen_hpsv3_dir
echo "downloading $tdir"
./generic-downloader -r wepsdlpriv/test_data/$tdir -d $tdir -u $username -p $token

# oteam4-4 40000 step
oteam4_4_dir=hf-hub/wechat/oteam4_4-step-40000
tdir=$oteam4_4_dir
echo "downloading $tdir"
./generic-downloader -r wepsdlpriv/test_data/$tdir -d $tdir -u $username -p $token
ln -s ${PWD}/hf-hub/wechat/oteam4_3/vae $oteam4_4_dir
ln -s ${PWD}/hf-hub/Qwen/Qwen2.5-7B $oteam4_4_dir
ln -s ${PWD}/hf-hub/Qwen/Qwen3-Embedding-8B $oteam4_4_dir
mkdir -p $oteam4_4_dir/glyph_byt5_v2/glyph
ln -s ${PWD}/hf-hub/google/byt5-small $oteam4_4_dir/glyph_byt5_v2
ln -s ${PWD}/hf-hub/bghira/Glyph-SDXL-v2/checkpoints/glyph-sdxl_multilingual_10-lang $oteam4_4_dir/glyph_byt5_v2/glyph
ln -s ${PWD}/hf-hub/bghira/Glyph-SDXL-v2/assets $oteam4_4_dir/glyph_byt5_v2/glyph

# welmv4 moe 80b
# rsync -av /mnt/ceph-nj2-csp/mm-base-plt2/group7/user_marcoszhu/hz/model/welm_v4_instruct hf-hub/wechat
tdir=hf-hub/wechat/welm_v4_instruct
echo "downloading $tdir"
./generic-downloader -r wepsdlpriv/test_data/$tdir -d $tdir -u $username -p $token

# rsync -av /mnt/ceph-nj2-csp/mm-base-plt2/group7/user_marcoszhu/hz/model/welm_v4_80A3B_128k_20251015 hf-hub/wechat
tdir=hf-hub/wechat/welm_v4_80A3B_128k_20251015
echo "downloading $tdir"
./generic-downloader -r wepsdlpriv/test_data/$tdir -d $tdir -u $username -p $token

# rsync -av /mnt/ceph-sg1-csp/mmsearch-luban-universal/group_7/user_kodazhang/oriontian_0407_part1_2_mix_sft_hf/epoch_008_step_0000240 .
tdir=hf-hub/wechat/oriontian_0407_part1_2_mix_sft_hf/epoch_008_step_0000240
echo "downloading $tdir"
./generic-downloader -r wepsdlpriv/test_data/$tdir -d $tdir -u $username -p $token
