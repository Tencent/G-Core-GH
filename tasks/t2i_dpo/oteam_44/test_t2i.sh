
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"


DIR="$(cd "$( dirname "$0" )" && pwd)"
cd ${DIR}/../../..
CUR_DIR=$(pwd)
pkill -9 -f python
readonly MCORE_PATH="${CUR_DIR}/../../Megatron-LM"
readonly mbridge_path="${CUR_DIR}/../../mbridge"
export PYTHONPATH="${CUR_DIR}:$MCORE_PATH:${mbridge_path}"

MYWD=$PWD
repo_id=runwayml/stable-diffusion-v1-5
model_dir=$MYWD/hf-hub/$repo_id


#export HF_ENDPOINT=https://hf-api.gitee.com
#export HF_HOME=~/.cache/gitee-ai
repo_id=xzuyn/pickapic_v2_only_some
data_dir=$repo_id


python tasks/t2i_dpo/oteam_44/test_t2i_dpo.py \
  --config-dir "./tasks/t2i_dpo/oteam_44" \
  --config-path "./" \
  --config-name "simple_fsdp2" \
  MODEL_NAME=${model_dir} \
  DATASET_NAME=${data_dir} \




