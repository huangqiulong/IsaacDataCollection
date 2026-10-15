#!/usr/bin/env bash
# EgoSuite-Open100K（Lightwheel）Hugging Face Bucket 下载助手
#
# 前置条件：
#   1) 已安装新版 huggingface_hub 并登录：  hf auth login
#      且账号已通过 LightwheelAI/EgoStandard 的访问审批（HF 页面申请）。
#   2) 走代理访问 huggingface.co（默认 socks5h://127.0.0.1:2080，
#      可用环境变量 EGOSUITE_PROXY 覆盖；hf-mirror 不支持 Bucket）。
#
# 用法：
#   bash egosuite_download.sh list                      # 列出顶层目录
#   bash egosuite_download.sh list "EgoStandard/mcap"   # 列出任意前缀
#   bash egosuite_download.sh task "Adjust Seal Sticker"  # 列出某任务全部 mcap 及大小
#   bash egosuite_download.sh get "EgoStandard/mcap/<任务>/<文件>.mcap" <本地目录>
#   bash egosuite_download.sh sync "EgoStandard/mcap/<任务>" <本地目录>   # 整任务同步（很大，慎用）
set -euo pipefail

REPO="LightwheelAI/EgoStandard"
PROXY="${EGOSUITE_PROXY:-socks5h://127.0.0.1:2080}"
HF="${HF_CLI:-$HOME/miniconda3/envs/lerobot/bin/hf}"

if ! command -v "$HF" >/dev/null 2>&1 && [ ! -x "$HF" ]; then
    echo "找不到 hf CLI（可用 HF_CLI 环境变量指定路径）" >&2
    exit 1
fi

run_hf() {
    env -u HF_ENDPOINT HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY" "$HF" "$@"
}

cmd="${1:-}"
case "$cmd" in
    list)
        prefix="${2:-}"
        if [ -n "$prefix" ]; then
            run_hf buckets list "$REPO/$prefix" -h
        else
            run_hf buckets list "$REPO" -h
        fi
        ;;
    task)
        task="${2:?请给出任务名，例如 \"Adjust Seal Sticker\"}"
        run_hf buckets list "$REPO/EgoStandard/mcap/$task" -R -h
        ;;
    get)
        bucket_path="${2:?请给出桶内路径，例如 EgoStandard/mcap/<任务>/<文件>.mcap}"
        local_dir="${3:?请给出本地目录}"
        mkdir -p "$local_dir"
        run_hf buckets cp "hf://buckets/$REPO/$bucket_path" "$local_dir/"
        ;;
    sync)
        bucket_path="${2:?请给出桶内前缀}"
        local_dir="${3:?请给出本地目录}"
        mkdir -p "$local_dir"
        run_hf buckets sync "hf://buckets/$REPO/$bucket_path" "$local_dir"
        ;;
    *)
        sed -n '1,20p' "$0"
        exit 1
        ;;
esac
