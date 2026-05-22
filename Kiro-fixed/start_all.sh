#!/bin/bash
# ============================================================
# ChatWiki 一键启动脚本
# 启动顺序：BGE-M3 Embedding → ChatWiki API
# 使用：bash start_all.sh
# ============================================================

set -e

LOG_DIR="/var/log/chatwiki"
mkdir -p $LOG_DIR

# ============================================================
# 1. 启动 BGE-M3 Embedding 服务
# ============================================================
echo "═══════════════════════════════════════"
echo "[1/2] 启动 BGE-M3 Embedding 服务..."
echo "═══════════════════════════════════════"

# 检查是否已经在跑
if lsof -i:8001 > /dev/null 2>&1; then
    echo "  ⚠️  端口 8001 已被占用，BGE 可能已在运行，跳过"
else
    nohup vllm serve /root/embedding_model/bge-m3 \
        --port 8001 \
        --api-key qwertyuiop1A. \
        --gpu-memory-utilization 0.7 \
        --task embed \
        > $LOG_DIR/bge.log 2>&1 &

    echo "  BGE PID: $!"
    echo "  日志: $LOG_DIR/bge.log"
    echo "  等待 BGE 加载模型（约30-60秒）..."

    # 等待 BGE 就绪
    for i in $(seq 1 60); do
        if curl -s http://localhost:8001/health > /dev/null 2>&1; then
            echo "  ✅ BGE-M3 已就绪 (${i}s)"
            break
        fi
        sleep 1
        if [ $i -eq 60 ]; then
            echo "  ⚠️  BGE 60秒内未就绪，继续启动 API（可能需要更久）"
        fi
    done
fi

echo ""

# ============================================================
# 2. 启动 ChatWiki API 服务
# ============================================================
echo "═══════════════════════════════════════"
echo "[2/2] 启动 ChatWiki API 服务..."
echo "═══════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if lsof -i:8686 > /dev/null 2>&1; then
    echo "  ⚠️  端口 8686 已被占用，API 可能已在运行，跳过"
else
    cd "$SCRIPT_DIR"
    nohup python api_server.py > $LOG_DIR/api.log 2>&1 &

    echo "  API PID: $!"
    echo "  日志: $LOG_DIR/api.log"

    sleep 3
    if lsof -i:8686 > /dev/null 2>&1; then
        echo "  ✅ ChatWiki API 已启动"
    else
        echo "  ⚠️  API 可能还在初始化，请查看日志"
    fi
fi

echo ""

# ============================================================
# 完成
# ============================================================
echo "═══════════════════════════════════════"
echo "✅ 所有服务已启动"
echo ""
echo "  BGE-M3 Embedding:  http://10.7.10.102:8001"
echo "  ChatWiki API:      http://xx.x.x.xx:8686"
echo "  API 文档:          http://xx.x.x.xx:8686/docs"
echo ""
echo "  查看日志:"
echo "    tail -f $LOG_DIR/bge.log"
echo "    tail -f $LOG_DIR/api.log"
echo ""
echo "  停止所有服务:"
echo "    bash $(dirname "$0")/stop_all.sh"
echo "═══════════════════════════════════════"
