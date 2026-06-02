#!/bin/bash
# ============================================================
# ChatWiki 服务器重启后一键全部拉起脚本
#
# 适用场景：服务器（c01-rdai02）重启 / 断电恢复后，
#           一条命令把整条链路按正确顺序拉起来：
#           Milvus(docker) → BGE-M3(vllm) → ChatWiki API
#
# 使用：bash restart_all.sh
#
# 注意：DeepSeek(10.0.6.89) 和 同事RAG(10.0.6.88) 在别的机器上，
#       本脚本不负责，它们由对应机器各自维护。
# ============================================================

set +e   # 单步失败不中断，继续往下走

# ---------- 配置（按实际环境，如有变化在此修改）----------
PROJECT_DIR="/root/BYX_F/Kiro-fixed"
CONDA_BIN="/root/anaconda3/envs/byx_1/bin"
MILVUS_COMPOSE_DIR="/root/BYX/milvus-offline"
BGE_MODEL="/root/embedding_model/bge-m3"
BGE_PORT=8001
API_PORT=7002
LOG_DIR="/var/log/chatwiki"
# --------------------------------------------------------

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════"
echo "  ChatWiki 全服务一键拉起"
echo "═══════════════════════════════════════════════════"

# ============================================================
# 1. 启动 Milvus（docker 容器）
# ============================================================
echo ""
echo "[1/3] 启动 Milvus（docker）..."
echo "───────────────────────────────────────────────────"

if docker ps --format '{{.Names}}' | grep -q "^milvus-standalone$"; then
    echo "  ✅ Milvus 已在运行，跳过"
else
    # 优先用 docker compose 拉起（最稳，依赖 etcd/minio 顺序由 compose 管理）
    if [ -f "$MILVUS_COMPOSE_DIR/docker-compose.yml" ]; then
        echo "  使用 docker compose 启动..."
        (cd "$MILVUS_COMPOSE_DIR" && (docker compose up -d 2>/dev/null || docker-compose up -d))
    else
        echo "  compose 文件未找到，尝试直接 start 容器..."
        docker start milvus-etcd milvus-minio milvus-standalone milvus-attu 2>/dev/null
    fi

    echo "  等待 Milvus 就绪（最多 90 秒）..."
    for i in $(seq 1 90); do
        if curl -s http://localhost:9091/healthz 2>/dev/null | grep -qi "ok"; then
            echo "  ✅ Milvus 已就绪 (${i}s)"
            break
        fi
        sleep 1
        if [ "$i" -eq 90 ]; then
            echo "  ⚠️  Milvus 90秒内未就绪，请手动检查: docker logs milvus-standalone"
        fi
    done
fi

# ============================================================
# 2. 启动 BGE-M3 Embedding（vllm）
# ============================================================
echo ""
echo "[2/3] 启动 BGE-M3 Embedding（vllm, 端口 $BGE_PORT）..."
echo "───────────────────────────────────────────────────"

if lsof -i:$BGE_PORT > /dev/null 2>&1; then
    echo "  ✅ 端口 $BGE_PORT 已占用，BGE 可能已在运行，跳过"
else
    # setsid 让进程脱离当前终端，SSH 断开也不会被杀
    setsid nohup "$CONDA_BIN/vllm" serve "$BGE_MODEL" \
        --port $BGE_PORT \
        --api-key qwertyuiop1A. \
        --gpu-memory-utilization 0.7 \
        --task embed \
        > "$LOG_DIR/bge.log" 2>&1 &

    echo "  BGE 启动中，日志: $LOG_DIR/bge.log"
    echo "  等待 BGE 加载模型（最多 120 秒）..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:$BGE_PORT/health > /dev/null 2>&1; then
            echo "  ✅ BGE-M3 已就绪 (${i}s)"
            break
        fi
        sleep 1
        if [ "$i" -eq 120 ]; then
            echo "  ⚠️  BGE 120秒内未就绪，请查看日志: tail -f $LOG_DIR/bge.log"
        fi
    done
fi

# ============================================================
# 3. 启动 ChatWiki API
# ============================================================
echo ""
echo "[3/3] 启动 ChatWiki API（端口 $API_PORT）..."
echo "───────────────────────────────────────────────────"

if lsof -i:$API_PORT > /dev/null 2>&1; then
    echo "  ✅ 端口 $API_PORT 已占用，API 可能已在运行，跳过"
else
    cd "$PROJECT_DIR" || { echo "  ❌ 项目目录不存在: $PROJECT_DIR"; exit 1; }
    # setsid 让进程脱离终端，SSH 断开/电脑关机都不影响
    setsid nohup "$CONDA_BIN/python" api_server.py > "$LOG_DIR/api.log" 2>&1 &

    echo "  API 启动中，日志: $LOG_DIR/api.log"
    echo "  等待 API 就绪（最多 60 秒）..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:$API_PORT/api/health > /dev/null 2>&1; then
            echo "  ✅ ChatWiki API 已就绪 (${i}s)"
            break
        fi
        sleep 1
        if [ "$i" -eq 60 ]; then
            echo "  ⚠️  API 60秒内未就绪，请查看日志: tail -f $LOG_DIR/api.log"
        fi
    done
fi

# ============================================================
# 健康总检查
# ============================================================
echo ""
echo "═══════════════════════════════════════════════════"
echo "  健康检查"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Milvus  : $(curl -s http://localhost:9091/healthz 2>/dev/null || echo '无响应')"
echo ""
echo "  ChatWiki API /api/health:"
curl -s http://localhost:$API_PORT/api/health 2>/dev/null || echo "  无响应"
echo ""
echo ""
echo "  服务地址："
echo "    BGE-M3:        http://10.7.10.102:$BGE_PORT"
echo "    ChatWiki API:  http://10.7.10.102:$API_PORT"
echo "    API 文档:      http://10.7.10.102:$API_PORT/docs"
echo ""
echo "  查看日志："
echo "    tail -f $LOG_DIR/bge.log"
echo "    tail -f $LOG_DIR/api.log"
echo "═══════════════════════════════════════════════════"
echo "✅ 全部拉起完成。请确认上方 /api/health 中"
echo "   milvus / embedding 均为 true（llm 需对端 10.0.6.89 在线）"
echo "═══════════════════════════════════════════════════"
