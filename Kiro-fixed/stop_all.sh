#!/bin/bash
# ============================================================
# ChatWiki 一键停止脚本
# 使用：bash stop_all.sh
# ============================================================

echo "停止 ChatWiki API (port 8686)..."
kill $(lsof -t -i:8686) 2>/dev/null && echo "  ✅ API 已停止" || echo "  ⚠️  API 未在运行"

echo "停止 BGE-M3 (port 8001)..."
kill $(lsof -t -i:8001) 2>/dev/null && echo "  ✅ BGE 已停止" || echo "  ⚠️  BGE 未在运行"

echo ""
echo "✅ 所有服务已停止"
