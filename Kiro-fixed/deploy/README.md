# 部署指南

## 一键部署（开机自启）

### 1. 复制服务文件到系统目录

```bash
sudo cp deploy/chatwiki-bge.service /etc/systemd/system/
sudo cp deploy/chatwiki-api.service /etc/systemd/system/
```

### 2. 修改路径（按实际情况改）

编辑 `/etc/systemd/system/chatwiki-api.service`：
- 把 `WorkingDirectory=/path/to/Kiro-fixed` 改成你的实际路径

编辑 `/etc/systemd/system/chatwiki-bge.service`：
- 确认 `vllm` 的路径（用 `which vllm` 查看）

### 3. 启用并启动

```bash
sudo systemctl daemon-reload

# 设为开机自启
sudo systemctl enable chatwiki-bge
sudo systemctl enable chatwiki-api

# 立即启动
sudo systemctl start chatwiki-bge
sudo systemctl start chatwiki-api
```

### 4. 查看状态

```bash
sudo systemctl status chatwiki-bge
sudo systemctl status chatwiki-api
```

### 5. 查看日志

```bash
journalctl -u chatwiki-bge -f    # BGE 日志
journalctl -u chatwiki-api -f    # API 日志
```

### 6. 停止/重启

```bash
sudo systemctl stop chatwiki-api
sudo systemctl restart chatwiki-bge
```

## 手动启动（不需要开机自启时）

```bash
bash start_all.sh   # 启动
bash stop_all.sh    # 停止
```
