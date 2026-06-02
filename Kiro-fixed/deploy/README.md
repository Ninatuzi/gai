# 部署指南（开机自启 / 脱离 SSH 运行）

> 目标：本地电脑关机、SSH 断开、甚至服务器重启后，BGE 和 ChatWiki API 都能自动运行，
> 保证 Dify 工作流在任何人的电脑上都能正常调用接口。

当前部署环境（已写入 service 文件，如有变化需同步修改）：
- 项目路径：`/root/BYX_F/Kiro-fixed`
- Python 环境：`/root/anaconda3/envs/byx_1/`
- ChatWiki API 端口：`7002`
- BGE-M3 端口：`8001`

---

## 一、为什么要做这个

手动 `python api_server.py` 跑的进程是挂在你 SSH 会话里的，**你电脑一关机 / SSH 一断，进程就被杀掉**，Dify 就调不通接口了。
用 systemd 把服务托管给系统后：
- 你电脑关机 / SSH 断开 → 服务照常跑 ✅
- 服务器重启 → 服务自动拉起 ✅
- 进程崩溃 → 自动重启（Restart=always）✅

---

## 二、一键部署（systemd 开机自启）

### 1. 复制服务文件到系统目录

```bash
cd /root/BYX_F/Kiro-fixed
sudo cp deploy/chatwiki-bge.service /etc/systemd/system/
sudo cp deploy/chatwiki-api.service /etc/systemd/system/
```

### 2. 重新加载 systemd 配置

```bash
sudo systemctl daemon-reload
```

### 3. 设为开机自启 + 立即启动

```bash
# 设开机自启
sudo systemctl enable chatwiki-bge chatwiki-api

# 立即启动（先启 BGE，API 会等 30 秒再起，确保 BGE 就绪）
sudo systemctl start chatwiki-bge
sudo systemctl start chatwiki-api
```

> 注意：启动前先把你手动跑的旧进程停掉，避免端口冲突：
> ```bash
> pkill -f api_server.py
> ```
> （BGE 如果已经在后台跑且占用 8001，systemd 的 bge 服务会启动失败，可先 `pkill -f "vllm serve"` 再统一交给 systemd 管理。）

### 4. 验证

```bash
sudo systemctl status chatwiki-bge    # 看是否 active (running)
sudo systemctl status chatwiki-api

# 健康检查（返回 JSON 且各项为 true 即正常）
curl http://10.7.10.102:7002/api/health
```

---

## 三、Milvus 也要设开机自启（docker）

Milvus 是 docker 容器，服务器重启后默认不一定会自启，执行一次：

```bash
docker update --restart unless-stopped milvus-standalone milvus-etcd milvus-minio milvus-attu
```

---

## 四、常用运维命令

```bash
# 查看实时日志
journalctl -u chatwiki-bge -f
journalctl -u chatwiki-api -f

# 改了代码后重启 API
sudo systemctl restart chatwiki-api

# 停止 / 启动
sudo systemctl stop chatwiki-api
sudo systemctl start chatwiki-api

# 取消开机自启
sudo systemctl disable chatwiki-api
```

---

## 五、改了代码怎么更新

```bash
cd /root/BYX_F/Kiro-fixed
git pull origin feature/dynamic-module-truncation
sudo systemctl restart chatwiki-api      # 重启 API 生效
```

---

## 六、手动启动（临时调试用，不推荐生产）

```bash
bash start_all.sh   # 启动 BGE + API
bash stop_all.sh    # 停止
```

> 手动方式起的进程仍然会在你 SSH 断开后被杀，仅适合临时调试。生产环境请用 systemd。

---

## 验证「电脑关机也能用」

设置完 systemd 后，可以这样验证：
1. 断开 SSH（或直接关掉本地电脑）
2. 过几分钟，从别人的电脑打开 Dify，正常对话
3. 或从任意机器执行：`curl http://10.7.10.102:7002/api/health`，能返回 JSON 即说明服务独立运行中
