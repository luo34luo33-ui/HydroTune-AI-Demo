# HydroTune Dash 服务器部署

- 地址：`http://47.98.207.25/`
- 目录：`/home/hydrotune`
- 分支：`codex/hydrotune-dash-uiux`
- 运行环境：Python 3.11，独立 `.venv`，专用用户 `hydrotune`
- 服务：`hydrotune-dash.service`，单 Gunicorn worker、4 个请求线程，监听 `127.0.0.1:8050`
- Nginx：`/etc/nginx/conf.d/hydrotune-dash.conf`，仅匹配上述 IP 的 80 端口；保留其他站点和服务。
- 持久配置：`/etc/hydrotune/dash.env`，保存会话签名密钥，不纳入 Git。

## 运维命令

以下命令在服务器上以 root 执行：

```bash
systemctl status hydrotune-dash --no-pager
journalctl -u hydrotune-dash -n 100 --no-pager
systemctl restart hydrotune-dash
```

服务已配置开机启动和异常退出自动恢复。重启应用会清空内存中的会话、文件及结果，须重新上传。

更新代码：

```bash
cd /home/hydrotune
runuser -u hydrotune -- git pull --ff-only origin codex/hydrotune-dash-uiux
systemctl restart hydrotune-dash
```

## 服务器依赖

服务器的 `/tmp` 为约 945MB 内存盘，安装使用 `TMPDIR=/var/tmp`。CPU 实例使用官方 `xgboost-cpu` 包，保留相同 Python 导入及 CPU 训练接口，不安装 GPU 组件。服务器实际依赖清单位于 `/etc/hydrotune/requirements-server.txt`，由项目两份 requirements 合并后替换 XGBoost 包名，并增加 Gunicorn 和 pytest；项目原依赖文件保持不变。

```bash
cd /home/hydrotune
TMPDIR=/var/tmp .venv/bin/pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  -r /etc/hydrotune/requirements-server.txt
```

已安装 Chromium 和中文字体，供 PNG/SVG 图表导出使用。LLM API 密钥未随代码上传，可在工作台的 LLM 配置中设置。当前入口为 HTTP，尚未配置域名及 HTTPS。

## 实际验证

- 服务器上 Dash / HBV / GIS 相关测试 35 项通过，原相关回归 33 项通过，共 68 项。
- 应用首页、Nginx 入口与 Dash 布局接口均返回 HTTP 200。
- 通过公网浏览器打开首页、进入工作台，页面回调正常，未记录浏览器错误。
- 以生产运行用户实际导出 PNG（20162 字节）、SVG（7204 字节）成功。
- 服务为 `active/running`，开机启动为 `enabled`；检查日志未发现应用错误。
- 未配置 LLM 服务密钥，未发起真实付费 LLM 请求。
