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

## 2026-10-13：48 场洪水预分析页面修复

服务器应用更新至 `a7ac011`（页面修复为 `35e2000`）。原每秒轮询直接触发结果区和图表选项回调，大结果在公网传输期间会不断被后续刷新覆盖，表现为任务已完成、文件已确认，但统计区空白或停留在旧结果。新增轻量结果版本通知，复用现有递增 revision；状态继续每秒刷新，结果区仅在版本、模块或筛选变化时更新。未增加内容 hash，未修改水文算法。

使用 `example_data/去除烂泥/去除烂泥` 的全部 48 个 CSV 公网复测：多文件模式、小时尺度，时间列 `Time`、降水 `avg_rain`、蒸发 `E0`、流量 `GZ_in`，未启用上游汇流。此映射用于复现测试，不自动修改用户配置；其他流量列选择仍由用户确认其语义。

实际显示 48 个文件、4,976 条记录、48 场洪水；基础统计、过程图和第 48 场画像成功展示，浏览器无回调错误。服务器 54 项 Dash/HBV/LLM 测试通过（现有 Rasterio 弃用警告保留）。本次未调用 LLM，未率定模型。模型简介更新一并上线。

GitHub 直连拉取曾无响应，采用本地 Git bundle 经现有 SSH 传输、服务用户 `git fetch` 后 `git merge --ff-only` 的方式部署，保留 origin 和原有站点配置。新增 `markdown-it-py` 已记录于服务器依赖清单，依赖文件归属 `hydrotune` 用户。重启会清空会话；用户须刷新页面加载新回调后重新上传文件。
