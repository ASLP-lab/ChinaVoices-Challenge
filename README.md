# SinoVoices Challenge 2026

SinoVoices Challenge 2026 是面向中文多方言语音识别与方言识别的挑战赛网页项目。页面用于展示赛事背景、任务设置、数据说明、时间安排、评测规则、组织者信息以及参赛入口等内容。

## 项目简介

本挑战赛拟以 NCMMSC 2026 为平台，围绕中文多方言语音识别与方言识别任务，构建统一、公开、可复现的评测流程。赛事关注低资源中文方言建模、跨方言泛化能力和真实语音场景下的系统鲁棒性。

页面包含两个主要赛道：

- 方言识别：识别语音所属的中文方言类别。
- 多方言语音识别：对多种中文方言语音进行文本转写。

## 目录结构

```text
.
├── index.html                 # 网页入口
├── README.md                  # 项目说明
└── assets
    ├── css                    # 页面样式
    ├── img                    # 页面图片资源
    ├── js                     # 页面交互脚本
    └── vendor                 # Bootstrap、AOS 等前端依赖
```

## 本地预览

这是一个纯静态网页项目，不需要安装依赖或构建。可以直接在浏览器中打开 `index.html`。

也可以在项目根目录启动一个本地静态服务器：

```bash
python3 -m http.server 8000
```

然后访问：

```text
http://localhost:8000
```

## 部署到 GitHub Pages

将代码推送到 GitHub 仓库：

```bash
git add -A
git commit -m "Update website content"
git push
```

进入仓库页面：

```text
Settings -> Pages
```

设置：

```text
Source: Deploy from a branch
Branch: main
Folder: /root
```

保存后，GitHub Pages 会自动部署。页面通常可以通过以下地址访问：

```text
https://wwwxiaoliao.github.io/SinoVoices-Challenge/
```

## 维护说明

- 修改网页内容：编辑 `index.html`。
- 修改页面样式：编辑 `assets/css/style.css` 或 `index.html` 中的内联样式。
- 替换图片：将新图片放入 `assets/img/`，并更新 `index.html` 中对应路径。
- 更新部署：提交并推送到 `main` 分支后，GitHub Pages 会自动重新部署。

## License

请根据赛事组织方要求补充本项目的授权协议。
