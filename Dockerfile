FROM ubuntu:24.04

# 配置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 替换为国内镜像源（加速系统包安装）
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources

# 安装基础工具（fish、git），并启用缓存加速
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt-get install -y \
    fish git


# 配置 uv 工具（依赖管理）
ENV UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 安装 Python 3.12（与项目需求匹配）
RUN uv python install 3.12

# 复制项目文件（包括 pyproject.toml、uv.lock 和代码）
COPY pyproject_prod.toml pyproject.toml

# 关键步骤：先安装基础依赖，再替换为 GPU 版本的 PyTorch 和 DGL
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

COPY *.py .
ENV PATH="/app/.venv/bin:$PATH"