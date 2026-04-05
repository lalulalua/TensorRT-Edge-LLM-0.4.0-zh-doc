# 文档

本目录包含 TensorRT Edge-LLM 项目的文档源码。

## 构建文档

文档使用 Doxygen、Sphinx 与 Breathe 构建。

### 前置条件

1. **安装 Doxygen**（1.14.0 或更高版本）：
   ```bash
   wget https://www.doxygen.nl/files/doxygen-1.14.0.linux.bin.tar.gz
   tar -xzf doxygen-1.14.0.linux.bin.tar.gz
   cd doxygen-1.14.0
   make install
   ```

2. **安装 Python 依赖**：
   ```bash
   cd docs
   pip install -r requirements.txt
   ```

### 构建命令

在 `docs` 目录下执行：

1. **生成 Doxygen 文档**：
   ```bash
   doxygen
   ```

2. **构建 Sphinx HTML 文档**：
   ```bash
   make html
   ```

   或：
   ```bash
   sphinx-build -M html ./source/ ./build
   ```

### 输出

生成的 HTML 文档位于：

- `docs/build/html/`

在浏览器中打开 `docs/build/html/index.html` 即可查看文档。
