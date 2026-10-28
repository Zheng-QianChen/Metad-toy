# CLI 安装验证 — 310，2026-10-27

## 结果

10 项安装检查全部通过，使用 Python 3.10.6、CMake 3.22.5、GNU C++ 7.5，
在 310（excelsior，glibc 2.17）直接运行，未使用 Slurm。

1. ZQC_CLI_ONLY=ON 可在无有效 C++/CUDA 编译器的配置下打包。
2. 未修改源码的重复构建复用 wheel，ready.stamp 不改变。
3. cmake --install --prefix 可覆盖配置时前缀，路径带空格可用。
4. 在项目外通过 PATH 直接调用 metadtoy --help/--version，无需激活环境。
5. 安装的依赖可执行 FES 重建、PNG/CSV/JSON 导出及进度显示。
6. 已存在的无关 metadtoy 命令保持原样，安装明确失败。
7. 不支持的 DESTDIR 在写入目标环境前报错。
8. 修改源码但保留版本 0.1.0 后，增量打包/安装会更新命令行为。
9. 临时移走源码目录后，已安装命令仍可运行。
10. ZQC_BUILD_CLI=ON 可与 Host 偏势库共同构建；zqc_bias 编译通过。

测试使用离线 wheelhouse：NumPy 1.23.4、Matplotlib 3.6.2，以及它们所需的
平台兼容 wheels。标准 Python 包的元数据从 requirements-analysis.txt 读取依赖，
命令入口由 pyproject.toml 的 project.scripts 提供。

本次未重新编译 CUDA/LAMMPS/GPUMD；CLI 选项默认关闭，CLI-only 直接进入
LANGUAGES NONE 的构建路径。完整 MD 插件的数值结论沿用原验证记录。

## 本机实际安装

随后通过同一 CMake 安装组件，在 WSL Ubuntu-22.04 为用户实际安装：

- 命令：/home/zqc/.local/bin/metadtoy
- 环境：/home/zqc/.local/libexec/metadtoy/venv
- 版本：0.1.0
- CMake CLI 预设配置/打包：3.31.10。
- 安装后 pip check、NumPy/Matplotlib 导入检查通过。
- 从 /tmp 的登录 shell 执行 metadtoy --help / --version 成功。

用户现有 ~/.profile 已将 ~/.local/bin 加入 PATH，未改写 shell 配置。
已打开的终端如未更新 PATH，执行 export PATH="$HOME/.local/bin:$PATH" 即可。

## 复跑

    export PIP_NO_INDEX=1
    export PIP_FIND_LINKS=/absolute/path/to/wheelhouse
    python benchmarks/cli/check_installation.py \
      --source /path/to/Metad-toy --work /path/to/new-validation-directory \
      --cmake /path/to/cmake

--work 应为新验证目录，测试在其中复制源码并创建临时安装前缀。
测试需要已有可用的 Python/pip 和 Host C++17 编译器；只有 CLI-only 用例无需 C++。
包含实际 FES 导出，兼容命令路径含空格的情况。
源码回归工具见 [check_installation.py](check_installation.py)。

- [完整安装日志](cli_installation_310.log)
- [结果 JSON](cli_installation_310.json)

远端证据目录：
/home/saitang/work/MetadToy-gpumd-20261023/build/cli-install-20261027

Windows 工作区证据目录：
work_MetaDtoy/cli_install_20261027
