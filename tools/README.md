# MetaD-toy 离线命令行工具

第一版使用 CLI，功能层位于 metadtoy_cli，后续 TUI 可以直接调用同一套 Python 接口。
读取沉积记录、生成表格和图片都在离线分析进程进行，MD timestep 继续使用 C++/CUDA。

## 安装与入口

Python >=3.9。可以通过 CMake 安装裸命令，也可以直接安装 Python 包。

### 使用 CMake 安装 metadtoy

只安装分析 CLI（配置阶段不需要 C++、CUDA、LAMMPS 或 GPUMD）：

    cmake --preset cli \
      -DPython3_EXECUTABLE=/path/to/python \
      -DCMAKE_INSTALL_PREFIX="$HOME/.local"
    cmake --build --preset cli
    cmake --install build/cli --component cli
    export PATH="$HOME/.local/bin:$PATH"
    metadtoy --help

Python 需提供 pip；已创建项目环境时，可使用
-DPython3_EXECUTABLE="$PWD/.venv-analysis/bin/python"。
预设要求 CMake 3.24+。CMake 3.18+ 可使用等价的手动配置：

    cmake -S . -B build/cli -DZQC_CLI_ONLY=ON \
      -DPython3_EXECUTABLE=/path/to/python \
      -DCMAKE_INSTALL_PREFIX="$HOME/.local"
    cmake --build build/cli --target metadtoy-cli
    cmake --install build/cli --component cli

编译 MD 时同时打包 CLI，在原配置命令上增加 -DZQC_BUILD_CLI=ON。
这个选项默认 OFF。打包目标名为 metadtoy-cli；安装组件名为 cli，
与 GPUMD 动态库的 metadtoy 目标分别构建。

安装布局（以 $HOME/.local 为例）：

- bin/metadtoy：可直接调用的启动器。
- libexec/metadtoy/venv/：CLI 包和 NumPy、Matplotlib 等依赖。
- libexec/metadtoy/install-owner.json：标识本工具管理的环境及基础 Python。

启动器使用安装目录中的 Python，无需 source activate，运行时不依赖源码或构建目录。
基础 Python 安装仍须保留；虚拟环境应在最终路径创建。换安装位置请重新安装。
cmake --install 支持 --prefix /final/path 覆盖前缀；此组件不支持 DESTDIR 暂存。
ZQC_CLI_BINDIR 和 ZQC_CLI_RUNTIME_DIR 可指定前缀内的相对目录。

配置阶段只检查 Python/pip；构建阶段生成 wheel 并准备 pip bootstrap，
安装阶段创建专用环境并安装分析依赖。缺少系统 ensurepip 时也可完成安装。
默认使用 pip 配置的包源。离线构建/安装可预备兼容当前 Python/平台的 wheelhouse，
包含分析依赖及 setuptools、wheel、pip>=22.3，并设置：

    export PIP_NO_INDEX=1
    export PIP_FIND_LINKS=/absolute/path/to/wheelhouse

增量构建在源码/依赖清单改变时重新打包。再次安装会更新同版本号的开发代码，
保留已满足要求的依赖；已有无关 metadtoy 命令或无归属标记的运行目录会报错。
安装检查见 [310 安装验证](../benchmarks/cli/INSTALLATION_310.md)。


使用指定目标的 MD 构建预设时（例如 lammps），还需显式打包 CLI：

    cmake --build build/lammps --target metadtoy-cli
    cmake --install build/lammps --component cli --prefix "$HOME/.local"

Python 包入口使用 [project.scripts](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#creating-executable-scripts)；
虚拟环境的位置约束见 [Python venv](https://docs.python.org/3/library/venv.html)。

### 直接使用 Python 环境

已有虚拟环境或 Conda 环境时，也可使用标准 console_scripts 入口：

    python -m pip install ./tools
    metadtoy --help

开发过程中可以继续从源码运行：

    python3 -m venv .venv-analysis
    . .venv-analysis/bin/activate
    python -m pip install -r tools/requirements-analysis.txt
    python tools/metadtoy.py --help

源码方式的 hills inspect/export 仅需标准库；绘图和 FES 需要分析依赖。
下文的 python tools/metadtoy.py 均可替换为安装后的 metadtoy。
旧 plot_hills.py 为历史脚本，新分析请使用统一入口。

## 沉积轨迹与 CSV

    python tools/metadtoy.py hills inspect /path/to/HILLS
    python tools/metadtoy.py hills export /path/to/HILLS --csv analysis/depositions.csv

支持 MetaD-toy 原生 HILLS v1/v2/v3，以及旧的 step、D 个中心、height、sigma 格式。
原生参数来自文件头，旧格式从列数推断维度；--dimensions 可以指定预期维数进行校验。
export 保留整数步号与 17 位浮点精度，sigma 展开为每轴列；JSON 侧文件保存周期、
WT 参数、核类型及实际导出范围。--through-step N 选择截止步号。

原生格式记录实际沉积高度。PLUMED 原生 HILLS 的列和 WT 高度约定不同，第一版明确
拒绝该格式，后续另做 importer。读取过程中拒绝 NaN、负高度/非正宽度、倒序步号和
不完整尾行。建议分析已结束运行或复制后的文件快照；分析期间源文件变化会报错。

## 偏势与图片

    python tools/metadtoy.py hills preview /path/to/HILLS --out analysis/bias \
        --axes 1 2 --bins 160 200 --range 1 0 1 --range 2 -1 1

输出 bias.csv、bias.png、bias.json。轴号从 1 开始，CSV 的最后一列默认是 bias。
无图形桌面也能生成 PNG。未指定范围时，非周期轴采用沉积中心范围加两侧 3 sigma，
周期轴采用 [0, period)，实际网格记录在 JSON 中。空沉积历史可以 inspect/export，
preview 则要求至少一条选中的记录。

三维或更高维的图片使用显式切片：

    python tools/metadtoy.py hills preview /path/to/HILLS --out analysis/slice \
        --axes 1 2 --at 3 0.5 --bins 160

所有未绘制的坐标都必须用 --at 指定；切片不等于对其他坐标积分得到的边缘自由能。
PBMetaD 支持组合偏势切片，--pb-normalization mean/sum 明确组合的零点约定。
此约定不在原生 HILLS 头中，默认 mean；用户应与模拟配置保持一致。

求值采用分块的解析 hill 核：普通 HILLS 使用 Gaussian/平滑周期核；PB 使用头中记录
的 SmoothGaussian 或 PLUMED_2_10 截断核。--hill-batch 和 --point-batch 控制临时矩阵。
这会重建解析 hill 势，不重放模拟时的网格插值及其边界夹紧；其差别应按网格精度评估。

## FES：默认采用轨迹重加权

FES 重建默认使用按帧重加权。独立入口为：

    python tools/metadtoy.py fes reconstruct /path/to/COLVAR \
        --hills /path/to/HILLS --out analysis/fes \
        --from-step 200000 --range 1 1.05 1.60 --bins 120 \
        --ct-range 1 0.85 3.0 --ct-bins 1024 --energy-unit eV

上述数值来自仓库 O2 示例；其他体系应按实际 CV 范围和热力学单位设置。
原有入口 hills preview HILLS --quantity fes 也默认采用同一方法，自动查找 HILLS
同目录的 COLVAR；可以用 --colvar 指定。缺少所需轨迹或权重会报错。
hills preview 默认的 quantity=bias 仍然输出偏置图。

权重和自由能采用：

    log(w_t) = (V(s_t,t) - c(t)) / kBT
    P(bin)   = sum(w_t in bin) / (sum(w_t in displayed region) * bin_volume)
    F(bin)   = -kBT * log(P(bin)) + C

C 令有样本 bin 的最小 FES 为零。指数权重和直方图在对数域累计，避免溢出。
没有样本的 bin 概率为零、FES 为未定/正无穷；CSV 的 FES 留空，PNG 对应位置遮蔽。
输出不会通过伪计数补出未采样区域。

--weight-source auto 按以下顺序选择一种权重契约：

1. log_weight/logweight 列：已经准备好的无量纲对数权重，原样使用。
2. rbias 或 *.rbias 列：已减去 c(t) 的偏置能，除以 kBT。
3. bias 与 rct/ct/offset 列：计算 (bias-c(t))/kBT；offset 的含义是 c(t)。
4. 只有瞬时 bias 时，使用同一运行的原生 WT HILLS 离线积分 c(t)。

也可用 --weight-source log-weight/rbias/bias-offset/wt-hills 显式选择，并通过
--log-weight-column、--rbias-column、--bias-column、--offset-column 指定列名。
有多个同类候选列时须明确指定；实际采用的列写入 JSON。提供的权重须包含要消除的
全部偏置贡献，并与该帧相对应。若同时有多个可用来源，auto 采用上面的优先级。

静态偏置轨迹可以显式指定 --weight-source static-bias，用 exp(V/kBT) 去偏置。
它要求所选轨迹使用同一个固定偏置；不能把动态 MetaD 的原始 bias 列直接当作静态
偏置处理。工具不会在缺少 c(t) 时自动降级到这种做法。

### 输入与时间选择

支持普通空白分隔 COLVAR、PLUMED 风格 #! FIELDS 表头，以及带列名的数值 CSV。
GPUMD 当前输出的 # step value bias_eV dV_dvalue 可以直接读取。LAMMPS 自定义的
标量输出可用 --fields 给无表头文件命名；同一个表的各列必须在同一帧采集。
step 是非负 int64，不能用浮点时间冒充。重复或倒序 step/time 会报错。

--cv NAME [NAME ...] 指定 CV 列及轴号顺序；默认读取 value 或 s1,s2,...。
--axes 1 2 选择绘图坐标，其他 CV 自动被边际化。此结果是联合/边际概率的自由能；
没有对其他坐标进行固定切片，重加权方法不接受 --at。

--from-step/--through-step 选择闭区间，--discard-rows 丢弃文件开头若干数据行。
选择窗口内的 time（优先）或 step 间隔须一致；不等间隔记录需先整理采样测度。
没有 time/step 的输入按等权时间观察解释，由使用者保证这一前提。
这些选择不会自动判断是否平衡；应按体系选择热化/稳定区间。

--kbt 是热能 k_B*T，单位必须与 bias/rbias/c(t) 相同；默认读取原生 HILLS。
它不是以 K 为单位的温度，不能直接传 300 表示 300 K。
--energy-unit 仅改变标签。当前单次重建要求固定温度和一致的能量单位。

### 离线 c(t)

普通 WTMetaD 使用以下近似，其中 gamma 是 HILLS 中的 bias factor：

    c(t) = kBT * log[
        integral exp(gamma*V(s,t)/((gamma-1)*kBT)) ds
        / integral exp(V(s,t)/((gamma-1)*kBT)) ds
    ]

该估计依赖准平稳 WT 假设 F≈-gamma/(gamma-1)*V，不保证早期非平衡轨迹可靠。
积分使用全部偏置维度上的等体积中点网格，逐步追加 hills；Native LAMMPS/GPUMD
在沉积后求值并记录偏置，因此计入 hill.step <= COLVAR.step 的 hills。
--from-step 丢弃早期样本时仍保留这些样本之前的沉积历史。

--ct-range AXIS LOWER UPPER 必须覆盖每个偏置维度，可重复指定。
它独立于 --range（展示范围），应覆盖积分有显著贡献的区域，不能只截取图片附近；
应根据实际可访问区间和 hill 尾部选择，并检查扩大范围、加密网格后的稳定性。
非周期样本或已应用 hill 的中心落在积分范围外时会报错，但仅通过这个检查不能证明
积分域足够大。周期积分范围必须刚好覆盖一个周期。

--ct-bins 控制积分精度，默认 1D=1024、2D=128×128、3D=48×48×48，
总网格不超过一百万格。--bias-cv 明确按 HILLS 维度顺序列出全部偏置 CV；
未指定时采用 --cv 的顺序，维度必须一致。绘图 CV 与偏置 CV 不同时应分别指定。
积分重建采用解析 hill 核，未逐点重放宿主网格插值，仍有相应离散化差异。

当前自动 c(t) 只支持 1–3 维原生完整 HILLS 的 WTMetaD。
PBMetaD、legacy HILLS 和更高维可以提供已经计算好的 log_weight/rbias/bias+c(t)，
再重建二维联合 FES；不把普通 WT 公式套到 PB 的组合偏置上。
不能用这条路径读取 PLUMED HILLS 的不同高度约定；PLUMED 风格 COLVAR 的显式
校正权重则可以单独使用。

周期 CV 用 --period AXIS PERIOD 显式指定，直方图范围需覆盖一个完整周期。
使用离线 c(t) 且偏置 CV 的映射已明确时，自动继承对应的原生 HILLS 周期；
其他权重路径不把 HILLS 周期猜测映射到任意绘图 CV。

### 输出与诊断

每次生成 fes.csv、fes.png、fes.json 和 fes.weights.csv：

- CSV：bin 中心、FES、概率密度、对数概率密度、原始样本数和 bin 权重 ESS。
- PNG：1D 曲线或 2D 色彩图；空 bin 留空。
- JSON：输入路径与 SHA256、kBT、采用的方法/列、积分网格、采样区间、范围外
  样本数、保留权重比例以及总权重 ESS。
- weights.csv：源数据行号、step、log_weight、采用的 c(t)、是否落入绘图区和 CV。
  可用于检查或后续训练数据对齐；当前不自动与原子轨迹做 join。

ESS=(sum w)^2/sum(w^2) 衡量权重集中程度，尚未修正时间自相关；它不是独立样本数。
裁剪显示范围时，密度在所保留区域归一化。距离 F(r) 仍包含坐标测度的影响，
不会自动施加 O2 的径向熵修正，也不会把 FES 标成势能面。

### 保留 HILLS 渐近估计

旧方法需明确指定 --method hills：

    python tools/metadtoy.py hills preview /path/to/HILLS --out analysis/fes-hills \
        --quantity fes --method hills --energy-unit eV --range 1 1.05 1.60 --bins 250

它继续使用 F=-gamma/(gamma-1)*V+C；旧格式需要 --bias-factor。
这与轨迹重加权是不同估计量，有限采样时不要求两者逐点相同。

已有输出默认拒绝覆盖；--force 才替换。输入变化或解析失败时不发布新的输出。
每个输出文件原子替换，多文件不构成文件系统层面的一次事务。

公式依据：
[PLUMED REWEIGHT_METAD](https://www.plumed.org/doc-v2.9/user-doc/html/_r_e_w_e_i_g_h_t__m_e_t_a_d.html)、
[METAD c(t)](https://www.plumed.org/doc-v2.9/user-doc/html/_m_e_t_a_d.html)、
[静态 REWEIGHT_BIAS](https://www.plumed.org/doc-v2.10/user-doc/html/_r_e_w_e_i_g_h_t__b_i_a_s.html)。


### 处理速度与终端进度

FES 重建按批处理 COLVAR，默认每批 8192 帧。步号单独保存在 int64 列中。
直方图与权重求和采用 NumPy 批处理；离线 c(t) 的规则网格采用逐轴高斯因子，
并在相邻两次沉积之间复用 c(t)。积分范围、网格分辨率和沉积时序由原有参数决定。
可用 --sample-batch N 调节批大小；内存随批大小及积分网格大小增长。

交互终端默认显示进度；重定向或脚本调用默认保持安静。两种 FES 入口均支持：

    python tools/metadtoy.py fes reconstruct COLVAR --hills HILLS --out analysis/fes \
        --ct-range 1 0.85 3.0 --progress always

- --progress auto：默认，在交互终端显示。
- --progress always：强制显示；重定向时按行输出，适合日志。
- --progress never：关闭显示。

阶段包括输入校验、轨迹扫描、重加权、表格导出、绘图、复核和保存。
首次扫描显示已读取帧数；重加权显示已处理/总帧数、百分比和当前阶段的预计剩余时间。
刷新经过限流，进度写入 stderr，stdout 保留 JSON 结果。
Ctrl+C 退出码为 130；计算阶段取消会清理暂存输出，保留原有结果。

当前提供的是 CLI 的终端进度条。未来 TUI 可直接传入
create_reweighted_fes(..., progress=callback)，接收 phase、completed、total、
unit、elapsed_seconds、eta_seconds、status 等字段；total=None 表示总量尚未知。
done 事件在结果发布后发送。JSON 的 execution 保存批大小和发布前各阶段耗时。
完整命令的对比计时见 [310 性能记录](../benchmarks/cli/FES_PERFORMANCE_310.md)。

## 原子轨迹与 Distance 数据集

    python tools/metadtoy.py trajectory inspect trajectory.dump
    python tools/metadtoy.py trajectory inspect dump.xyz
    python tools/metadtoy.py dataset build trajectory.dump \
        --features tools/examples/distance_ids.json --out analysis/distances
    python tools/metadtoy.py dataset build dump.xyz \
        --features tools/examples/distance_rows.json --assume-fixed-order \
        --out analysis/distances --validation-fraction 0.2 --gap-frames 100

这两个命令需要 NumPy。源码中的描述符接口与 argparse 分离，之后 TUI 可以直接调用
TrajectoryReader、DistanceFeatures、build_dataset。导出过程在离线 Python 进程执行。

支持的输入：

- LAMMPS 文本 atom/custom dump，含正交盒、restricted/general triclinic 盒；
  支持 x/y/z、xu/yu/zu、xs/ys/zs、xsu/ysu/zsu，若同时存在则按这个顺序选择。
  scaled 坐标按每帧盒子和 origin 转换成笛卡尔坐标。支持可选 ITEM: TIME/UNITS。
- GPUMD 常见的扩展 XYZ：Properties 指定列，Lattice 保存连续三个晶格向量，
  pbc 指定三个周期标志，Time/Step 可选；id:I:1 是显式原子 ID。
  支持传统引号包围的平铺 key=value 语法。现代嵌套数组语法暂不支持。
- 普通四列 XYZ，以及以上文本格式的 .gz 文件。每个输入是一个完整连续片段，
  步号或时间必须严格递增；重启后的倒序片段应拆成独立文件。缺失值保留为空。
  不支持 LAMMPS 二进制、分 rank 的 % 文件合并、通配符展开和 .zst。

若文件没有 pbc，minimum 距离要求显式提供 --pbc T T T 或 --pbc F F F 等标志。
这里对缺失 PBC 采用显式确认规则，没有采用部分 extxyz 阅读器的默认周期性。
传入标志不能与文件已有标志冲突；周期帧必须有三个独立的盒子向量。

描述符文件采用有序 features 数组，例如：

    {
      "schema_version": 1,
      "identity": "id",
      "length_unit": "angstrom",
      "features": [
        {"name": "r_OO", "kind": "distance", "atoms": [1, 2], "image": "minimum"}
      ]
    }

identity=id 使用文件中的真实整数 ID，不依赖原子行顺序。identity=row 使用从 0
开始的行号，必须显式传 --assume-fixed-order。后者会检查原子数、类型和已有 ID
序列是否改变；没有 ID 的同元素原子互换无法自动识别。两个示例文件分别演示这两种
选择。atoms 是固定的两个原子，不是每帧重新寻找的最近邻对。当前描述符为 distance，
用户可以在 features 中定义多个距离，输出列严格保持定义顺序。

image=minimum（默认）使用本帧的 PBC 和盒子求欧氏最短周期像；斜盒通过有界枚举
避免简单分数坐标取整的错误。超过一百万候选像会明确报错，需先简化过度倾斜的盒基。
image=direct 直接用文件坐标的差；只有坐标本身已展开时，它才表示展开后的距离。
LAMMPS 宿主的 minimum_image 由其 domain 实现：这里的斜盒欧氏最近像是离线工具
明确采用的定义，尚未逐例验证与宿主的斜盒选像惯例一致。当前 GPUMD 接口的正交盒
距离和 O2 例子没有这个差别。将斜盒数据接回 MD 前需补对应的数值交叉验证。

length_unit 必填，表示输入坐标的单位，工具不自动换算。若 LAMMPS dump 写了
ITEM: UNITS，会检查是否匹配：metal/real→angstrom、lj→reduced、si→m、
cgs→cm、electron→bohr、micro→micrometer、nano→nanometer。其他情况由使用者保证
各输入单位一致。原始 time 保留源文件单位；不由帧号推算时间或 MD 步号。

输出为三份配套文件：

- distances.csv：source、frame、step、time、split 以及按定义排序的特征列。
  source 是输入列表从 0 开始的编号；frame 是源文件中从 0 开始的帧号。
- distances.frames.jsonl：每个选中帧的盒子行向量、origin、PBC、原子数、所选
  原子的 ID/行号/类型和读取格式。source+frame 与 CSV 精确对应。完整原子表
  留在源文件中，避免在侧文件重复存储大体系全部坐标和 ID。
- distances.json：特征定义、单位、列顺序、输入绝对路径与 SHA256、帧范围、
  参数和划分统计。分析前后检查源文件；失败时不发布新的输出。

--stride N 每 N 帧保留一帧，从第 0 帧开始；读取过程中仍检查全部帧。
默认 split=unassigned。指定 --validation-fraction 后，对每个源文件分别取末尾
ceil(帧数×比例) 帧作 validation；前面留 --gap-frames 个已选帧标为 gap，其余
标为 train。训练时必须排除 gap；数据太短则报错。gap 单位是已选帧数，
不是 MD 步数或物理时间，应根据采样间隔和相关时间选择；此划分不证明统计独立。
来自同一物理轨迹的相邻片段也不能被当成独立副本来消除泄漏。
已有输出保护与 HILLS 命令一致，--force 才允许替换。每个文件单独原子替换；
三个文件不是文件系统层面的一次事务。

格式依据：
[LAMMPS dump](https://docs.lammps.org/dump.html)、
[LAMMPS triclinic](https://docs.lammps.org/Howto_triclinic.html)、
[extxyz specification](https://github.com/libAtoms/extxyz)。

## 后续 TUI 和 ML 训练边界

当前已实现 HILLS 分析、轨迹重加权 FES、轨迹读取、Distance 数据集导出。接下来按需要补：

1. QL/CSS 描述符，通过统一接口生成有顺序、单位和选择定义的特征。
2. COLVAR 权重与原子轨迹的显式时间或步号对齐。重加权已实现，自动 join 尚未提供。
3. train：共同选择目标（已有 CV 的监督代理、慢变量/tICA 类目标、或其他目标），
   保存配置、种子、损失和验证结果。带偏轨迹不能自动视为无偏训练样本。
4. model：导出 forward 和 VJP/梯度契约，做坐标力及 virial 验证后接回 MD。
5. TUI：选择文件、显示进度、管理任务和打开图片；耗时训练独立执行。

prototypes/extension_lab/ml_models 保留模型文件/VJP 的旧原型。新数据集模块尚未
调用它训练模型，也未修改 MD 的在线 Graph 或引擎接口。

