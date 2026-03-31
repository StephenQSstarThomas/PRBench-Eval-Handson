### 2.1 标准目录结构

```
data/tasks/task_<author>_<year>/
├── task.yaml                      # [核心] 任务总配置文件
├── instruction.md                 # [给AI] 任务指令（不含答案）
├── metadata.md                    # [给评判] 包含完整公式和参数（ground truth）
├── <paper_name>.md                # 论文全文（markdown格式）
├── [<paper_name>.pdf]             # [可选] 论文PDF
├── data/                          # [ground truth] 参考CSV数据
│   ├── table1_xxx.csv
│   ├── fig6_xxx.csv
│   └── ...
└── reproduction/                  # [AI输出区] 代码和分析文档
    ├── ANALYSIS.md                # AI必须写的方法论分析
    ├── core.py                    # AI必须写的代码文件
    └── ...
```

### 2.2 task.yaml — 总配置文件（所有字段详解）

这是**最核心的文件**，pipeline 的所有行为都由它驱动。

```yaml
# ==================== 基本信息 ====================
task_id: "task_alexander_2016"       # 唯一标识，与目录名一致

# ==================== 论文信息 ====================
paper:
  title: "论文标题"                   # 完整标题
  author: "作者列表"                  # 如 "R. Alexander et al."
  doi: "10.1103/PhysRevB.94.024103"  # DOI
  year: 2016                         # 发表年份
  paper_file: "Alexander2016.md"     # 论文markdown文件路径（相对于task目录）

# ==================== 文件路径 ====================
instruction_file: "instruction.md"   # AI agent 收到的指令文件
metadata_file: "metadata.md"         # 评判用的ground truth文件
ground_truth_data_dir: "data"        # ground truth CSV所在目录

# ==================== 期望输出 (Full模式) ====================
expected_outputs:
  analysis:                          # 分析文档
    - "reproduction/ANALYSIS.md"
  code:                              # 代码文件（1-8个不等）
    - "reproduction/core.py"
    - "reproduction/fig6_plot.py"
    - ...
  data:                              # CSV数据文件（1-17个不等）
    - "data/table1_parameters.csv"
    - "data/fig6_Fe_formation.csv"
    - ...

# ==================== Docker 环境 ====================
docker:
  image: "python:3.11-slim"          # 固定使用 Python 3.11
  memory_limit: "1g" | "2g" | "4g"   # 内存限制
  timeout: 300-21600                  # 超时（秒）: 5分钟 ~ 6小时
  pip_install:                        # 依赖包
    - numpy
    - scipy
    - matplotlib

# ==================== 评分标准 (Full模式) ====================
grading:
  dimensions:
    - name: "methodology_understanding"
      weight: 0.20                    # 方法论理解 (20%)
      description: "..."              # 具体评判标准
    - name: "code_correctness"
      weight: 0.15                    # 代码正确性 (15%)
      description: "..."
    - name: "data_accuracy"
      weight: 0.50                    # 数据精度 (50%) — Full模式最重要
      description: "..."
    - name: "completeness"
      weight: 0.15                    # 完整性 (15%)
      description: "..."

# ==================== Code-Only 模式覆盖 ====================
code_only:
  instruction_suffix: |              # 追加到instruction.md末尾的额外说明
    ## IMPORTANT: Code-Only Mode
    ...不需要跑模拟和生成CSV...

  expected_outputs:                   # 覆盖: 只要分析和代码，不要数据
    analysis:
      - "reproduction/ANALYSIS.md"
    code:
      - "reproduction/core.py"
      - ...
    # 注意: 没有 data 字段

  grading:                            # 覆盖: 不同的评分权重
    dimensions:
      - name: "methodology_understanding"
        weight: 0.25                  # 方法论 (25%)
      - name: "code_correctness"
        weight: 0.50                  # 代码正确性 (50%) — Code-Only最重要
      - name: "code_quality"
        weight: 0.10                  # 代码质量 (10%) — Full模式没有
      - name: "completeness"
        weight: 0.15                  # 完整性 (15%)
```

### 2.3 instruction.md vs metadata.md 对比

| 维度 | instruction.md | metadata.md |
|------|---------------|-------------|
| **受众** | AI agent（被评估者） | Model-judge（评分者） |
| **目的** | 任务说明书 | 评分答案参考 |
| **ground truth 数值** | **不包含**（agent 必须自行从论文推导） | **完整包含**（用于评判验证） |
| **公式** | 概述性，指出需要用到哪些公式 | 完整 LaTeX 数学公式 |
| **参数** | 部分给出（或指向论文表格） | 所有参数精确数值 |
| **长度** | 150-210 行 | 100-500 行 |
| **数学深度** | 概念层面 | 公式推导层面 |

**instruction.md 典型结构**:

```markdown
# Paper Reproduction Task
## 1. Article Information          # 论文标题/作者/DOI
## 2. What You Must Do             # 步骤: Read → Analyze → Implement → Generate
## 3. Input Parameters             # 部分输入参数（表格形式）
## 4. Constraints                  # 禁止使用的库/方法
## 5. Output Data File Specs       # 每个CSV的列名、行数、格式要求
```

**metadata.md 典型结构**:

```markdown
# Ground Truth Reference for AI Grader
## 1. Article Information
## 2. Core Equations               # 完整的 LaTeX 方程 (如 E_f(n) = a₀√n·ln(n) + a₁√n + a₂)
## 3. Ground Truth Parameters      # 所有参数精确值
## 4. Expected Output Values       # 期望结果数值（含容差说明）
## 5. Physical Interpretation      # 物理意义，帮助评判判断AI是否真正理解
```

### 2.4 data/ 目录 — Ground Truth CSV 格式规范

`data/` 目录存放的是**参考答案 CSV 文件**，AI agent 生成的 CSV 必须与之匹配。

#### CSV 格式要求

1. **注释行**: 以 `#` 开头的行被跳过（用于标注来源和说明）
2. **表头行**: 第一个非注释数据行为列名
3. **数据行**: 后续行为数值数据
4. **精度**: 通常 6-8 位有效数字
5. **缺失值**: 用 `nan` 或空字段表示

#### CSV 的三种类型

| 类型 | 命名模式 | 内容 | 容差 | 示例 |
|------|----------|------|------|------|
| **Table 数据** | `table{N}_xxx.csv` | 论文表格中的参数值 | ~1% | `table1_parameters.csv`: `element,cluster_type,a0,a1,a2` |
| **Figure 数据** | `fig{N}_xxx.csv` | 图表中的数据点（计算结果） | 1-30% | `fig6_Fe_formation.csv`: `n,E_111_dft,E_100_dft,...` |
| **派生数据** | 自由命名 | 计算得到的派生量 | 视任务而定 | `crossover_points.csv`, `string_tension.csv` |

#### CSV 示例

**确定性分析任务** (容差 ~1%):
```csv
element,cluster_type,a0,a1,a2
Fe,111,1.604850,5.352260,-0.147319
Fe,100,1.776770,7.159510,-5.181010
```

**Monte Carlo 模拟任务** (容差 ~30%):
```csv
# Figure 1: Average plaquette convergence at beta=2.3
iteration,L4_cold,L4_hot,L6_cold,L6_hot,L8_cold,L8_hot
0,1.00000000,0.01741039,1.00000000,0.00098352,...
```

**示意图/参数曲线任务** (容差 ~1%):
```csv
tau_hat,x_outer,x_inner,bag_separation,dxo_dtau
0.00000000,0.00000000,12.56637061,12.56637061,2.00000000
```

### 2.5 reproduction/ 目录 — AI 输出区

AI agent 必须在此目录下生成：

1. **ANALYSIS.md**: 方法论分析文档（必须）
2. **Python 代码文件**: 1-8 个不等，按 `task.yaml` 中 `expected_outputs.code` 指定

代码组织模式：

| 模式 | 任务示例 | 说明 |
|------|----------|------|
| **单文件** | `task_fu_2001` | 一个 `fig3_knee_structure.py` 完成所有工作 |
| **核心+图表脚本** | `task_alexander_2016` | `core.py`(公式) + `tables_compute.py`(数据) + `fig{N}_plot.py`(绘图) |
| **完全模块化** | `task_vautherin_1972` | `constants.py`, `grid.py`, `orbitals.py`, `densities.py`, `meanfields.py`, `solver.py`, `scf.py`, `main.py` |

### 2.6 data/ vs reproduction/ 的核心区别

| 维度 | `data/` 目录 | `reproduction/` 目录 |
|------|-------------|---------------------|
| **内容** | CSV 数据文件 | Python 代码 + ANALYSIS.md |
| **作用** | 数值结果（用于精度对比） | 实现过程（用于代码审查） |
| **评判方式** | 自动/半自动数值比较 (`data_accuracy` 维度, 权重50%) | Model-judge 代码审查 (`code_correctness` 维度) |
| **Full 模式** | 必须生成 | 必须生成 |
| **Code-Only 模式** | **不需要生成** | 必须生成 |
| **ground truth 存在** | 是（`data/tasks/xxx/data/` 中有参考CSV） | 否（没有"标准答案"代码） |
| **文件格式** | `.csv`（逗号分隔，含可选注释） | `.py` + `.md` |

**关键理解**:
- `data/` 里的 CSV 是**可量化评判**的——通过数值对比可以客观评分
- `reproduction/` 里的代码是**需要专家判断**的——由 model-judge 审查代码正确性
- 在 Full 模式下，`data_accuracy`（50%权重）是最重要的维度，直接比较 CSV 数值
- 在 Code-Only 模式下，`code_correctness`（50%权重）替代了 `data_accuracy`

---

