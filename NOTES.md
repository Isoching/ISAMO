# ISAMO 项目说明与使用指南

## 1. 项目用来做什么？

ISAMO 是 Interpretable Surrogate-Assisted Multi-objective Optimisation，即“可解释的代理模型辅助多目标优化”。本项目研究如何设计和比较以稀烟气为碳源、以电力驱动的绿色甲醇生产系统。过程包含碳捕集、CO2 电还原、水电解、甲醇合成及公用工程；8 个设计变量将这些环节联系起来。

项目回答的工程问题是：**在给定进料、过程模型和变量边界下，如何提高碳利用率，同时降低单位甲醇能耗和设备规模，并理解折中方案为什么被选中？**

三项优化目标为 CU 最大化、等效 SEC 最小化和无量纲资本规模代理指标最小化。甲醇产量、LGWP 代理值及电解槽面积强度属于优化后的派生报告指标。资本规模代理指标用于相对设备规模比较，不能直接解释为货币 CAPEX；LGWP 代理值也不等于完整生命周期评价。

主要适用场景是研究阶段的流程筛选、运行/设备配置比较、代理模型与主动学习研究，以及已有结果的复算和审查。源码和必要数据位于 [source/](source/)；下文所有运行命令均在该目录中执行。

## 2. 意义、主要贡献与前景

耦合过程中的局部调整会改变整体碳、能量和设备需求。多目标设计需要同时比较这些变化，而高保真仿真的调用预算通常需要合理分配。另一个关键问题是：模型在全局样本上表现良好，并不代表它在优化器集中搜索的 Pareto 附近同样可靠。

| 需要处理的问题 | 本项目的做法 | 对工程设计的意义 |
| --- | --- | --- |
| 需要反复评估候选配置 | 使用真实 gPROMS 数据训练目标专属异构代理集成 | 用拟合模型支持候选方案比较 |
| 优化将搜索集中到局部区域 | 将 Pareto 局部选点的真实仿真返回用于主动学习更新 | 把新增仿真用于决策相关区域的模型改善 |
| 不确定性与评估数据容易混用 | 分组交叉保形校准，区分开发、STOP 和独立确认数据，记录冻结哈希 | 明确每项指标能支持什么结论 |
| 多个目标互相竞争 | NSGA-III 搜索 Pareto 折中，TOPSIS/VIKOR 按指定权重排名 | 给出有偏好条件的设计选择 |
| 代理预测缺少解释 | 结合 SHAP 与设计输入分析 | 解释拟合模型中的主要关联与折中驱动因素 |

这里的贡献是这些环节在同一研究流程中的整合、决策区域验证及可追溯实现。原确认集记录显示，相对于未加入 126 个新 AL 案例的对应最强模型，CU、SEC 和资本规模代理指标的 RMSE 分别降低 42.5%、8.5% 和 11.2%；相关依据见 [方法贡献表](source/data/validation_evidence/confirmation96_method_contribution.csv)。这些数字描述本案例，不能直接作为其他流程的改善幅度。

该流程可作为其他电气化 CO2 利用或氢基化工过程的研究起点：替换高保真过程模型、数据、目标定义和边界后，重新建立代理及验证证据。进一步研究还可加入波动电力、变化的电力碳强度、经济目标和可操作性约束。这些是扩展方向，当前发布的固定过程与数据并未实现这些新场景。

## 3. 选择正确的使用路径

| 想完成的任务 | 是否需要模型权重 | 是否需要实时 gPROMS | 从哪里开始 |
| --- | --- | --- | --- |
| 检查数据、复算已记录指标、复现稿件排名 | 否 | 否 | 第 4 节 |
| 对新输入预测、搜索新的折中方案 | 是；可用原冻结模型或重新训练 | 代理计算本身不需要；新方案的物理确认需要真实仿真 | 第 5–7 节 |
| 重新训练并校准原案例的代理模型 | 可从随附数据重新训练 | 已有训练数据可直接使用 | 第 5 节路径 B |
| 为新的化工流程开展研究 | 需要重新开发模型 | 需要相应过程仿真或真实数据来源 | 第 10 节 |

轻量仓库包含 86 个原发布包文件，以及根目录 README、NOTES 和忽略规则。3 个历史冻结模型共约 1.04 GiB，保存在本地结果包 `results/v30_model_freeze/models/`。GitHub 仓库没有托管这些大权重文件；初次使用可以先完成第 4 节。

## 4. 第一次运行：安装、核验和复算

### 4.1 获取源码并安装

使用 Python 3.12。依赖版本以 `source/pyproject.toml` 为准，它们与已验证的原模型运行环境匹配。

```bash
git clone https://github.com/Isoching/ISAMO.git
cd ISAMO/source
python -m venv .venv
```

macOS/Linux 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

激活后安装基础依赖并检查命令入口：

```bash
python -m pip install -e .
python -m isamo_release --help
python -m isamo_release verify
```

`verify` 成功时返回 `status: PASS`，核对源码清单、数据身份、开发/留出数据排除关系，以及记录中的验证指标。它不会重新训练模型。安装后也可使用 `isamo` 命令作为 `python -m isamo_release` 的简写。

### 4.2 复算已有独立验证结果

```bash
python -m isamo_release evaluate --output outputs/recorded_validation
```

打开 `outputs/recorded_validation/validation_metrics_recomputed.csv`。该文件从已保存的代理预测、gPROMS 标签和预测区间重算指标，包含固定区域、全局收敛子集和冻结后确认集。

最终确认数据在结果表中沿用 `confirmation96` 名称；有效样本为 **94**，原发出 96 个案例。对应三目标的固定范围 NRMSE 为 3.99%、1.81%、1.52%，名义 95% 区间的实测覆盖率为 100%、96.81%、100%。固定范围 NRMSE 是用冻结目标跨度归一后的 RMSE，不是每个样本相对误差的上界。其他验证集的覆盖率不同，不应将确认集结果推广到全部输入域。

### 4.3 复现当前稿件的折中排名

```bash
python -m isamo_release rank --output outputs/manuscript_equal_weight
```

打开 `outputs/manuscript_equal_weight/pareto_ranked.csv`，按相应排名列筛选：

| 列 | 如何读取 |
| --- | --- |
| topsis_rank_prequery | 等于 1 的行为 TOPSIS 首选 |
| topsis_closeness_prequery | 越大越接近该排名规则的理想点 |
| vikor_rank_prequery | 等于 1 的行为 VIKOR 首选 |
| vikor_Q_prequery | 越小越优 |

输出保留输入行顺序，CSV 第一行不一定是首选。默认使用 589 点的稿件固定候选池前沿；等权重下 TOPSIS 的首选 ID 为 `C012542`，VIKOR 的首选为 `C012422`。

若要研究“更重视碳利用率”的偏好，用一个新的输出目录：

```bash
python -m isamo_release rank --weights 2 1 1 --output outputs/manuscript_carbon_priority
```

3 个权重依次对应 **CU、SEC、资本规模代理指标**，必须为有限非负数且总和为正；程序会归一化。修改权重会改变折中推荐，不能将某一权重下的首选称为普遍最优设计。

### 4.4 绘制确认集预测对照图

```bash
python -m pip install -e ".[plot]"
python -m isamo_release plot --output outputs/confirmation_plots
```

输出 `confirmation_parity.png`、`.pdf` 和 `.svg`，展示记录中的 gPROMS 值与代理预测值。至此已完成无需大模型权重的基本使用路径。

## 5. 获得用于新预测和优化的模型

先安装学习与绘图可选依赖：

```bash
python -m pip install -e ".[learn,plot]"
```

### 路径 A：使用原冻结权重

如果拥有原结果包，将以下 3 个文件放到 `source/models/`，或自行通过 `--models PATH` 指定其所在目录：

```text
carbon_utilization.joblib
specific_energy_kwh_per_kg_meoh.joblib
capex_proxy.joblib
```

假设已放入 `models/`，执行：

```bash
python -m isamo_release verify --models models
python -m isamo_release predict --models models --input data/confirmation_94.csv --output outputs/frozen_prediction
```

第一条命令额外检查原权重哈希及 140 个样本的预测均值/集成标准差；第二条生成 `outputs/frozen_prediction/predictions.csv`。原权重要求匹配的环境与原类名；不要删除源码中的 `V29_FINAL` 兼容命名空间。

### 路径 B：从随附真实数据重新训练

没有历史权重时可训练当前选定方案：

```bash
python -m isamo_release train --variant pooled_all_al --output outputs/retrained_v30
```

训练使用 2,586 个允许的开发案例，输出：

- `models/`：3 个新模型文件。
- `training_metrics.csv`、`candidate_audit.csv`：训练指标和候选学习器审计。
- `development_oof_predictions.csv`：开发数据的折外预测记录。
- `training_record.json`：本次训练配置、数据身份与新模型哈希。

未加入新增 126 个 AL 案例的比较方案可单独运行：

```bash
python -m isamo_release train --variant pooled_existing60 --output outputs/retrained_comparator
```

它使用 2,460 个开发案例。完整训练的计算开销较大；上述命令生成新实验，不承诺新权重与原冻结权重逐字节相同。`verify --models` 是原冻结模型的精确校验，不能用于把重新训练的权重判定为原冻结权重。

以下第 6–8 节的示例使用 `outputs/retrained_v30/models`；若选择路径 A，将该参数换成 `models`。

## 6. 正确输入新方案、预测与校准

### 6.1 自定义设计输入

在 `source/inputs/my_designs.csv` 中创建 CSV，保留以下 8 个列名，例如：

```csv
P_Cap,J_eCO2R,T_eCO2R,N_eCO2R,P_WE,N_WE,P_MeOH,T_utility
4.5,0.5,343.15,640,3.0,200,55.0,513.15
```

| 输入 | 含义 | 单位 |
| --- | --- | --- |
| P_Cap | 碳捕集压力 | bar |
| J_eCO2R | CO2 电还原电流密度 | A/cm2 |
| T_eCO2R | CO2 电还原温度 | K |
| N_eCO2R | CO2 电解槽数量参数，沿用冻结模型定义 | count |
| P_WE | 水电解功率 | MW |
| N_WE | 水电解数量参数，沿用冻结模型定义 | count |
| P_MeOH | 甲醇合成压力 | bar |
| T_utility | 公用工程温度 | K |

准确上下界见 [protocol.json](source/config/protocol.json) 的 `fixed_input_bounds`；单位与目标定义见 [target_reconstruction_constants.json](source/config/target_reconstruction_constants.json)。两个数量变量使用整数。预测命令不会替用户完成全部边界和工程适用性检查，因此应先核对列名、单位、有限数值和模型输入域。

```bash
python -m isamo_release predict --models outputs/retrained_v30/models --input inputs/my_designs.csv --output outputs/new_prediction
```

输出 `predictions.csv` 包含原输入、三个 `<target>_pred` 均值及 `<target>_std_prequery` 集成标准差。标准差不是已经校准的 95% 区间。

### 6.2 为新模型重新做分组交叉保形校准

```bash
python -m isamo_release calibrate --models outputs/retrained_v30/models --output outputs/retrained_calibration
```

该命令对允许的开发集进行五折来源/聚类分组重拟合，输出 `cross_conformal_oof_scores.csv` 与 `cross_conformal_calibration.json`。完整校准也具有较高计算开销；STOP 和独立验证集不用于拟合校准分数。

校准 JSON 的每个目标包含分位数 `finite_sample_quantile` 和 `scale_floor`。与同一模型的预测均值 `mu`、集成标准差 `sd` 结合时，区间半宽按 `q * max(sd, floor)` 计算，再采用原约定的物理支持裁剪：CU 为 `[0, 0.999]`，SEC 和资本规模代理指标下限为 `1e-6`。原冻结模型对应的校准记录为 `freeze/cross_conformal_freeze.json`；不要把它的分位数直接套用到新训练模型上。

当前 `predict` 命令输出均值与标准差，`calibrate` 输出校准记录；它们不会自动将新的区间合并到预测 CSV。复算历史区间与覆盖率可用第 4.2 节的 `evaluate`。

如果已完成上述新预测和新校准，可按同一模型的记录合并区间。以下为 macOS/Linux 的运行示例；Windows 用户可将其中的 Python 部分保存为脚本后运行：

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd

pred = pd.read_csv("outputs/new_prediction/predictions.csv", float_precision="round_trip")
cal = json.loads(Path("outputs/retrained_calibration/cross_conformal_calibration.json").read_text())
for target, contract in cal["quantiles"].items():
    mu = pred[f"{target}_pred"].to_numpy()
    sd = pred[f"{target}_std_prequery"].to_numpy()
    half = contract["finite_sample_quantile"] * np.maximum(sd, contract["scale_floor"])
    lower = np.clip(mu - half, 0.0, 0.999) if target == "carbon_utilization" else np.clip(mu - half, 1e-6, None)
    upper = np.clip(mu + half, 0.0, 0.999) if target == "carbon_utilization" else np.clip(mu + half, 1e-6, None)
    pred[f"{target}_interval_lower_95"] = lower
    pred[f"{target}_interval_upper_95"] = upper
out = Path("outputs/new_prediction/predictions_with_intervals.csv")
if out.exists():
    raise FileExistsError(out)
pred.to_csv(out, index=False)
PY
```


## 7. 搜索新的 Pareto 折中，并使用正确的档案排名

可先执行一个小规模流程检查：

```bash
python -m isamo_release optimise --models outputs/retrained_v30/models --population 12 --generations 2 --partitions 2 --seeds 42 --output outputs/optimisation_smoke
```

这只检查流程，不用于评价优化质量。正式默认配置为种群 256、300 代、参考方向分区数 20：

```bash
python -m isamo_release optimise --models outputs/retrained_v30/models --seeds 42 --output outputs/new_nsga3
```

输出包括每个种子的 `pareto_seed_<seed>.csv`、`history_seed_<seed>.csv`，合并后的 `pareto_ranked.csv`，以及 `run_record.json`。该流程使用实际 pymoo NSGA-III，并投影整数数量变量。

如果要改变**本次新搜索**的排名偏好，必须指定它产生的文件：

```bash
python -m isamo_release rank --input outputs/new_nsga3/pareto_ranked.csv --weights 2 1 1 --output outputs/new_search_carbon_priority
```

若省略 `--input`，程序会回到随附的稿件 589 点档案。新搜索中的设计可由 `input_row_sha256` 和 8 个输入识别；它们不一定有稿件的 `C...` 候选 ID。

随附的另一个档案是 6,221 点的十种子合并前沿，可明确指定后单独研究：

```bash
python -m isamo_release rank --input data/final_pareto_archive.csv.gz --output outputs/expanded_archive_equal_weight
```

它与稿件 589 点前沿是不同分析范围，推荐点不同。若需要按原种子列表开展新的十种子搜索：

```bash
python -m isamo_release optimise --models outputs/retrained_v30/models --seeds 42 137 271 419 613 809 1009 1217 1429 1693 --output outputs/new_ten_seed_search
```

新推荐仍是代理预测的折中候选。用于形成新的工程结论前，需要真实过程仿真确认，并按预先规定的冻结、排除与验证流程评估。

## 8. 解释设计，并理解主动学习接口

### 8.1 SHAP 解释

```bash
python -m isamo_release explain --models outputs/retrained_v30/models --input inputs/my_designs.csv --eval-rows 1 --background 24 --permutations 64 --output outputs/new_explanations
```

输出 `permutation_shap.csv` 与 `explanation_record.json`。每个贡献值相对于所选背景预测解释该输入在拟合模型中的关联；SHAP 不是干预实验，不能据此单独声称因果过程作用。增加背景样本和排列次数会增加预测调用次数；重新计算的解释也不承诺与历史图逐字节相同。

### 8.2 主动学习与真实仿真

```bash
python -m isamo_release select-active --help
```

选点器保留原冻结模型/前沿/目标定义的哈希校验、已标记与禁止进入数据的排除规则。完整新增 AL 研究还需要准备评分候选池、标签和排除清单、不可变哈希文件，配置真实 gPROMS 执行及返回质检。

参考执行源码在 [reference/original_execution/](source/reference/original_execution/)，原生工程在 [gproms/](source/gproms/)。运行真实仿真需要原授权环境及相应路径配置；轻量命令不自动完成一次新的端到端 gPROMS 主动学习实验。

## 9. 数据角色与结果边界

| 文件（位于 source/data/） | 样本数 | 允许的用途 |
| --- | ---: | --- |
| development_2586.csv | 2,586 | 模型开发及分组交叉保形校准 |
| legacy_holdout_600.csv | 600 | 历史学习器评估/guardrail，不用于拟合 |
| stop_30.csv | 30 | 迭代停止与模型版本选择，不作为独立确认 |
| confirmation_94.csv | 94 | 原冻结后的独立确认记录；原发出 96 个案例 |
| fixed_region_60.csv | 60 | 独立固定区域验证 |
| global_converged_12.csv | 12 | 原发出 40 个全局案例的收敛子集，描述性分析 |

开发集包含 2,400 个原案例、60 个已核准来源的局部案例及四轮 AL 接受的 32、31、32、31 个案例；校准保留 2,531 个来源/聚类分组。`SOURCE_MANIFEST.json` 校验 85 个文件，清单自身是第 86 个文件；`DATA_PROVENANCE.csv` 记录数据来源。不要在冻结数据目录中直接写入自己的实验结果。

稿件原六方案表为 `manuscript_reported_six_solutions.csv`。原 LGWP/面积报告值与先前公式/几何修正分开保存，相关一致性问题仍需按本地审查记录处理。数据校验通过只能证明发布文件和记录结果一致，不能替代对这些物理公式的核验。SEC 与年产量还沿用各自冻结的年度小时数约定，不能直接互换 8,400 与 8,760 小时。

本次整理和上传没有重新执行完整训练、五折重拟合、十种子论文搜索或实时 gPROMS；用户执行这些命令产生的是新实验。对于新的方法开发，已可访问的确认集标签不能替代一个新模型冻结后、未参与调参的新独立确认集。

## 10. 如何扩展到其他过程

复用的是“仿真—代理学习—决策相关采样—验证—多目标选择—解释”的研究结构。对其他进料、其他过程拓扑或新变量范围，需要建立新的仿真接口、输入单位/边界、目标与归一化基准、数据身份和冻结/验证协议；原模型不能直接迁移为可靠预测器。

建议在独立实验副本中保存新协议和数据，保留当前原始证据。当前 CLI 的训练、校准与核验绑定本案例的数据契约；换流程需要相应修改，而非仅替换一个 CSV。先完成可行性及质量检查，再训练、搜索，并用新的决策区域真实案例做独立确认。

## 11. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到模块或依赖 | 确认已激活环境、位于 `source/`，并执行 `python -m pip install -e .`；训练/预测需安装 `.[learn]`，绘图需 `.[plot]` |
| 提示需要模型目录 | 提供包含三个模型文件的 `--models PATH`，或先完成训练 |
| 输出目录已经有内容 | 换用新的输出目录；程序要求新目录或空目录 |
| 校验清单或输入身份不一致 | 使用原文件重新核对来源；不要通过修改清单来掩盖差异 |
| 原模型哈希不同 | 检查是否拿到了原三权重；重训模型不能通过原冻结模型哈希校验 |
| 排名与稿件不同 | 核对输入是 589 点稿件档案还是 6,221 点扩展档案，以及权重和 rank 列 |
| 运行耗时较长 | 全训练、分组重拟合、多个种子搜索和大量 SHAP 预测计算开销较大；可先完成无模型快速路径及小规模优化检查 |

版权和许可见 [LICENSE_NOTICE.md](source/LICENSE_NOTICE.md)。本发布未指定新的开源许可证。更完整的数据与复现说明见 [DATA_AND_REPRODUCIBILITY.md](source/docs/DATA_AND_REPRODUCIBILITY.md)。
