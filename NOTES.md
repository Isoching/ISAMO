# ISAMO V30 release notes / 源码发布说明

## 当前入口

当前轻量源码及必要数据位于 [source/](source/)，完整运行说明见 [source/README.md](source/README.md)。请在 `source/` 目录中安装和执行命令。这是独立的轻量源码仓库 `Isoching/ISAMO`，仅包含当前源码发布包及说明文件。

本次发布将已经核验的 GitHub 源码 ZIP 的 86 个文件放入主页可见的 `source/` 文件夹。源码、数据和冻结记录保持原始字节；主页说明和本文件单独维护，不属于原 ZIP 的科学文件清单。

## 算法与版本

- 8 个有界变量覆盖碳捕集、CO2 电还原、水电解、甲醇合成及公用工程。
- 3 个优化目标为碳利用率 CU、等效单位甲醇能耗 SEC 和无量纲资本规模代理指标。
- 保留原始目标专属异构代理学习器；便携命令支持验证、训练、预测、NSGA-III、TOPSIS/VIKOR、分组交叉保形校准和 SHAP 分析。
- `V29_FINAL` 是冻结学习器的兼容命名空间，保留它是为了读取原模型及复用原训练逻辑；当前研究发布版本为 V30。

## 数据集及用途

| 文件（均在 source/data/） | 样本数 | 用途 |
| --- | ---: | --- |
| development_2586.csv | 2,586 | 模型开发和分组交叉保形校准 |
| legacy_holdout_600.csv | 600 | 历史模型评估及 guardrail，不用于拟合 |
| stop_30.csv | 30 | 停止判断与模型版本选择，不用于训练 |
| confirmation_94.csv | 94 | 冻结后独立确认；原发出 96 个案例 |
| fixed_region_60.csv | 60 | 独立固定区域验证 |
| global_converged_12.csv | 12 | 原发出 40 个全局案例中收敛子集的描述性分析 |

开发集由 2,400 个原始案例、60 个已核准来源的局部案例及 126 个新主动学习案例组成。校准保留原始 2,531 个来源/聚类分组。数据身份、开发集与留出集排除关系、模型冻结记录和记录中的指标可由 `python -m isamo_release verify` 检查。

## 排序结果与稿件口径

默认排序使用 `manuscript_fixed_pool_front.csv` 的 589 个点，复现当前稿件的 TOPSIS `C012542` 和 VIKOR `C012422`。`final_pareto_archive.csv.gz` 的 6,221 个点属于十种子合并档案；它的推荐点不同，需明确指定输入后再分析。

稿件原六方案表保存在 `manuscript_reported_six_solutions.csv`。原表中的 LGWP 与电极面积指标维持作者选定的报告值；先前提出的公式与几何修正保存在本地结果包的审查记录中。发布源码不替换这些稿件值，公式与表格的一致性仍需作者按审查记录处理。

## 模型、过程仿真与复现范围

3 个冻结模型共约 1.04 GiB，保留在本地结果包 `results/v30_model_freeze/models/`，未包含在轻量源码上传中。验证记录、指标重算和档案排序可直接使用随附数据执行。预测与优化可通过 `--models PATH` 指定这 3 个模型，或按源码说明从真实开发数据重新训练。

原 gPROMS 工程、执行参考代码和重构常数保存在 `source/`。运行原生过程仿真需要授权的 gPROMS 环境。此次整理与上传验证源码、数据和记录中的结果，没有重跑完整训练、五折模型重拟合、十种子论文优化或实时 gPROMS 仿真。

`SOURCE_MANIFEST.json` 校验源码包内的 85 个文件；清单自身为第 86 个文件。`DATA_PROVENANCE.csv` 记录数据来源。原 ZIP 的 SHA-256 为：

```text
d0cdfbe5ab26265eb182296f0804e8bd56d1ca8996346a36795d7d9ae90ad542
```

## 版权

本次发布沿用 [source/LICENSE_NOTICE.md](source/LICENSE_NOTICE.md)，未指定新的开源许可证。代码、数据及第三方工程的版权声明以原记录为准。
