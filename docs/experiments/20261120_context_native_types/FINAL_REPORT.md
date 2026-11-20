# C类型检查循环：完整三组配对结果

结论：**按预注册准则保留为本负载opt-in**。
源码 fdad8bc258064c6c308554a43e57e26c5746955d，默认不改；原Python路径与f2ab684c上下文缓存保留。
这是CPU类型predicate，不是原生vLLM SD，也不是新CUDA模型kernel。系统总体仍Partial，
未证明严格输出等价、未声称超过历史Native SD4或达到全局最优。

修改只有C exact builtin-int循环、CPU构建脚本、17专项测试及原辅助模块中的predicate选择/
执行计数。完整tuple相等、原摘要/range校验、特殊类型回退、弱引用/有界缓存、冻结/进度/
KV lease/取消/抢占/Refill/Forward均不改。H8/HF0.8/cap8/C16/seq4固定，不调整最终请求选择。
CPython3.11/Linux CPU研究扩展，现有GCC+Python.h构建，无安装依赖；不是通用wheel打包验收。
Flag1要求context缓存1并有可导入的SO，flag0不调用该C predicate。BUILD记录源/头/SO哈希、ABI、
完整构建命令；原始模型权重/Tokenizer文件不修改。

CPU216通过+11子测试；Native ON子集41通过+11子测试，存在重叠，不相加当唯一测试数。
17项C专项覆盖空序列、float/bool/int子类、负数/巨大int仅类型判断、特殊序列不消费、
随机边界、GC不持有对象、实际调用计数和错误参数。原hasher继续拒绝无效Token范围。
GPU80请求门槛通过：32次原摘要独立比较、706 Draft/410 Target Graph、215跨Target Carry
字节检查、1199唯一交接/验证；清理all0。日志实际C predicate22528次与SO导入路径匹配，
此计数不是GPU Forward次数或性能证明。当前model config/权重header复读匹配，未查整个数据SHA。

正式数据：同一冻结512 ShareGPT、同seed请求计划/HTTP构造、BF16/FA2/TP1、两个sequence floor1024、
context摘要缓存ON，外层eager+manualGraph、noasync/no-prefix-cache。每轮128×128预热排除，
正式预算每请求16–384、合计101003输出token，不是512×128。所有重型诊断正式关闭。
三组顺序42 OFF/ON、43 ON/OFF、44 OFF/ON；准则在首轮结果前登记，不追加挑幸运样本。

| seed | Native类型OFF tok/s | ON tok/s | TPS变化 | E2E P95变化 | 完整输出ID相同 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 283.139196 | 288.659909 | +1.950% | -4.207% | 195/512 |
| 43 | 284.310227 | 288.916383 | +1.620% | -2.781% | 203/512 |
| 44 | 286.118802 | 289.332468 | +1.123% | -0.716% | 244/512 |

OFF **284.522742±1.501128**，ON
**288.969587±0.339421** output tok/s（均值±样本标准差，各3次）。
均值比 **+1.563%**；配对变化均值
+1.564±0.416个百分点。
逐轮E2E P95平均 21.253757→20.703980秒
（-2.587%），TTFT P95平均 0.345791→0.350935秒。
这是逐轮分位数平均，不是合并请求P95；各seed回退和输出差异完整保留在ANALYSIS。
输出差异尚未分类，不能用“纯CPU等价predicate”替代整个生成系统的严格等价验证。

Target实际Forward合计40470/40398，Draft dispatch
48632/48497，平均验证K 2.687504/
2.713358；候选接受率0.801199/0.803045。
不把缓存/Graph命中或C组件速度直接等同于端到端收益，不把不同试验百分比相加。
Native历史341.073279±5.496631来自已存同计划对照，未重新跑原生，且原生编译路径与CDB外层
eager+手工Graph不同。更多Target扫描与验证密度差距仍是系统层的重要事实。

六轮3072/3072、失败0、606018输出token，均关闭本轮独立服务并核对18084和GPU进程。
不停止他人进程，不清历史结果/备份。原始响应/日志/指标/采样保留服务器，源码目录有软链接。
各pair/launch.sh保存完整实际环境、模型路径、GPU、端口和Serve命令。安全复测使用新tag：

```bash
cd /disk14T/yypan/CDB-AV-runtime/results/context_native_types_20261120
/home/yypan/CDB-AV/.venv/bin/python driver.py --tag "reproduce_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --seed 42 --native-types 1
```

仅审查参数加--dry-run，短功能按REPORT_FUNCTIONAL命令；回退--native-types 0，context缓存仍ON。
若从全新clone构建此可选CPU模块，在源码树运行：

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-context-native-types-20261120
/home/yypan/CDB-AV/.venv/bin/python scripts/build_cdb_prefix_types.py --output-dir vllm/v1/spec_decode
```

已有SO/软链接时不要覆盖；本实验绑定已在results/build保存且哈希核对，复测无需重建。
更早f2ab684c的286.883±1.115是上一单独cache实验，不冒充当前候选结果。最终推荐只依本组
同提交对照及完整安全门槛；当前结论不代表全部底层优化已经结束。
