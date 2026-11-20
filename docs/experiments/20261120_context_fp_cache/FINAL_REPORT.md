# 完整前缀摘要缓存：三组OFF/ON正式性能结果

本负载推荐opt-in缓存；默认仍OFF，未合入main，系统总体仍是Partial Implementation，
尚未超过历史原生SD4，也未完成严格输出等价验收。源码 f2ab684c06ba4f41fb9ac4f6afa2393695ed8444，
远端同名分支已核对。修改只有独立CPU缓存辅助模块、RefillQueue.context入口、19+5专项测试；
不改KV复用、lease、冻结检查、Forward、H、HF或请求选择。

完整builtin-int Token tuple相等且同Request/原始hasher才缓存摘要；不是按ID/长度缓存。
同长度变更重新走原始校验，弱引用、64条/1024token上界、跨线程/特殊类型安全回退；
原始BLAKE2b/Token编码不变，所有KV/进度/失效断言保留。GPU门槛32次独立原算法审计一致，
706 Draft Graph/410 Target Graph、215跨Target Carry字节检查、1199唯一handoff/verify通过。
CPU199项与11子测试、workflow15均通过；先前入口/fixture失败及修正记录保留。

条件：Llama8B/1B、GPU3、BF16/FA2/TP1、C16/seq4/len1024/budget1024、
H8/HF0.8/cap8、Draft/Target sequence floor1024、no-async/no-prefix-cache，外层eager+manualGraph。
正式每轮512请求、同一冻结ShareGPT及seed计划；输出预算16–384/token请求，合计101003，
不是每请求128。128请求×128token预热排除。重型metadata/KV/Graph/FP诊断正式阶段关闭，
保留低开销累计计数。交替顺序42 OFF/ON、43 ON/OFF、44 OFF/ON，三组，不追加挑选样本。
每pair已校验相同提交、HTTP计划、client/observer/data/driver哈希、完整命令和其它环境一致。

| seed | OFF tok/s | ON tok/s | TPS变化 | E2E P95变化 | 完整输出ID相同 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 278.536615 | 286.019941 | +2.687% | -1.583% | 231/512 |
| 43 | 281.696364 | 288.141836 | +2.288% | -2.749% | 202/512 |
| 44 | 281.803595 | 286.487900 | +1.662% | +1.184% | 192/512 |

OFF **280.678858±1.856011**，ON
**286.883226±1.114819** output tok/s（均值±样本标准差，各3次）。
均值比 **+2.210%**；配对变化
+2.212±0.516个百分点，三组均正。
逐轮E2E P95均值 21.313165→21.085320秒；
TTFT P95均值 0.350145→0.345617秒。
这是逐轮P95平均，不是合并请求P95。seed44的E2E P95+1.184%、TTFT P95约+4.54%，
不能宣称所有延迟均改善或稳定保证单请求延迟下降。

Target Forward合计40374/40393，Draft dispatch
48579/48653；验证平均K 2.700353/
2.692565，接受率 0.802807/0.801769。
工作量相近，改的是重复CPU摘要计算，不能用缓存命中率本身宣称提速；
内部计数区间与边界排除口径详见FINAL_DECISION.json。完整输出差异保存在ANALYSIS，
尚未分类；缓存数学身份与请求级最终输出等价属于不同证据。

六轮3072/3072、失败0、606018输出token；每轮服务退出到15MiB、18084关闭、无本轮GPU PID。
后续独立布局原语20组也已退出，当前GPU3同样15MiB。其他进程不终止，历史日志/源码/备份不覆盖。
各pair/launch.sh保存完整实际环境、模型路径、GPU、端口、Serve参数。安全复测采用新tag：

```bash
cd /home/yypan/CDB-AV/results/context_fp_cache_20261120
/home/yypan/CDB-AV/.venv/bin/python driver.py --tag "reproduce_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --seed 42 --context-cache 1
```

此命令一轮512并自动回收；只审查参数加--dry-run。回退只改--context-cache 0，同源码同配置，
不reset/checkout。GPU空闲前不得另起服务。结果软链接在运行源码树及主代码目录保留。

推荐组合在本轮为两个sequence floor1024+完整上下文摘要缓存ON；各增量单独配对，
不要把跨实验均值/百分比简单相加。ON的286.88仍低于历史原生SD4
341.073279±5.496631；没有宣称全局最优或超过基线。

只读工作量审查显示：前一完整Target-floor配置每轮约13456 Target/16218 Draft，
原生SD4约7559 Target/30237 Draft；验证请求密度1.695/3.920、候选接受率80.24%/60.46%。
少算低收益Draft并不自动抵消更多昂贵Target扫描。Native历史启用了Inductor+原生Graph，
CDB是eager外层+手工Graph，这是明确的实现配置差别，不能掩盖或当作隔离变量结论。
本轮不改最终请求选择策略。剩余底层候选需依据真实开销继续审查，不能承诺必然超过原生。
