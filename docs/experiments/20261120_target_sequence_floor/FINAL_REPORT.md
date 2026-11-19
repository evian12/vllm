# 固定Target Graph序列上界：三组正式对照结论

本报告取代性能待定快照。**保留1024作为本负载的吞吐优先实验推荐配置**；不是全局最优，
仍未超过原生SD4，也未完成严格输出等价验收。默认源码开关不改动，不重启生产/他人服务。
代码 `23eaa57d50af9c740088d0f375acc59c3cf2474d`（仅新增测试），运行时源码与008c04f8相同；独立变量
CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0/1024，Draft floor1024固定。其余条件：Llama8B/1B、
GPU3、ShareGPT512/C16、seq4/len1024/budget1024、BF16/FA2/TP1、eager外层+manual Graph、
no-async/no-prefix-cache、H8/HF0.8/cap8；兼容bounds、GEMM、两类Metadata优化OFF。
每轮新服务；128/C16/128token预热排除；每轮101003输出token，六轮3072/3072完成、无失败。

| seed | floor0 tok/s | floor1024 tok/s | TPS变化 | E2E P95变化 | 完整输出ID相同 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 276.598093 | 279.530657 | +1.060% | -0.118% | 238/512 |
| 43 | 278.214178 | 281.034740 | +1.014% | +0.345% | 210/512 |
| 44 | 276.956185 | 280.047512 | +1.116% | -1.105% | 181/512 |

OFF **277.256152±0.848774**，ON
**280.204303±0.764202** output tok/s（均值±样本标准差，各3次）。
均值比 **+1.063%**；配对变化均值
+1.063±0.051个百分点，三组均正。
逐轮E2E P95均值 **21.751129→21.687126秒**，基本持平；
TTFT P95 **0.335225→0.351917秒**，增加
16.692ms，明确保留这项代价。
上述是逐轮分位数的平均值，不是合并请求后的分位数。

Target实际Forward合计40378/40369，Draft实际dispatch
48569/48655；平均验证K 2.698669/
2.686551、接受率0.803736/0.802424。
工作量相近，优化减少序列上界重复键/Host回退，并未减少Target阶段次数。
内部完整聚合采样区间的Target replay比例约67.856%→
94.029%（不含首尾小区间，不能当全程精确命中率）；
更大的命中率改善只转化成约1%端到端收益，GPU计算、同步及其它系统开销仍存在。
不能把命中率变化直接等同于TPS变化或把不同试验均值相加。

GPU功能门槛80请求、706 Draft/410 Target Graph、215跨Target Carry字节检查、
1199唯一交接/验证通过；186CPU和6专项/15版本工具检查通过。实际config与safetensor
header确认维度和BF16，Draft共享embedding按原加载代码处理；不修改任何权重/Tokenizer。
同shape元数据/位置/slot/ownership等断言保留。完整输出差异已存ANALYSIS，尚未分类，
不能宣称严格生成等价；不为通过而放宽检查。原生基线未重复测试。

全部六轮服务均释放到15MiB、18084关闭、无本轮GPU PID。后续只为独立实验使用GPU3，
不停止其他进程。原日志/响应/metrics/observer/GPU采样在服务器同名目录，不覆盖历史数据。
各pair/launch.sh保存全部实际环境变量、模型、端口和Serve参数，属于历史命令记录；
复测应使用新tag的受控driver，避免重用旧observer目录：

```bash
cd /home/yypan/CDB-AV/results/target_seq_floor_20261120
/home/yypan/CDB-AV/.venv/bin/python driver.py --tag "reproduce_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --seed 42 --target-floor 1024
```

此命令启动同配置独立服务、跑一轮512请求并自动回收；只审查参数可加--dry-run。
同代码回退命令仅将--target-floor改为0，保留其它固定条件，不reset/checkout覆盖。
推荐组合为Draft/Target两个sequence floor均1024；从已有三组数据支持这个配置，
仍低于历史原生SD4 **341.073279±5.496631**，不声称全部底层优化已经完成。
下一项为保留完整前缀相等检查的CPU指纹memoization，需独立GPU正确性与OFF/ON配对。
