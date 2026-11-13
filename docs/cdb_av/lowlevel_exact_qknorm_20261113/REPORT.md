# CDB-AV 底层优化收敛复查 — 2026-11-13

## 判断与推荐

**本阶段按已有候选清单收敛**：新增保持原BF16舍入/归约语义的QK RMSNorm+NeoX RoPE融合，
通过独立张量、真实模型及同代码三组交错开关对照；原始API地址对齐保护补齐，
最新加固版另补三次fresh-service吞吐回归。
这是当前固定配置的可交付研究版本，**不是全局最优，也不是完整系统严格输出等价验收**。
整体仍为Partial Implementation。默认开关仍0；推荐实验配置显式1/both。
未整合、修改或分析网页版算法分支；不研究动态H或阈值调参。

- 运行源码：`/home/yypan/CDB-AV/worktrees/cdb-exact-qknorm-rope-20261113`。
- 同代码开关性能SHA：`b6da217f6ba92b728f1aad7ea92ce1943fa4dc97`。
- 加固版实现/功能/三轮回归SHA：`c7531aaef4438b138c600e22aae384519a4522f6`。
- 分支`codex/cdb-exact-qknorm-rope-20261113`正常推送，文档/集成SHA另见GIT_DELIVERY.json。
- 原目录`/home/yypan/CDB-AV/vllm`仍为`e6b5defc8a4b15bace432c9f7b8b8b52c0a17a8f`，9项未跟踪备份/zip保留。
- GPU3/18084仅测试期间使用；完成后端口关闭、GPU3=15MiB，无本轮计算进程。其他服务未结束。

## 性能

Qwen3-8B/Qwen3-1.7B，RTX4090物理GPU3（内部cuda:0），vLLM0.15.1，
Torch2.9.1+cu129，BF16/FLASH_ATTN（4090本分支FA2 AOT路径）、TP1；
max_num_seqs4/max_model_len1024/seed42，eager外层/manual Graph，无async/prefix cache。
固定H8/HF0.8/Kcap8；Draft query bound8/mixed keys24、Target keys24、metadata packing OFF。

每次fresh服务先128请求/C16/128输出预热，再512请求/C16/128输出正式实验；
各轮实际65536输出token。闭环固定并发，不是匀速到达，不是GPU batch16。
同一b6顺序OFF0/ON0/ON1/OFF1/OFF2/ON2，逐项核验配置/源码仅CDB_AV_EXACT_QKNORM不同。
Trace/数值审计/Profiler关闭，保留相同聚合日志及NVML1Hz。
加固版ON-only另三轮是回归，**不冒充新一组开关因果对照**。

| 代码/开关 | 三轮output tok/s | 均值±样本SD | 各轮P95秒的均值±样本SD |
|---|---|---|---|
| b6同代码 OFF | 234.983, 235.806, 232.748 | 234.512 ± 1.582 | 10.663 ± 0.026 |
| b6同代码 ON | 238.929, 239.067, 240.038 | 239.345 ± 0.604 | 10.400 ± 0.048 |
| c753加固版 ON | 237.745, 239.159, 240.370 | 239.091 ± 1.314 | 10.449 ± 0.128 |

b6配对吞吐+1.679%、+1.383%、+3.132%，平均
**+2.065% ± 0.936%**；
P95配对平均-2.467%。
仅三组/固定负载，不宣称普遍或全局收益。
9轮正式4608/4608成功、589824输出token；另1152预热请求，均实际128输出。
两次32请求功能验收单独计数，重型审计的TPS不作为性能。

此前222/233使用不同代码/预热，不能直接相减归因。本次融合效应由b6
234.512→239.345同配置对照支持。加固前后CUDA kernel源码相同，
1234条函数/反汇编相关行相同；host对齐检查另由三轮回归监测。
非streaming HTTP elapsed包含服务等待；未测TTFT/ITL/TPOT。
P50/均值/wall/GPU采样见两个PERFORMANCE.json。
NVML utilization不是SM占用率或精确kernel-active；线程顺序/内部UUID/批次形状不能由seed全部固定。

## 改动

1. `experiments/cdb_exact_qknorm/kernel.cu`和`binding.cpp`：
   Q/K逐头RMSNorm+full head128 NeoX RoPE一个kernel、分别连续输出Q/K。
   保持native vector8累积、CUB1024/valid16归约、归一化/乘权重/RoPE逐次BF16舍入。
   QKV/V、权重、位置、cos/sin只读，共享缓冲16B对齐。
   API检查设备/dtype/布局/形状/别名/epsilon/head数；拒绝非空QKV/qw/kw未16B对齐offset视图，
   GPU位置边界断言保留。
2. `vllm/model_executor/layers/cdb_exact_qknorm.py`、`models/qwen3.py`：
   实验默认0；both/draft/target可选，本轮both。
   仅BF16/SM89/TP-PP-DP-DCP1/maxseq4/eager/静态权重/无LoRA/量化/prefix/async等受控域。
   norm/rotary后端不符拒绝；超行数/不支持布局完整回退native。
   严格二进制/Torch版本/5个源码头文件SHA receipt防止旧库。
   没有修改投影、Attention、V、KV管理、采样、模型权重。
3. `build.py`和独立验证器/CPU测试：
   现有CUDA12.4/GCC11手工构建SM89私有扩展，对Torch2.9.1+cu129；
   没安装Ninja/改依赖/改现有vLLM.so，不宣称跨版本兼容。
4. `scripts/cdb_av/git_workflow.py`及回归：
   原权重目录检查误拦HEAD已跟踪canonical models/*.py源码。
   只消除此类既有源码目录名误报；新建/仅暂存/外部模型文件仍拒绝，
   权重/大小/内容/秘密扫描保留。fixture使用有效Python，正常独立提交，未绕过checkpoint。

完整CODE.patch含实现/测试/必要工具修补，源文件快照、receipt、失败记录保留。
不改K/H、请求选择、reuse_span、fill/commit/invalidate/free/lease或Scheduler；
不删槽位碰撞、上下文指纹、Progress断言。

## 验收

- CDB CPU **400 passed/5 GPU-only skipped**；版本工具15项；
  扩展CPU输入拒绝2项；原始GPU API负测试**14 passed**。
- 独立106案例：104非空Q/K原始字节native一致，2空案例不作数值证据；
  全64层真实学习norm权重及零/微小/次正规/大/最大BF16/常数/交错符号/零权重；
  输入/权重/cos-sin cache/positions保持。这里cache是RoPE表，不是KV Cache；后者由实际Graph审计验证。新旧候选二进制分别通过。
- 两模型追加1024行、位置0..1023边界：Q/K字节相同，QKV保持；1025行raw API拒绝。
- 每次真实模型32/32、4096输出token。加固版**2048次Q/K字节检查**
  （Draft28层896、Target36层1152），其中非零位置Draft672/Target1116；
  两次合计4096检查，不宣称覆盖全部Forward/输入。
- 加固版Draft Graph 669次、Target Graph 276次通过。
  Draft KV为torch.equal数值比较，Target当前写入KV为原始字节，Token精确/logits容差另验；
  不混称独立全量Prefill等价。
- **177次真实跨Target Carry快照/恢复Decode关联通过**；
  最终lease/ready/inflight=0，slot collision=0。
- Worker真实K1..8、low_confidence/max_k停止；客户端effective_K等仍null，不用回显作证据。
  Ready/Handoff/Verified各1149，未丢失/重复交接。
  16个首次进入Draft的HTTP请求与曾执行请求出现在后续同批GPU Forward；
  window32 step7 admit请求16，step8与请求8/14实际mixed_prefill_decode，
  3请求24输入行/padding0。逐条匹配实际Forward，不仅计admit标签，也不声称每次补位满batch。
- 功能轨迹真实reused 115126 / prefill 3857 tokens，不是本轮KV复用TPS因果对照。
- **严格整段输出等价未证明**：b6配对文本hash仅146/166/161个一致（各512）。
  闭环批次轨迹不同，单Forward字节精确不等于整段证明；
  未把所有差异直接归为浮点，也未放宽断言。
  本轮没重新覆盖全部取消/自然内存压力/并行GPU组合。

## 收敛与限制

CANDIDATES.md逐项记录。Q8/keys24/精确QK融合保留；
metadata packing和Target32无确认端到端收益、BLAS简单切换未复现收益，不采用。
旧native fused QK改变中间舍入，不用它替换。

另测图外LM-head/置信度：实际共享embedding(151936,2048)、行1..4，10组交错，
64随机hidden样本packed字节相同；每次含D2H/CPU决策，只节省28.7–32.1μs。
按诊断753Forward/32请求外推512，约0.386s/0.14%，**是估计，不是TPS实测或严格上界**。
考虑捕获/cache/状态复杂度，本阶段不接入，未证明所有负载无效。
Query16当前热点不足也暂不实现；不宣称每种Graph状态均已优化。

此前Nsight GEMM/GEMV约占kernel累计92.8%，不是端到端份额/带宽饱和证明。
剩余成本可能在线性层及小batch权重复用，没证据支持简单BLAS替换获益。
Host memcpy API时间不等于DMA，UniProc逻辑RPC不按网络往返估计。
新profile/负载出现可重新打开候选；不研究动态H或网页版算法。

## 复现与清理

REPRODUCE.md含完整实际服务/回退/测试命令；新目录复跑入口防止覆盖旧结果。
serve_exact_on.sh用b148加固库，serve_exact_off.sh只置实验开关0，不结束任何服务。
全部日志/请求摘要/manifest/哈希/源码/构建命令/失败记录保留，代码目录有结果软链接。
只清理已归档可再生.o及本轮pytest缓存，精确路径/哈希见CLEANUP_FILES.json；
不删除历史实验/备份/研究代码/日志/他人文件。

单次22:00监控实际投递2026-11-12T22:00:20Z=11月13日06:00北京时间，晚8小时；
原因未核实，不宣称已证明时区bug。结束后删除本次一次性监控，旧3小时保持PAUSED。

Git文档目录的CODE.patch采用零上下文导出，应用需git apply --unidiff-zero；服务器交付目录的完整上下文CODE.patch原样保留。此差别只影响补丁包装，不改变源码。
