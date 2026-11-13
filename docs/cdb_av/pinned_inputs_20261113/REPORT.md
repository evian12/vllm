# CDB-AV 完整 Draft 输入 staging — 2026-11-13

## 结论

本轮实现可关闭的`CDB_AV_PINNED_INPUTS=1`，默认0。
把每个Draft Step的IDs、Positions、query starts、seq lengths、block table、slots、last indices
合并为一次锁页CPU→GPU非阻塞传输。CUDA Event保护CPU源缓冲复用/扩容；
GPU payload和CPUquery shadow仍每次新建，不复用前一批有效数据。
不修改KV租约、请求选择、K/H、停止规则、Target验证或模型权重。

有限五组结果支持小幅正向收益，仍保留默认0及回退开关。
OFF **237.923±0.888**，
ON **239.826±0.896 output tok/s**；
配对平均 **+0.801%±0.632%**，
4/5组正向。五组配对均值的t区间仅作探索性参考：[+0.017%,+1.586%]；
第三组后固定追加至五组的决定已保存，不把该顺序审查当正式预注册显著性检验。
整体仍为Partial Implementation，严格整段输出等价及全局最优未证明。

## 代码与隔离

- 测试源码`1096679a1401517852a18248564bc8f66b181508`，
  分支`codex/cdb-pinned-inputs-20261113`；
  实现`a7f42ad144efd5e2e1b52de37956465c6ba63263`，后续109增加有界真实输入字节审计。
- 工作树`/home/yypan/CDB-AV/worktrees/cdb-pinned-inputs-20261113`。
- 前一239版本f3e148b工作树及Root e6b5和原有9个未跟踪备份均保留，未整合网页版算法。
- 只用物理GPU3/18084，其他实验不结束；最终GPU3=15MiB、端口关闭。
- 沿用精确QK融合both ON及既有b148私有库，未改CUDA kernel/依赖/权重/Tokenizer。

修改文件：
1. `vllm/v1/spec_decode/cdb_av_inputs.py`：CPU整数布局/编码、锁页源缓冲、完成事件、错误后拒绝复用、
   新鲜GPUtyped views/CPUshadow、受控域及有界真实字节审计。
2. `cdb_av_refill.py`：仅在Step已有ownership/position/member检查之后选择新传输分支；
   默认0走原输入创建路径，不跳过断言、不增删Forward。
3. `test_cdb_av_inputs.py`：布局/整型边界/非法输入、默认不开CUDA、控制状态回归、真实GPU异步复用/扩容及审计边界。

## 公平实验

Qwen3-8B/Qwen3-1.7B，RTX4090 GPU3(内部cuda:0)，vLLM0.15.1/Torch2.9.1+cu129，
BF16/FLASH_ATTN/FA2、TP1、max_num_seqs4/max_model_len1024/seed42。
H8/HF0.8/Kcap8固定；Draft Q8/mixed24、Target24、packed-metadata0、QK融合1/both。
全部显式`--no-async-scheduling`，不在论文对照中混用默认async值。

每臂新服务128请求/C16/128输出预热，再512/C16/128输出正式测量；
每轮65536输出Token。闭环固定并发16，不是GPU Batch16或匀速到达。
唯一有效配置差异`CDB_AV_PINNED_INPUTS`；相同源码/ClientSHA/服务参数/监测/重型trace关闭。
顺序OFF0/ON0/ON1/OFF1/OFF2/ON2/ON3/OFF3/OFF4/ON4。
最初三组小幅混合结果后，固定补到五组；不反复追加至正向，不挑最高单轮。

|组|OFF tok/s|ON tok/s|配对变化|OFF P95秒|ON P95秒|
|---|---:|---:|---:|---:|---:|
|0|238.483|240.248|+0.740%|10.3147|10.4089|
|1|238.699|238.610|-0.037%|10.5790|10.4207|
|2|237.695|239.144|+0.610%|10.5810|10.5244|
|3|236.481|240.513|+1.705%|10.4994|10.3542|
|4|238.255|240.613|+0.990%|10.4103|10.4103|

P95各轮均值±样本SD：
OFF 10.4769±0.1146s，
ON 10.4237±0.0620s。
10轮**5120/5120完成，655360输出Token**；另1280warmup，实际各128输出。
非streaming elapsed含服务等待；未测TTFT/ITL/TPOT。
NVML1Hz不是kernel-active/SMoccupancy；内部UUID/线程完成顺序/动态batch不能由seed完全固定。
不同运行的输出文本hash并不全相同，不将其归为已证明无害的浮点差异；
原始匹配数逐组保留，不把客户端K回显/开关作Worker执行证据。

## 验收证据

- 完整CPU **430 passed,13 GPU-only skipped**，聚焦CPU30项；
  真实CUDA8项通过，包含刻意排队DMA后复用/扩容、CPUshadow默认设备、独立GPU存储及有界审计。
- 组件10组交错：单请求242.62→163.12μs，decode4 350.32→156.61μs，
  ragged362.85→164.06μs，长短混合368.70→166.38μs。
  这是初始a7组件区间，不能代替109真实模型/吞吐。
- 真实模型32/32、4096输出；
  **64个真实输入样本、448个整数字段原始字节检查**与native CPU参考一致，
  并逐条关联后续GPU Forward的实际请求ID/窗口/step/行数。
- 679次28层Draft Graph数值KV/Token/logits检查、
  240次36层Target当前写入KV原始字节/Token/logits检查通过。
- 32请求的153组跨Target Carry是CPU状态/上下文及实际Decode关联，
  **不把它混称物理KV字节检查**。后续16请求独立检查物理页，见CARRY_ACCEPTED.json。
- 物理Carry检查：16/16完成，
  37个28层页字节比较，其中37个覆盖真实Target阶段；
  未用虚拟resume计数代替。最终Lease/Ready/inflight为0。
- Worker实际K1..8、HF low_confidence/max_k；32请求Ready/Handoff/Verified各1180。
  所有槽位碰撞/指纹/进度断言保留，未出现CUDA非法地址/越界/NaN/shape异常。

## 瓶颈解释及后续

输入准备组件加速不等于全系统加速：GEMM/GEMV权重读取/计算、LM-head、
GPU执行等待及batch轨迹仍占主要成本。CPU memcpy API墙钟不等于PCIe传输时间。
本轮同时保留每组实际stager累计前缀counter，包括source wait/capacity/growth；
这些含warmup且未必覆盖最终所有Forward，不冒充完整工作量或性能归因。

后续小范围诊断记录8个真实Target replay的storage检查、六缓冲复制和模型区间，
详见TARGET_TIMING.json；其重型审计运行不能与正式TPS混比。
此前Target32、序列floor256、仅Metadata合包、BLAS简单切换无可靠系统收益，避免重复。
新的矩阵物理存储布局仅提出独立clone组件候选，尚未运行/接入，未改模型权重。

用户已决定论文AR/原生SD/CDB统一async OFF。当前0.15.1独立draft_model本身不支持原生async；
不删除scope限制来假装支持。ASYNC_PROTOCOL.json/ASYNC_AUDIT.md记录约束，
原生async兼容开发暂缓；CPU→GPU非阻塞输入传输与原生async scheduling是不同机制。

## 复现、回退与资源

REPRODUCE.md及每臂实际launch/config文件提供完整命令。
回退仅`CDB_AV_PINNED_INPUTS=0`；旧f3e工作树保留，不需reset/覆盖checkout/改依赖。
不直接重跑已有输出目录，使用新的结果目录，保存manifest/导入路径/SHA。
代码目录结果软链接、原始日志/请求摘要/源码patch/脚本/哈希全部保留。
仅归档或清理本轮确认可再生的缓存，不删除历史结果/备份/研究代码或他人文件。

详细分阶段解释见BOTTLENECKS.md：Target replay模型区间中位16.37ms，LM-head1.39ms；CPU storage检查0.117ms、六copy区间0.070ms。不同区间不能简单相加为完整请求耗时，诊断数据不用于TPS对照。
