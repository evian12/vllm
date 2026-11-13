# CDB-AV Draft LM-head + native HF confidence Graph
## 结论与状态
本轮为Partial Implementation；默认CDB_AV_DRAFT_POST_GRAPH=0。
冻结运行源码629c220b364e6a2806292a01c31b49caaeb46d27，独立分支codex/cdb-draft-post-graph-20261113。
在既有Pinned1、精确QK both1、Draft/Target Graph配置上，仅切换新的Draft post Graph。

三组同配置配对：
OFF 239.150±1.282，
ON 239.866±0.631 output tok/s；
配对平均+0.301%±0.540%（样本SD）。
P95各轮均值：OFF 10.4069±0.0751s，
ON 10.4164±0.0442s；
配对P95变化+0.093%±0.370%。
固定三组，无看结果后追加或挑选最好单轮。
组件节省几十微秒，但三组配对收益未稳定超出波动，P95略变差且观察到60–88MiB的峰值显存差，
**本阶段不推荐默认开启**；保留默认0作为生产/论文推荐配置，实验代码独立保留。
不能把开启后的绝对TPS与前轮异运行版本直接相减归因。
严格跨运行全段输出等价、所有自然内存压力生命周期与全局最优仍未证明。

|组|OFF TPS|ON TPS|配对变化|OFF P95s|ON P95s|
|---|---:|---:|---:|---:|---:|
|0|237.703|239.843|+0.900%|10.4917|10.4581|
|1|240.144|240.509|+0.152%|10.3489|10.3700|
|2|239.603|239.247|-0.149%|10.3800|10.4211|

## 代码修改
1. vllm/v1/spec_decode/cdb_av_post_graph.py：
   默认关闭的有界四形状缓存；捕获原生compute_logits及原有FP32 max/logsumexp/exp、FP64 transport打包。
   每次拷贝新hidden；参数/processor地址与布局检查、单线程/单stream/非嵌套捕获约束；
   出错拒绝复用；可开启前64次（1..256可配置）logits/packet原始字节检查。
2. vllm/v1/spec_decode/cdb_av_refill.py：
   只在两个原logits调用点选择新分支，HF分支消费对应logits身份的packet；
   每次Forward清除旧pending，未启用时保持原数学运算。
3. tests/v1/spec_decode/test_cdb_av_post_graph.py：
   CPU协议/停止判断/默认路径测试；CUDA动态hidden、1..4行、head替换、缓存边界、异stream拒绝及失败后拒绝复用。

不修改模型权重数值、dtype或文件、不改Tokenizer、不改KV reuse/lease/commit、
Carry/Refill调度、请求选择、K策略、H及停止规则。未整合网页版算法修改。
新缓存只持有head/hidden/logits，不持有请求或KV页。
缓存输出借用到下一次调用，旧有阻塞D2H在下一次Draft调用前消费；不能移除串行约束。

## 实验配置与可比性
Qwen3-8B/Qwen3-1.7B，RTX4090物理GPU3(内部cuda:0)，端口18084，
vLLM0.15.1/Torch2.9.1+cu129/BF16/FA2/TP1，
maxseq4/maxlen1024/seed42，H8/HF.8/Kcap8固定。
outer enforce-eager配合既有CDB手动Graph，显式no-async/no-prefix。
每臂新服务128请求/C16/max128预热，再512/C16/max128正式。
客户端闭环并发16，不是匀速到达，也不是GPU batch16。
六臂3072/3072成功、393216输出token，另768warmup请求。
唯一有效配置差异为CDB_AV_DRAFT_POST_GRAPH；source/client SHA、完整环境及启动命令逐臂保存。
重型metadata/Graph/head/Carry审计在正式性能阶段关闭；相同1Hz NVML采样。
三组ON/OFF峰值显存差分别62/60/88MiB，完整统计见RESOURCE_METRICS.json；
NVML峰值也受其他Graph缓存轨迹/allocator影响，不当作新模块独占allocated字节。
非streaming elapsed含排队和整段生成，未测TTFT/ITL/TPOT。
时钟/温度、内部UUID和并发完成顺序未完全固定；输出hash并不全部相同。
不将hash差异一概归为无害浮点误差，也不把HTTP请求K回显作Worker证据。
本轮不重跑原生AR/SD，不拿历史异配置TPS作对照。

## 正确性与运行证据
- 完整CPU447 passed,19显式GPU-only skipped；版本workflow15通过。
- CUDA专项命令7 passed包含1项CPU guard和6项真实CUDA测试，不能混报成7项GPU。
- 真实32请求/4096输出：64批logits和packet逐字节相同，全部关联实际请求/窗口/step/phase的后续Forward。
- 该64样本覆盖行1/3/4；后续独立2请求/64输出补到行2：
  8次真实行2字节检查，全部与对应Forward匹配。
- 原有Draft Graph 636次、Target Graph 251次检查通过。
- 157次28层Draft物理页字节检查，
  其中157次跨真实Target；累计比较2069889024字节。
  累计字节含重复页，不代表独占显存容量。
- Ready/Handoff/Verified均1135；actual K1..8，HF low_confidence/max_k实际生效。
- 主功能审计最终Lease/Ready/inflight为0；碰撞/指纹/进度断言保留，没有CUDA非法地址/NaN/越界/shape错误。
- 新Graph在实际服务中捕获/复用了行1..4，缓存最多4形状，性能阶段验证计数为0。

## 瓶颈与未采用候选
详见BOTTLENECKS.md、WORKLOAD_AUDIT.json和TARGET_WORK_AUDIT.json。
32请求重型功能数据中，Draft主Forward平均有效行3.60，Target平均调度3.87、验证候选1.95。
976个工作项在已有其他成员Ready后实际加入同窗后续Forward；不是976个唯一API请求。
77次无验证候选的Target调用均有正数scheduled token，零scheduled调用为0；
不能仅凭无候选跳过真实Prefill/AR工作。
独立全矩阵物理布局30项：23项变慢，16项输出字节不一致，未接入。
选择性down-projection布局仍是未验证候选，见NEXT.md；不宣称所有底层优化已穷尽。
此前Target32/seqfloor256/BLAS偏好/仅Metadata合包已有对照，不重复盲测。
Graph减少host提交，无法免除每步矩阵乘及权重读取，也无法保证接受深度收益覆盖多次Draft成本。
前轮Target分段计时与本轮组件微测均不冒充本轮正式TPS或真实DRAM利用率。

## 同步、复现与资源
代码已正常push；最终交付SHA/远端核对见GIT_DELIVERY.json。
REPRODUCE.md与serve_post_0/1.sh包含完整命令，回退仅CDB_AV_DRAFT_POST_GRAPH=0。
旧2c12933工作树、root未提交备份、历史日志/基线全部保留。
代码目录提供结果软链接；只清理本轮可再生缓存，清单见CLEANUP_FILES.json。
全部测试完成即关闭本轮服务并核对GPU3/18084，见CLOSEOUT.json。
论文AR/nativeSD/CDB统一no-async，但原生CUDA Graph应独立保留适当配置并记录。
