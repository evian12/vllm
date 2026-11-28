CDB-AV 底层基座 d6eb6bd0：已完成限定范围验收并冻结

当前验收状态以本目录RELEASE.json为准。payload/PACKAGE.json保留构建时accepted=false的原始记录；最终验收不改写该已测量载荷，单独在RELEASE.json记录最终验收结论。核心Git版本d6eb6bd0d358b116245fc2384f779cbc65ddb7c7，开发分支codex/cdb-av-350-dev-20261127，myfork远端SHA一致；旧350基线e2228d47及标签保持。

使用环境与独立性
在rtx4090的Linux x86_64、CPython3.11环境执行，解释器/home/yypan/CDB-AV/.venv/bin/python。payload含完整Git源码、原生二进制、运行时补丁、冻结数据、C源与精确依赖索引；运行和复现工具使用本目录相对路径，不需要旧实验源码树。四模型权重与解释器为外部要求，路径/每个文件SHA、ABI及关键依赖版本在payload/PACKAGE.json。支持在满足这些约束的环境复制本目录；未认证任意主机或依赖升级。

在rtx4090上校验全部载荷及外部模型：
/home/yypan/CDB-AV/.venv/bin/python /disk14T/yypan/CDB-AV-runtime/system-bases/cdb-av-base-d6eb6bd0-20261128/release.py verify --models

检查实际启动计划，不启动模型：
/home/yypan/CDB-AV/.venv/bin/python /disk14T/yypan/CDB-AV-runtime/system-bases/cdb-av-base-d6eb6bd0-20261128/release.py serve --profile llama_1024 --seed 42 --plan-only

启动服务（仅GPU空闲、没有其他持锁作业时，NEW_UNIQUE_RUN必须是新目录）：
/home/yypan/CDB-AV/.venv/bin/python /disk14T/yypan/CDB-AV-runtime/system-bases/cdb-av-base-d6eb6bd0-20261128/release.py serve --profile llama_1024 --gpu GPU-b493bb52-b935-1cf4-47fe-7869125775b9 --port 18136 --run-dir /home/yypan/CDB-AV/results/NEW_UNIQUE_RUN --seed 42

复现九例冻结协议（约数小时；输出目录必须全新，不能覆盖归档）：
/home/yypan/CDB-AV/.venv/bin/python /disk14T/yypan/CDB-AV-runtime/system-bases/cdb-av-base-d6eb6bd0-20261128/release.py benchmark --output /home/yypan/CDB-AV/results/NEW_UNIQUE_SUITE --gpu GPU-b493bb52-b935-1cf4-47fe-7869125775b9 --port 18136 --coordinator-lock /home/yypan/CDB-AV/baseline-comparison-20261113/gpu3.lock
benchmark可用--case选择单例；--functional-cold则运行两模型各4条原CodeElo长请求的功能门槛，不生成性能采纳结论。运行结果与锁必须置于本冻结目录之外。完整原始计划/seed/预算保留在tools/CASES.json及payload/data/frozen。客户端原benchmark.py未变，tools/idle_timing_policy.py限制每次空闲selector等待为5ms，策略源码/SHA单独索引。未来复现对所有案例采用此计时策略；现有八例原策略+一例修正策略的来源分开记录，不将新计时策略当作GPU吞吐增益。

配置与验证范围
四个配置llama_1024、qwen_1024、llama_4096、qwen_4096。B4、HF0.8、H8、K上限8、step1024，greedy token-ID输入、ignore_eos及原请求输出预算；自动prefix KV cache关闭，strict输出审计、分块Draft、KV进展预留开启。正常首token单计，此后新输出必须经Target Verify。1024 Target fullGraph与Draft decode/ragged Graph开启；4096仅Target fullGraph关闭，Draft两类Graph保留。--enforce-eager不关闭自写Graph。未验证的采样/API特性不属于本次零AR认证范围。

证据与性能判断
九例合格证据显式由原八例+计时修正后的一例组成，共1545137 token含预热、0 AR输出回退、0请求失败、0测量抢占。另有当前包3个WAITING/RUNNING取消及8192 token冷恢复，CPU/Worker所有权清空、KV恢复651；两模型d6受控抢占、冷prefill与同布局模型数值门槛保留。取消的客户端只接收部分输出，不声称全部交付；受控抢占不代表所有自然压力场景。
原R3最后一例发送延迟P9510.279503ms超过10ms门槛的失败永久保留；修正后P951.253621ms，门槛未放宽，不回写原矩阵成功。evidence/ASSEMBLED_ACCEPTANCE_AUDIT.json列出8旧+1新路径，RAW_TRACE_INDEX.json绑定原始请求/服务日志的远端目录和SHA索引。完整包及四模型哈希已经检查。最终目录已实际启动两模型，各4条长冷请求8192 token，均首4/Verify8188/AR0；21项最终路径包装测试及已有输出拒绝门槛通过。
原350 Llama/ShareGPT1024闭环clientC16/serverB4三seed历史均350.8686；当前均349.7283 tok/s，差−0.3250%。原短上下文Target/Draft Graph保留。不同源码及strict/进展配置，不能把差异孤立归因Graph或宣称统计显著。两模型ShareGPT3×、AIME3×和Qwen/CodeElo3×，以及Llama/CodeElo0.5×是本次开放负载范围；不宣称Llama/CodeElo3×。原native引用GPU1、当前GPU3，不计算跨卡正式加速比。

已知限制与后续算法边界
Qwen/CodeElo3×到达TTFT P95约2258.01秒、E2E约2280.90秒，排队严重。当前651可用KV块，完整Target+私有Draft上下文预留与8块余量下，正式128请求的8128对中7610对（93.62697%）在保守下界已无法并存；其余对不保证可准入。这是容量界限分析，不是精确分配器重放或延迟独立归因。无预定义SLO，零失败不代表高压延迟达标；冻结为有明确边界的正确性/复现实验基座，不声称所有负载最高效率。
4K Target Graph扩域、长指纹缓存未获稳定收益，未采纳；不恢复退休候选。full-vs-chunk超过原容差的历史结果和所有启动/审计/探针失败保留，形状匹配的数值通过证据单独声明。上下文覆盖为冻结工作负载及指定探针，不覆盖全部4096边界或vLLM特性。
BASE_INTERFACE_CONTRACT.json列出当前实际方法位置/AST及所有权、KV重算、正式输出边界。后续算法或KV/准入优化在独立版本迭代，保持正式token、原计划/预算、KV所有权、Worker ACK和strict不变量，并重做受影响门槛。不要直接关闭进展预留。
