# 固定Target Graph序列上界：配置验收，性能待定

源码 `23eaa57d50af9c740088d0f375acc59c3cf2474d`，父提交008c04f8。**运行时源码未改动**，只新增6项CPU回归测试；
验证现有CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0/1024开关。这里1024是Graph launch上界，
不是H、K或实际上下文长度；H8/HF0.8/cap8不变，max_model_len仍1024。
新上下文Token、位置、query offsets、GPU seq_lens、KV slot/block table保持实际值；
没有Mask或额外请求行补齐。query1/混合路径、table宽度、split状态等键域仍不混合。

选择原因是短时推荐配置样本发现24份捕获只有10个不含序列上界的键域、35/64次
Target因budget回退；固定上界可能减少重复捕获，但也可能改变attention kernel成本。
**尚未取得本配置重复吞吐结果，不能用命中率/捕获数量宣称提速。**

CPU186项通过，新增6项和版本工具15项通过。GPU实际Llama模型64预热+16短请求完成，
706次Draft Graph与410次Target Graph检查、
215次跨Target Carry字节检查通过；1199唯一候选交接=
1199次验证。所有410次实际Target捕获/回放的key序列上界均1024、
padding0；只使用17个实际键，最终lease/Ready/inflight等状态归0。
功能服务已退出并确认15MiB/端口关闭。实际模型config和safetensor header已重新读取，
记录BF16与真实矩阵维度、Draft共享embedding，不加载或修改权重；无全部权重数据SHA声明。

正在执行同代码三组512请求0/1024配对（seeds42/43/44、交替顺序），GPU3/18084，
相同Llama8B/1B、冻结ShareGPT/C16、服务seq4、BF16/FA2/TP1/eager+manual Graph、
no-async/no-prefix-cache，len1024/budget1024，128/C16/128token预热排除，每轮101003输出token。
Draft floor1024固定，Target兼容bound、GEMM、两类Metadata优化均OFF。
重型审计与Profiler关闭，每轮结束立即回收本轮服务并核对PID/端口/显存，失败停止。
原生基线不重跑；输出差异照实保留，不等同于严格生成等价验收。

完整启动命令在gate_floor及各pair的launch.sh中。安全回退为同代码
CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0，其它配置保持一致，不reset/checkout覆盖。
此文件为功能门槛快照；后续以COMPLETE/ANALYSIS和逐轮released记录作最终性能判断。
原始日志、请求、模型头、哈希在服务器target_seq_floor_20261120目录，源码与主vllm均
有cdb_av_target_seq_floor_results软链接。GitHub源码提交已正常推送并核对远端SHA。
