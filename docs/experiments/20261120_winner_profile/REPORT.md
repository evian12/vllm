# 推荐配置的短时底层性能复查

运行源码008c04f89f319b51b26187e81f93406b2322b9c9；Llama8B/1B，GPU3，
H8/HF0.8/cap8/seq4、BF16/FA2/TP1/no-async，Draft sequence floor1024；其它未获
吞吐收益的实验开关均OFF。128请求预热后，16个短请求（每个最多128token），延迟1.5秒
开启最多64个Target迭代的Torch采样，record_shapes/stack/memory关闭。16/16完成。
这是诊断样本，**不报告为512性能结果，不用短样本TPS替代配对结果**。

实际配置和safetensor头另见Target序列上界实验ACTUAL_MODEL_HEADERS.json：Target
32层/H4096/I14336/32Q头8KV头/head128；Draft16层/H2048/I8192/32Q头8KV头/head64；
vocab128256，权重头BF16。Draft的tie_word_embeddings=true，原始checkpoint不保存
独立lm_head.weight；按llama.py的tie_weights/skip_prefixes路径解析到embedding权重。
这是检查器对共享权重的正确处理，没有修改权重或加载逻辑；未声称已校验全部权重数据SHA。

## 真实采样证据

64个Target Forward的GPU backbone内核耗时合计1029.544ms，
77个Draft合计218.802ms；平均约16.087与2.842ms。
两者输入shape不同，不能当作同形状微测；但Draft Forward确实明显更快。
Target logits71.424ms、Draft logits42.965ms另列，未重复计入backbone。

Target有35/64次因捕获预算满而回退eager，仅29次replay；没有大于36行的Target
Prompt Forward混入这64个调用。实际缓存24条，去掉**仅序列上界字段**后剩10个键域；
宽度、query1/混合边界、行数、request数、split状态等仍全部区分。35次预算回退中，
19次与已有键仅序列上界不同。这是缓存碎片的直接证据，**不是1024配置已提速的证据**。

Draft有72/77次Graph、5次eager；本阶段优先验证Target缓存键碎片，而不盲目重复小拷贝
融合。此前Target兼容大bound复用、GEMM补齐、两类Metadata实验均未证明稳定吞吐收益。

运行实例确认executor为vllm.v1.executor.uniproc_executor.UniProcExecutor、backend=uni，
同进程collective_rpc直接run_method；API到EngineCore仍有进程边界，两者不混淆。
p2_step调用77次，包含的Draft/采样时间约402ms；扣除嵌套CDB范围后的约5.667ms仍包括
框架和Profiler开销，不能当作纯IPC成本。reconcile有240次、CPU范围合计51.339ms；
可列为后续CPU审查点，当前没有通过删除安全检查或跳过状态回收优化它。

CPU范围含GPU等待，不能把Target eager方法742ms与GPU562ms简单相加，也不能把
Graph方法只占35.949ms理解为GPU只需35.949ms。Kernel时间求和与GPU活动union不同，
无活动记录的间隙也不直接等于硬件idle。所有数值受短时Profiler扰动，只有正式配对可
决定采用与否。当前目标仍未达到原生SD4历史341.073279±5.496631 output tok/s。

## 采样标签错误及保留证据

初次离线解析失败：我新增的record_function名称包含JSON key中的双引号，Kineto
没有转义，导致trace JSON不合法。GPU请求本身正常完成，服务已关闭；原始trace和
原STATUS/错误日志均保留，**没有重跑GPU来掩盖问题**。
仅对已知CDB标签name字段的JSON转义生成新副本：208个已知名称中156处需要转义，
解析后238707事件。逻辑名称恢复为原字符串；其余事件字节、时间戳、持续时间、顺序和
相关ID均不改动。TRACE_REPAIR.json保存原始/副本哈希与逐标签统计。后续安全标签用
hex编码；新增实际CPU Kineto导出测试验证可合法JSON解析，累计22项CPU测试通过。
历史profile_hooks.py及PLAN记录的是实际运行的原版；safe_labels是后续修正版，不能
把它冒充本次运行源码。分析恢复后STATUS标记completed，并引用原失败记录。

此前8项需显式GPU预留的pinned-input测试也已补跑：完整文件38/38通过、无skip；
测试了真实CUDA transport、未完成源拷贝等待、增长、CPU shadow、实际dtype与新GPU存储。
该原语进程与短时模型服务均已退出，端口18084关闭、无本轮GPU PID、GPU3恢复15MiB。
之后新的Target上界对照仅测试期间占用GPU；不停止他人进程。完整实际启动命令保存在
capture/launch.sh；勿用包含Profiler的短请求数据进行TPS比较。
