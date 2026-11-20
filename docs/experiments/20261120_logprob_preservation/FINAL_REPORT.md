# 未请求logprobs时避免保存完整logits

源码4baa0d1378b27cc1caaa64bac4909a72ab8394b8，独立分支codex/cdb-logprob-preserve-20261120。默认关闭，开关CDB_AV_SKIP_UNUSED_LOGPROBS。只在max_num_logprobs为None时取消候选raw logits副本和bonus完整logits保存；请求0、指定条数、全量或processed/raw logprobs时保留原路径。实际采样、softmax、接受/拒绝、HF0.8、H8、等待队列与KV守卫不变。

35项CPU模式/返回值/处理器分支测试通过；18项实际GPU sampler回归通过（19项未选入，不称完整测试集全通过）。诊断钩子另有3项CPU隔离与启动dummy回归测试。首次门槛在启动dummy随机采样时被过严诊断断言拦下，发生在请求前；修正钩子使非适用case原样运行，运行源码与安全标准未改。失败日志保留。

GPU模型门槛64预热+16短请求通过，32批相同输入旧/新sampler token完全相同。Draft/Target Graph检查1034/246次，Carry跨Target字节检查267次，唯一候选验证1644次，未分类AR行零。它们是受检输入的正确性证据，不等同于完整生成轨迹严格等价。

按用户要求只做一次ON筛选，不重复OFF。GPU3、Llama8B/1B、BF16/FA2/TP1、C16、max_num_seqs4、H8/HF0.8/cap8、等待队列开启；编译、replay host关闭，Target缓存24、双sequence floor1024。128×128token预热排除，正式冻结512ShareGPT请求、101003输出token。

首轮 **312.661 tok/s**，相对已验收历史 **312.565±0.499 tok/s** 为 **+0.031%**；E2E P95 **17.831秒**，相对历史17.627秒变化+1.154%；TTFT P95 **0.101秒**。这是单轮历史参考，不是配对因果估计。

筛选结论：**未达进展门槛，停止并保持候选关闭**。预登记门槛为TPS至少+0.5%、E2E P95最多增加1%。不固定追加三轮，不挑幸运样本。历史原生341.073±5.497未重跑；本轮相对历史原生-8.330%。当前正式推荐尚未改变。

所有本轮服务已退出，18084关闭，GPU进程释放。launch/manifest/summary/released保存实际配置，原始请求和日志保留服务器。复测本目录driver.py --tag 新标签 --mode hf --k-cap 8 --seed 42 --skip-unused 1；回退值0。后续测试仅在首轮值得确认时进行。
