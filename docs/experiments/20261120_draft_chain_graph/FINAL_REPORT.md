# Draft 完整计算链 CUDA Graph 试验

源码 253d603af54b50eda4bddd4adaf2b5e4f1b212bd，分支 codex/cdb-draft-chain-20261120，开关 CDB_AV_DRAFT_CHAIN_GRAPH 默认关闭。从已验收73d37c98基线出发，仅合并 Draft backbone、动态末行提取、LM head、原生FP32 HF confidence和FP64结果包为单个Graph。调度、等待队列、HF0.8、KV ownership与回退路径保留。

这与11月13日独立post Graph试验不同：旧方案增加静态hidden拷贝和第二次Graph launch，均值仅+0.301%且未采用；本次去掉这两个边界，以单Graph覆盖整条GPU计算链。未重新开启旧post Graph，也未叠加无收益logprobs候选。

代码已实现，60项相关功能测试和15项版本流程测试通过。GPU功能门槛64预热+16短请求通过；1034次合并链logits与HF结果包逐字节一致，其中569次为动态布局。Draft/Target Graph检查1034/246次，Carry跨Target检查267次，唯一验证1644次，未分类AR行零，清理状态零。它们是受检输入的证据，不等同于所有生成轨迹逐字节等价。

GPU3、Llama3.1-8B/3.2-1B、BF16、FA2、TP1、C16、Target max_num_seqs4、H8/HF0.8/cap8，等待队列开启；编译及replay host关闭，Target缓存24、双sequence floor1024。128×128token预热排除；正式512冻结ShareGPT请求、101003输出token、零失败。

唯一ON首轮 **311.365664 tok/s**；历史已验收 **312.564731±0.498818 tok/s**，变化 **-0.3836%**。E2E P95 **17.845717秒**，相对17.627044秒变化 **+1.2406%**；TTFT P95 0.095207秒。此处为单轮历史参考，不作配对因果归因。原生历史341.073279±5.496631 tok/s未重跑，本轮相差8.7100%。

结论：未达进展门槛，停止；保持候选关闭，当前推荐不变。预登记筛选要求TPS至少+0.5%且E2E P95最多增加1%。没有补跑OFF或无必要的后续轮次。

服务已退出，18084关闭，本任务GPU进程已释放。原始日志、请求结果和数据集保留服务器results/draft_chain_20261120；源码、manifest、启动命令、摘要与释放记录归档。复现：driver.py --tag 新标签 --mode hf --k-cap 8 --seed 42 --chain-graph 1；回退值0。
