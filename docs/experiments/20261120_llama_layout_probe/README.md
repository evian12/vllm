# 预注册：Llama 实际权重的物理布局组件检查

仅在当前指纹缓存六轮配对实验完成且 GPU3 空闲后运行。现有端口和服务不改动；
独立子进程持有共享锁，最多 240 秒 GPU 工作，退出释放。原生基线不重测。

此前 Llama 的 row-padding 与 Draft FP32 down-projection 已测过，后者端到端无稳定
收益；本轮不重复。旧 Qwen 的整体 column-layout 大多更慢，不能直接作为 Llama 结论。
这里先做 Llama Target 的组件筛查：核对实际 config / 原始 BF16 tensor，20 个 shape，
qkv/o/gate-up/down/head，M=1/4/8/16，每组轮换至少 256 MiB 的权重（超过 4090 L2）。
除 head 外使用真实不同层，不用热缓存单矩阵替代模型权重流。

仅物理 strides 不同，逻辑权重逐字节相同；不改 checkpoint、模型参数值、dtype、
调度、H、HF 或运行中的 vLLM 源码。输入是随机 hidden state，因此输出 argmax 仅为
线性层诊断，不是生成正确性证明。保留所有误差，不以容差或 argmax 相同宣称逐字节相同。

采用门槛：只有在实际常见 M 上存在一致、有意义的组件收益，才考虑单独 opt-in 的
运行时实现；之后仍须真实模型 Graph/KV/Carry/Verify 功能门槛和多轮 512 配对验证。
若收益不一致、成本或数值风险高，则不接入。任何组件数字都不能声称 CDB TPS 提升。
