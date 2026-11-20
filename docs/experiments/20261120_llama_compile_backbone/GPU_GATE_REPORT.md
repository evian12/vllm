# Llama Backbone 编译 GPU 功能验证

源码 ec0becf70c1fb9060ead21579e786f29a0ba8234。Llama 8B/1B，C16，max_num_seqs=4，H=8，HF confidence 0.8，FA2 BF16。保留 CDB CUDA Graph，vLLM native Graph 为 NONE。Target replay 主机开销选项关闭。只在此实验中启用编译。

初次编译完成了 Target 和 Draft backbone。默认范围 1–1028 的上界被选作预热输入，超过 1024 的批次预算并触发启动断言。重试在两臂都显式指定 compile_sizes=[1024]，保留了预算与断言。完整首次失败与自动释放记录保存在实验目录。

重试通过 64 请求预热和 16 请求测量。编译后的 Graph 与原始 eager 路径比较，校验用 Forward 在本次调用内显式绕过编译。Target/Draft Graph 检查 411/706 次，跨 Target Carry KV 字节检查 216 次，唯一交接与验证各 1202 次。关联原始补位日志与实际 GPU 调用，确认 967 次有 Ready 请求且与存活行合批的补位。GPU 服务已退出，18084 关闭，显存 15 MiB，无该服务 GPU 进程。

启动日志记录编译累计约 69.5 秒。端到端配对的正式测量另以冻结 512 请求、每臂 101003 token、三组 seed 运行，预热排除。当前仅报告 GPU 功能通过。最终生成等价和吞吐收益需由正式配对输出和结果判断。
