# Llama Target物理column布局：不接入

实际读取Llama3.1-8B的qkv/o/gate-up/down/head BF16权重，20组shape、M=1/4/8/16，
按真实config/shape核对，除head外轮换不同层，总权重每组至少256MiB以限制L2热缓存偏差。
单独进程、固定GPU3、8次交替CUDA-event采样/原语，不启动服务、不修改任何checkpoint、
模型参数值、dtype或运行时源码。3项CPU计划测试通过。

**19/20组更慢**，唯一正值head M16约+0.452%，不足以支持系统收益；
10/20组输出逐字节一致。全部逻辑权重逐字节一致，仍有输出误差；
head M1一个随机hidden行的argmax不同，完整误差保留。随机层输出不是最终生成Token。
既无一致收益又有数值风险，不接入、不运行额外512对照来碰运气。
这不是全局排除全部GEMM内核，而是排除本组native/column physical layout变体。

本原语等前序六轮配对完成后才取得共享锁，进程20秒内退出；RELEASED记录GPU3 15MiB，
18084没有服务。其它GPU/进程保留。原始样本、配置哈希、权重名/shape和完整结果见RESULTS.json。
复现需先确认无其它实验，使用本目录controller；它核对前序状态、GPU/端口并自动释放，
不能直接并行运行probe.py争抢GPU。
