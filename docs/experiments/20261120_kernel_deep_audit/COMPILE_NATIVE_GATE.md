# 原生 Torch 算子编译路径的功能门槛

源码73d37c98，等待队列开启，H8/HF0.8/cap8固定。此前编译实验保留custom_ops all，本次首次验证mode3/custom_ops none，允许原生Torch算子融合。诊断参考显式使用当前CUDA算子，编译捕获路径不受参考钩子影响；1项CPU隔离测试通过。

实际GPU首个Draft decode capture，rows1、sequence bucket64：token_exact=true，但KV exact=false、logits_close=false、max_abs=0.25。当前安全门槛触发并退出，没有运行正式性能对照，没有修改容差或关闭断言。不能用这一处相同argmax证明后续HF停止和输出不受影响。具体数值差异尚未逐层归因，该整条替换路径当前不采用。

服务和端口已释放，GPU3显存15MiB。原始编译、请求和错误日志保留服务器results/draft_wait_compile_native_20261120。下一步不重复此配置的长性能试验。
