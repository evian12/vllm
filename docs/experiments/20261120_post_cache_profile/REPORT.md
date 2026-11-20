# 推荐配置后的短时底层复查

源码f2ab684c06ba4f41fb9ac4f6afa2393695ed8444，两个sequence floor1024、
完整上下文摘要缓存ON；H8/HF0.8/cap8/C16/seq4固定，BF16/FA2/noasync，
未改变请求选择。128×128token预热排除，仅16个最多128token短请求，64个Target迭代。
**这不是512性能比较，不用诊断TPS宣称收益**。22项CPU instrumentation/解析测试通过；
安全hex标签实际GPU trace合法JSON，直接解析，不需旧实验的标签转义修复。

64 Target backbone GPU kernel合计约1027.363ms（16.053ms/次），79 Draft合计约224.674ms
（2.844ms/次），logits另列Target71.416ms、Draft44.070ms，不能重复计入backbone。
输入shape不同，这不是同shape模型速度基准。Draft仍明显更快。
Target 61/64 replay、3次budget fallback，Draft74/79 Graph、5次eager。
主导WMMA GEMM一个kernel家族约1246.563ms，占该分组全部kernel时间约90.4%；
这不是端到端时间比例，也不能证明DRAM带宽已饱和。

CPU reconcile 243次/32.455ms，均值133.559us；完整type检查仍在每次缓存命中前逐Token执行。
可做最小独立候选：只把all(type(token) is int)搬到等价CPython C循环，
保留完整tuple相等、原始摘要、越界/特殊类型回退、冻结与KV所有权所有断言。
这项尚无GPU功能/正式TPS结果，不能将C循环更快直接当作系统收益。

实际executor仍为UniProcExecutor/backend uni，同进程调用；p2_step79次包含432.554ms，
扣除嵌套CDB范围后的6.775ms仍含框架、Python和Profiler开销，不是网络IPC时间。
Target sampling CPU包含约1041.746ms异步GPU等待，不能和Target GPU约1027ms相加。
KV/Metadata小项既有CPU等待也有内部嵌套，不能按inclusive字段相加来预计可加速上限。
已否定的两类Metadata融合、padding/FP32与本轮column布局不盲目重测。

只读检查发现ncu未安装，驱动RmProfilingAdminOnly=1，本进程CapEff=0；
尚无真实DRAM硬件计数证据，不修改共享驱动权限或安装依赖来绕过。
没有记录kernel的timeline区间也不等于GPU硬件idle。

16/16完成，实际导入路径/源码与绑定哈希、环境、请求响应、trace和完整启动命令均保留。
独立服务已关闭，18084关闭、无本轮GPU PID、GPU3回到15MiB。其他GPU进程不处理。
完整启动命令见post_cache_cap8/launch.sh；安全复测使用新tag并自动清理：

```bash
cd /home/yypan/CDB-AV/results/llama_post_cache_profile_20261120
/home/yypan/CDB-AV/.venv/bin/python profile_driver.py --tag "profile_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --diagnostic --requests 16 --max-tokens 128
```

只审查参数可加--dry-run。调试服务只能在GPU3和18084空闲时启动；不能重复已有运行。
