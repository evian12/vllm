# 原生CPython类型检查：GPU功能通过，性能待定

源码 fdad8bc258064c6c308554a43e57e26c5746955d，父版本f2ab684c，独立分支codex/cdb-context-native-types-20261120。
默认OFF，CDB_AV_CONTEXT_NATIVE_TYPES=1要求完整上下文缓存ON。
仅将all(type(token) is int for token in tokens)的builtin list/tuple情况替换为相同C循环，
不是C++/CUDA模型kernel，不改变Token值、精度、摘要算法或range验证、完整tuple相等判断。
特殊序列仍走原Python检查，bool/int子类不作为builtin int缓存，负数/超界仍由原hasher拒绝。
持有CPython3.11 GIL、没有回调/Token转换、只读借用指针、不保留Token/Request/Tensor；
原始KV/Carry/进度/取消/抢占/Refill所有权逻辑与断言不变。不改H8/HF0.8/cap8/最终选择策略。

新增csrc/cdb_av_prefix_types.c、CPU构建脚本及17项专项测试；既有辅助模块只改类型predicate
和执行计数日志，模型Runner/Scheduler/Refill执行源码均未改。
现有GCC+Python.h构建，无安装依赖、无CUDA编译；命令/源码与头哈希、ABI、SO SHA见BUILD。
已构建二进制仅在服务器results归档，运行源码用软链接，Git只提交C/脚本/测试/小改动。

CPU基线组合 **216 passed，11 subtests passed**；Native ON原缓存/冻结/专项组合
**41 passed，11 subtests passed**（与216有重叠，不相加当唯一测试数），workflow15通过。
源码提交并正常推送，远端同名分支SHA核对。实际model config与safetensor headers刚复读并匹配：
Target32L/H4096/I14336/head128，Draft16L/H2048/I8192/head64，均BF16；
没有声称校验整个weights数据SHA。

GPU3功能门槛64预热+16短请求全部完成；原摘要独立校验32次通过；
706 Draft Graph/410 Target Graph、215真实跨Target Carry字节检查、1199唯一交接/验证通过，
最终lease/Ready/inflight/释放块为0。日志最后采样有 **22528次真实C predicate调用**，
模块实际导入路径为运行worktree里的CPython .so；不是只看ENV开关。
计数是CPU类型检查调用数，不是GPU Forward数量或TPS证明。
功能服务已释放到15MiB，18084关闭；当前独立512 OFF/ON三组正式实验串行运行。

两种sequence floor1024、完整上下文缓存ON固定；独立变量只有Native类型循环0/1。
同一512冻结ShareGPT、C16/seq4、每轮101003输出token，128×128预热排除，重诊断正式关闭。
三组固定顺序与保留准则在首轮结果之前写入CRITERIA.json，不追加挑选幸运结果。
**性能尚待全部三组，当前推荐值仍是上一f2ab684c配置的286.883±1.115 tok/s，
不能把它标成本候选成绩。** 严格输出等价及超过历史Native SD4均未达到。

完整GPU启动命令见gate_native/launch.sh；可复制短功能/安全回退命令：

```bash
cd /home/yypan/CDB-AV/results/context_native_types_20261120
/home/yypan/CDB-AV/.venv/bin/python driver.py --tag "native_smoke_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --seed 42 --native-types 1 --diagnostic --requests 16 --max-tokens 64 --gate-warmup 64
```

回退只改--native-types 0，完整上下文缓存仍ON，保留所有原断言；--dry-run只检查命令。
复测先确认当前controller已完成且GPU3/18084空闲，不并行抢占。每轮服务finally退出并核对释放。
