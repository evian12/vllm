# 完整前缀指纹缓存：功能门槛通过，性能待定

源码 f2ab684c06ba4f41fb9ac4f6afa2393695ed8444，父版本23eaa57d；独立worktree /home/yypan/CDB-AV/worktrees/cdb-context-fp-cache-20261120。
默认OFF，开关CDB_AV_CONTEXT_FINGERPRINT_CACHE；诊断CDB_AV_CONTEXT_FINGERPRINT_VALIDATE。
只修改RefillQueue.context调用，仍复制正式prompt+output并对完整builtin-int tuple逐项比较；
相同Request对象/同一原始hasher/完整Token前缀全部匹配才复用旧摘要。不是按ID/长度缓存。
BLAKE2b编码及范围校验原函数不变，所有冻结上下文、computed长度、取消、抢占、所有权
断言及Worker执行不改。弱引用、64条/1024token上界、不支持对象或跨线程回退原路径，
返回新dict，安全路径OFF可立即回退。本分支不整合网页版算法。

CPU **199 passed，11 subtests passed**，新增19辅助测试+5实际caller/冻结状态AST测试；
workflow15通过，validate与checkpoint通过，GitHub同分支SHA已核对。
此前入口误写了不存在的fingerprint测试文件，之后fixture缺少window字段导致1项失败；
两次失败记录保留，补齐真实fixture并重跑，没有修改核心断言或降低预期。

真实GPU3门槛：64预热+16短请求全部完成。32次缓存/原始摘要审计一致并有真实RequestID，
实际导入路径、运行日志命中计数都匹配本worktree。706 Draft Graph、
410 Target Graph的KV/token/logits诊断通过；215次真实跨Target
Carry字节审计通过。1199唯一候选交接与验证一一对应；最终Lease/Ready/inflight/释放块为0，
功能服务已关闭，GPU3回到15MiB、18084关闭。K真实覆盖1–8、low_confidence与max_k，H8/HF0.8固定。

三组512请求OFF/ON测试正在由单一controller串行运行。**本报告不声明端到端加速**；
每轮128请求×128token预热排除。正式ShareGPT预算是每请求16–384 token、合计101003，
不是每请求128；这是与历史Native同一冻结数据/计划，并未改变负载。C16是客户端固定
并发，不是GPU batch16；Target上限4，Draft/Verify实际批次另计。两种sequence floor均1024，
其余未通过收益门槛的实验开关OFF，重型诊断在正式阶段关闭。每轮服务退出后核对释放。
只有全部三组同提交/同client/data/manifest的TPS、E2EP95、TTFTP95和输出差异才决定采用。

完整实际启动命令见gate_cache/launch.sh；可复制、自动释放的独立短功能命令：

```bash
cd /home/yypan/CDB-AV/results/context_fp_cache_20261120
/home/yypan/CDB-AV/.venv/bin/python driver.py --tag "smoke_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --seed 42 --context-cache 1 --diagnostic --requests 16 --max-tokens 64 --gate-warmup 64
```

默认回滚用--context-cache 0，不重置Git、不改生产或他人服务。仅审查命令可加--dry-run。
原日志/响应/采样和失败记录保留服务器，结果在代码目录有软链接。
