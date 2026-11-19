# 后续 Draft Graph 范围审查（非性能结论）

从真实80请求诊断日志提取675个mixed_batch：q1=130、q2–8=477、q9–16=10、q17+=58。
这里query长度是正式上下文准备/增量补算长度，不是推测长度K；K仍受cap8约束。
DRAFT_SHAPE_AUDIT保留窗口/步骤/实际query_lengths示例。

仅扩展query bound到16最多覆盖这10个混合批次，约占本短样本mixed_batch的1.48%，
不能推广为完整512负载的比例或速度收益；此前未给该范围实现新Graph或做GPU验收。
该候选优先级低于已观测的Target缓存碎片。若将来扩域，须保留现有q<=8路径/键域、
查实FA2 backend与静态/动态CPU shadow约束、真实无padding行与KV索引，先CPU和
GPU逐层/Token检查，再同配置多轮性能比较。不得把context query bound16说成K16。

CPU重复指纹亦列为后续候选，研究原型19项CPU语义测试通过但未接入，也无GPU或
吞吐收益证明；见20261120_context_fingerprint_prototype。不得按requestID/长度
跳过完整前缀验证或移除frozen-context/computed-length/KV ownership断言。
