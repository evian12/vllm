# 本轮剩余瓶颈证据
独立16请求/64输出Carry诊断，Target timing8个实际warmed replay样本，heavy validation ON。
不是正式TPS，也不是CUPTI kernel-active总和；仅分析该配置发生的样本。
中位数：
- Target模型backbone Graph current-stream span16.3655ms，CPU enqueue0.0164ms。
- 相同live eager模型span16.9787ms，CPU enqueue14.6193ms。
- Target LM-head span1.3890ms，CPU enqueue0.1015ms。
- 每次36层KVstorage检查CPU0.1174ms。
- 六buffer复制CPU enqueue0.0557ms，stream span0.0702ms。

解释：
1. Eager CPU提交14.6ms与GPU执行重叠，不能把它全当作Graph可节省的端到端时间。
   Graph大幅减少host提交，当前模型span仍约16ms；支持模型GPU计算/权重读取成本重要。
2. storage检查和六copy有优化候选，但量级小于模型span；不删所有权断言去争取微小数字。
3. 完整输入staging组件约2.2x，五组端到端约+.80%，说明不能用组件倍数推断TPS。
4. 既有Nsight GEMM/GEMV92.8%是kernel累计份额，不是端到端占比或带宽饱和证明。
   后续优先量化线性层/LM-head内存布局、实际kernel选择和critical-path，而非继续增加日志。
5. Graph缓存策略/更大query域/Target32/floor256均有既有记录；不盲目重跑已否定配置。
   模型/并行组合、长上下文变化可能重新打开候选，当前不宣称全局最优。
6. BF16矩阵物理布局独立clone探针仅准备未运行；没有改文件/模型参数或宣称其收益。
7. 用户论文协议：所有AR/nativeSD/CDB对比显式--no-async-scheduling；不在本轮开发async兼容。
