# 本轮底层审查
## 已验证与边界
- 实际配置/权重header的 Target intermediate_size=12288、Draft=6144；不沿用旧误设的14336/7168。shape清单和config SHA存档。
- 30个同值BF16物理矩阵布局比较，仅7个更快、23个更慢，14个输出逐字节相同。不能全局替换布局；down-projection个别收益仍只是组件证据，未接入。
- 独立Draft head/native HF Graph组件每步约节省24–28us；同样保留max/logsumexp/exp/float64打包和D2H，不改变K策略或阈值。CPU提交减少不等于权重读取/矩阵乘执行消失。
- 前轮8个真实Target replay区间中位数：backbone16.3655ms、LM-head1.3890ms；六buffer复制0.0702ms。这是前轮有界重型诊断，不是本轮正式性能数据，不能简单相加为端到端临界路径。
- WEIGHT_SCALE.json只给BF16矩阵字节尺度，不能据此宣称DRAM饱和。当前不启用量化、不改权重数值/文件或Tensor dtype。
- 本轮32请求功能审计：主Draft平均有效行3.60；Target平均调度3.87请求、其中验证候选1.95。混合Prefill/AR/Verify在运行，不能用服务端容量4代替实际Verify batch。
- KV上下文准备累计已复用前缀114471 token、缺失准备3889 token；96.7%是上下文准备token口径，不是显存占用率或独立性能收益。
- 976个草稿工作项在已有其他成员Ready后加入同窗口后续实际Forward；这不是976个唯一API请求。总计32个API请求。
- 主路径753个Draft forward事件（不含诊断额外replay），47个无GPU空闲tick；不会把idle tick算为实际Forward。
- HF actual K均值2.389，真实接受候选均值1.666/verify；没有更改动态K算法、H或请求选择。单次Draft较快仍会被多次调用与低接受收益抵消。
## 结论规则
固定三组512请求配对结果决定是否推荐新head/confidence Graph。保留默认0及安全回退。不开启native async，论文所有AR/nativeSD/CDB显式--no-async-scheduling。
不得以组件倍数、Graph命中率或单组TPS宣称系统加速；严格全输出等价和全局最优仍未证明。
