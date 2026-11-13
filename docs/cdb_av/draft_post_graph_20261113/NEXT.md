# 后续候选与边界
## 未验证：只调整 down-projection 物理布局
本轮全矩阵布局探针中，Draft down M4有约25%组件收益、Target down M4/M16有约9%/7%收益；
Draft M1/M16则变慢。不能直接把局部最大值推广为模型TPS，且输出存在BF16逐字节差异。

下一步首先在完全独立clone上测真实小行数范围：Draft M1..4，Target M1..36，
使用真实N/K(2048/6144和4096/12288)、8组交错、足够大权重bank、全部输出误差统计。
只有按实际行数分布加权有清晰收益，且数值风险可接受，才考虑独立开关/独立worktree接入。
不修改模型权重数值、dtype或文件；不复制两份全模型常驻权重、不改变KV或H/K策略。
实际接入还必须验证同值权重、Token/位置/Lease/Commit、Target正确性及公平配对TPS/P95。
本轮未运行该扩展扫描、未接入选择性布局、未将其计为已完成优化。

## 本轮已做与不重复
- 全矩阵column布局30项探针：证据不足，不整体替换。
- Draft head/native confidence Graph：独立功能与固定三组512请求对照，结论见REPORT。
- 既有Target32、seq-floor256、BLAS偏好切换、仅Metadata合包已有结果，不重复盲测。
- H固定8、HF0.8；不整合网页版算法，不开发native async兼容。
- 新GPU服务只在实验期间启动；本轮所有服务结束后释放GPU3/18084。
