# 固定 K8 参考与请求平均 K 配对实验工具快照

本目录保留 2026-11-19 GPU3 实验的实际外部客户端、采集、观测、审计与恢复控制器，不是任意环境下一键部署的通用工具。模型执行源码为 SOURCE.json 的 8c94d2ac；其父提交673c0c为底层优化基线。不能把本交付分支的 docs 提交冒充运行源码。

## 数据协议与定义

HTTP `/v1/completions` 顶层 `vllm_xargs.cdb_trace_key` 携带 `measurement:<source_id>`；请求ID与prompt token IDs另存identities.jsonl。均值用Worker真实`p2_verified.accepted`，逐请求累计除以验证次数，ceil后clamp到1–8。它是Target verifier接受深度，原生Scheduler会在后续执行最终输出长度裁剪，因此末轮接受数不一定全部交付API；不含bonus/correction。

正式参考512/C16固定K8、关闭HF停止，437请求有验证；缺失75条的Target事件逐一证明仅Prefill+AR。随后仅为这75条以原Prompt/原输出预算、固定K8、C1补采，75/75有观测。合并时只补缺，不覆盖437条原均值，检查Prompt、trace key、重复/缺失记录；暖机别名不增添样本。

完整参考表512请求，10,851验证、34,210接受Draft tokens，SHA见AVERAGE_COMPLETE_FILE.json。没有把缺失当0；真实平均0则按最小合法K=1。两种采集的batch shape不同且使用同一测试Prompt，属于离线事后消融，不是在线泛化、逐轮oracle或系统性能上界。K8处接受长度被cap截断。

## 测试和安全

SOURCE.json 对应245项CPU回归通过，32/32独立GPU门槛、184次真实跨Target Carry KV字节检查。工具测试15项覆盖HTTP xargs、只取正式观测、混入重复/错误Prompt/缺失观测拒绝、真实0与未知区分、租约事件闭环及缺失reset不虚构为0。

补采后没有打印p2_reset，原控制器错误地要求该行而停止。这次脚本失败与采集成功都保留。恢复脚本未修改核心断言，另行验证1,631组Ready→handoff→verified一一对应、每条请求最终块表与释放匹配、75条释放、未关闭候选/所有者均0、最终running/waiting/KV usage均0、服务已退出/端口关闭/显存回到15MiB。reset字段保持null，不声称观测到reset ACK。原功能门槛对reset的要求不变。

所有性能臂正式512/C16/101,003 output tokens，128条/C16/128输出warmup，H8、cap8、seed42/43/44；HF与平均K使用相同观察器。服务参数/底层开关固定；只改K策略及其参考文件。采集TPS不参与性能比较。原生不重跑。

## 使用

原始运行目录 `/home/yypan/CDB-AV/results/request_avg_sharegpt_gpu3_20261119`，代码目录软连接 `cdb_av_request_avg_sharegpt_results`。原始模型、冻结请求集、响应、日志及扩展二进制均不提交Git。其sha、实际导入路径、启动命令在各run的manifest.json/runtime_imports.json/launch.sh。

快照脚本使用固定的原始运行路径、源码worktree和数据路径；不得在当前已有结果目录中盲目重新运行控制器。它们是精确追溯资料，重新实验需创建独立目录并显式更新与核验配置，不能覆盖历史结果。

CPU工具测试（不加载模型或占用GPU）：

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-sharegpt-delivery-20261119
TASK_SNAPSHOT="$PWD/docs/experiments/20261119_sharegpt_request_average/snapshot"
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$TASK_SNAPSHOT:/home/yypan/CDB-AV/worktrees/cdb-request-avg-sharegpt-20261119" /home/yypan/CDB-AV/.venv/bin/python -m pytest -q "$TASK_SNAPSHOT"
```

结果报告由当前配对测试完成后补齐；不能以本工具提交宣称性能收益。
