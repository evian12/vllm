# Llama Backbone 编译对照

源码 SHA ec0becf70c1fb9060ead21579e786f29a0ba8234。归档清单中的历史 RATIONALE.json 不存在；保留首次归档失败日志，没有重建该历史文件。保持默认关闭。保留门槛在首轮前登记，要求三组吞吐均为正、平均提升至少 0.5%、各轮 E2E P95 的均值最多增加 1%。

显式比较编译 mode=0/3，两个分支均使用 vLLM native Graph NONE 和 CDB 自己的 CUDA Graph。
编译配置只变更 mode，custom_ops 保持 all。模型权重、BF16、采样和 Scheduler 配置固定。首次门槛在 1028 的默认预热尺寸触发预算断言，两臂均以显式 compile_sizes=[1024] 复跑并保留记录。

2 项 CPU 测试验证可选诊断钩子。实测启动日志确认 Target 和 Draft 均编译。
GPU 功能门槛在 CDB 自带 Graph 校验里以 skip_compiled=True 运行原始 eager 参考。
通过 706/411 次 Draft/Target Graph 检查，
216 次跨 Target Carry 字节检查和 1202 次唯一候选交接与验证。
确认 967 个有 Ready 请求且与存活行合批的补位事件。
通过是对实际校验输入的证据。正式请求集的完整 Token 对比附在分析中。


Llama 8B/1B、RTX 4090 GPU3、BF16/FA2、TP1、H8、HF 0.8、cap8、客户端 C16、Target 上限 4。完整上下文指纹缓存、C 类型检查、Draft/Target sequence floor 1024 在两组保持启用。每轮排除 128 请求预热，正式运行冻结的 512 个 ShareGPT 请求并输出 101003 个 token。

| Seed | OFF tok/s | ON tok/s | 吞吐变化 | E2E P95 变化 | 完整输出一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 289.058 | 289.437 | +0.131% | -1.191% | 195/512 |
| 43 | 287.366 | 289.720 | +0.819% | -2.551% | 197/512 |
| 44 | 290.113 | 290.418 | +0.105% | +0.380% | 204/512 |

三轮均值和样本标准差为 **288.845 ± 1.386 → 289.859 ± 0.505 tok/s**（+0.351%）。各轮 E2E P95 的均值为 20.905 → 20.668 秒，TTFT P95 的均值为 0.3530 → 0.3500 秒。六轮 3072/3072 请求成功，均有独立服务退出、端口关闭和 GPU 进程释放记录。

输出差异见 `ANALYSIS.json`，严格生成等价仍未证实。历史原生 SD4 为 341.073 ± 5.497 tok/s。历史原生使用编译与原生 Graph，CDB 使用外层 eager 和手工 Graph。历史数据保持原样。

每轮 `launch.sh` 保存完整命令。复测用新目录名，脚本结束时自动回收独立服务。

```bash
cd /disk14T/yypan/CDB-AV-runtime/results/llama_compile_backbone_20261120
/home/yypan/CDB-AV/.venv/bin/python driver.py --tag "reproduce_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --seed 42 --compile-backbone 0
```

回退使用 `--compile-backbone 0`。
