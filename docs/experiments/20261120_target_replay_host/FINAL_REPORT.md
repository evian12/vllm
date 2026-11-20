# Target Graph replay 主机开销对照

源码 SHA ec0becf70c1fb9060ead21579e786f29a0ba8234。保持默认关闭。保留门槛在首轮前登记，要求三组吞吐均为正、平均提升至少 0.5%、各轮 E2E P95 的均值最多增加 1%。

`CDB_AV_TARGET_REPLAY_HOST=1` 在同次 Forward 中对重复出现的普通 Torch Slot Tensor 复用指针和形状检查。每一层仍检查对象，异对象仍检查实际指针与形状。普通 replay 省去仅供捕获使用的静态映射和模型闭包。Graph 输入、键、Storage 守卫和 KV 所有权校验沿用原路径。

220 项 CPU 测试和 11 项子测试通过。GPU 诊断中，优化版与逐层检查对照 636 次，其中 636 次使用跨层共享 Slot Tensor。真实 Draft/Target Graph 检查 690/411 次，跨 Target Carry 字节检查 208 次，唯一候选交接与验证各 1236 次。将补位记录连接至实际下一次 Draft Forward，确认 973 个有 Ready 请求且与存活行合批的补位事件。

Llama 8B/1B、RTX 4090 GPU3、BF16/FA2、TP1、H8、HF 0.8、cap8、客户端 C16、Target 上限 4。完整上下文指纹缓存、C 类型检查、Draft/Target sequence floor 1024 在两组保持启用。每轮排除 128 请求预热，正式运行冻结的 512 个 ShareGPT 请求并输出 101003 个 token。

| Seed | OFF tok/s | ON tok/s | 吞吐变化 | E2E P95 变化 | 完整输出一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 289.172 | 287.680 | -0.516% | +0.252% | 196/512 |
| 43 | 286.291 | 288.571 | +0.796% | -2.988% | 216/512 |
| 44 | 266.720 | 289.295 | +8.464% | -9.616% | 205/512 |

三轮均值和样本标准差为 **280.728 ± 12.216 → 288.515 ± 0.809 tok/s**（+2.774%）。各轮 E2E P95 的均值为 21.530 → 20.610 秒，TTFT P95 的均值为 0.3465 → 0.3403 秒。六轮 3072/3072 请求成功，均有独立服务退出、端口关闭和 GPU 进程释放记录。

seed 44 的首次 ON 启动在模型服务创建前遇到导入探针超时。两次探针均在服务启动前超时。第三次仅将导入探针时限从 30 秒延长至 120 秒，完整保留失败和驱动差异。原 OFF 样本的 GPU 利用率和功率较前四臂低，且与重试间隔较长。该对照保留在表中，归因受限。此前已有一组负收益，按预登记准则仍保持关闭。

输出差异见 `ANALYSIS.json`，严格生成等价仍未证实。历史原生 SD4 为 341.073 ± 5.497 tok/s。历史原生使用编译与原生 Graph，CDB 使用外层 eager 和手工 Graph。历史数据保持原样。

每轮 `launch.sh` 保存完整命令。复测用新目录名，脚本结束时自动回收独立服务。

```bash
cd /disk14T/yypan/CDB-AV-runtime/results/target_replay_host_20261120
/home/yypan/CDB-AV/.venv/bin/python driver.py --tag "reproduce_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --seed 42 --replay-host 0
```

回退使用 `--replay-host 0`。
