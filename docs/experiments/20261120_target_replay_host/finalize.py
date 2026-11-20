"""Archive the replay-host diagnostic and all six paired runs."""
from pathlib import Path
import json
import os
import shutil
import statistics
import subprocess
import time

from audit_refill_forward import audit


ROOT = Path('/home/yypan/CDB-AV')
RESULT = Path(__file__).resolve().parent
DOCS = ROOT / 'worktrees/cdb-sharegpt-delivery-20261119'
DEST = DOCS / 'docs/experiments/20261120_target_replay_host'


def read(path):
    return json.loads(path.read_text())


def copy(path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def archive_run(run):
    path = RESULT / run
    release = read(path / 'released.json')
    assert release['port_closed'] and not release['our_gpu_pids']
    for name in ['launch.sh', 'manifest.json', 'runtime_imports.json',
                 'preflight.json', 'summary.json', 'released.json']:
        copy(path / name, DEST / run / name)


def wait_for_controller():
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        state = read(RESULT / 'STATUS.json')
        if state['state'] == 'completed':
            return
        if state['state'] == 'failed':
            raise RuntimeError('Controller failed. See STATUS.json and controller.log')
        os.kill(state['pid'], 0)
        time.sleep(20)
    raise TimeoutError('Controller did not finish within two hours')


def main():
    wait_for_controller()
    data = read(RESULT / 'ANALYSIS.json')
    assert len(data['runs']) == 6 and len(data['paired']) == 3
    off, on = (data['groups'][str(key)] for key in [0, 1])
    gain = 100 * (on['mean_tps'] / off['mean_tps'] - 1)
    tail = 100 * (on['mean_run_e2e_p95'] / off['mean_run_e2e_p95'] - 1)
    criteria = read(RESULT / 'CRITERIA.json')
    retain = (all(p['tps_change_pct'] > 0 for p in data['paired'])
              and gain >= criteria['minimum_mean_tps_gain_pct']
              and tail <= criteria['maximum_mean_e2e_p95_increase_pct'])
    gate = read(RESULT / 'FUNCTIONAL_ACCEPTED.json')
    refill = audit(RESULT / 'gate_replay_host/server.log')
    assert refill['with_ready_and_surviving_active_rows'] > 0
    (RESULT / 'REFILL_FORWARD_AUDIT.json').write_text(json.dumps(refill, indent=2) + '\n')
    archive_run('gate_replay_host')
    for run in data['runs']:
        archive_run(Path(run['directory']).name)
    for name in ['PLAN.json', 'SOURCE.json', 'RATIONALE.json', 'CRITERIA.json',
                 'CPU_TESTS.log', 'FUNCTIONAL_ACCEPTED.json', 'driver.py',
                 'analyze.py', 'controller.py', 'pipeline.py', 'finalize.py',
                 'ACTUAL_MODEL_HEADERS.json', 'audit_refill_forward.py',
                 'REFILL_FORWARD_AUDIT.json', 'ANALYSIS.json', 'RECOVERY.json', 'resume_controller.py', 'driver_timeout120.py', 'resume_controller_timeout120.py']:
        if not name.endswith('.log'):
            copy(RESULT / name, DEST / name)
    sha = read(RESULT / 'SOURCE.json')['commit']
    tps_pairs = [p['tps_change_pct'] for p in data['paired']]
    result = dict(source_commit=sha, recommended_value=int(retain),
                  groups=data['groups'], mean_tps_gain_pct=gain,
                  mean_paired_tps_gain_pct=statistics.mean(tps_pairs),
                  sample_sd_paired_tps_gain_pct=statistics.stdev(tps_pairs),
                  mean_run_e2e_p95_change_pct=tail, strict_output_equivalence=False,
                  exact_output_requests=[p['exact_output_requests'] for p in data['paired']],
                  requests_completed=3072, output_tokens=606018, ports_released=True)
    table = '\n'.join(
        f"| {p['seed']} | {off['tps'][i]:.3f} | {on['tps'][i]:.3f} | "
        f"{p['tps_change_pct']:+.3f}% | {p['e2e_p95_change_pct']:+.3f}% | "
        f"{p['exact_output_requests']}/512 |"
        for i, p in enumerate(data['paired']))
    report = f'''# Target Graph replay 主机开销对照

源码 SHA {sha}。{'保留为此负载的可选优化。' if retain else '保持默认关闭。'}保留门槛在首轮前登记，要求三组吞吐均为正、平均提升至少 0.5%、各轮 E2E P95 的均值最多增加 1%。

`CDB_AV_TARGET_REPLAY_HOST=1` 在同次 Forward 中对重复出现的普通 Torch Slot Tensor 复用指针和形状检查。每一层仍检查对象，异对象仍检查实际指针与形状。普通 replay 省去仅供捕获使用的静态映射和模型闭包。Graph 输入、键、Storage 守卫和 KV 所有权校验沿用原路径。

220 项 CPU 测试和 11 项子测试通过。GPU 诊断中，优化版与逐层检查对照 {gate['slot_guard_compared']} 次，其中 {gate['slot_guard_shared_tensor']} 次使用跨层共享 Slot Tensor。真实 Draft/Target Graph 检查 {gate['draft_graph_checks']}/{gate['target_graph_checks']} 次，跨 Target Carry 字节检查 {gate['cross_target_checks']} 次，唯一候选交接与验证各 {gate['verified']} 次。将补位记录连接至实际下一次 Draft Forward，确认 {refill['with_ready_and_surviving_active_rows']} 个有 Ready 请求且与存活行合批的补位事件。

Llama 8B/1B、RTX 4090 GPU3、BF16/FA2、TP1、H8、HF 0.8、cap8、客户端 C16、Target 上限 4。完整上下文指纹缓存、C 类型检查、Draft/Target sequence floor 1024 在两组保持启用。每轮排除 128 请求预热，正式运行冻结的 512 个 ShareGPT 请求并输出 101003 个 token。

| Seed | OFF tok/s | ON tok/s | 吞吐变化 | E2E P95 变化 | 完整输出一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

三轮均值和样本标准差为 **{off['mean_tps']:.3f} ± {off['sd_tps']:.3f} → {on['mean_tps']:.3f} ± {on['sd_tps']:.3f} tok/s**（{gain:+.3f}%）。各轮 E2E P95 的均值为 {off['mean_run_e2e_p95']:.3f} → {on['mean_run_e2e_p95']:.3f} 秒，TTFT P95 的均值为 {off['mean_run_ttft_p95']:.4f} → {on['mean_run_ttft_p95']:.4f} 秒。六轮 3072/3072 请求成功，均有独立服务退出、端口关闭和 GPU 进程释放记录。

seed 44 的首次 ON 启动在模型服务创建前遇到导入探针超时。两次探针均在服务启动前超时。第三次仅将导入探针时限从 30 秒延长至 120 秒，完整保留失败和驱动差异。原 OFF 样本的 GPU 利用率和功率较前四臂低，且与重试间隔较长。该对照保留在表中，归因受限。此前已有一组负收益，按预登记准则仍保持关闭。

输出差异见 `ANALYSIS.json`，严格生成等价仍未证实。历史原生 SD4 为 341.073 ± 5.497 tok/s。历史原生使用编译与原生 Graph，CDB 使用外层 eager 和手工 Graph。历史数据保持原样。

每轮 `launch.sh` 保存完整命令。复测用新目录名，脚本结束时自动回收独立服务。

```bash
cd {RESULT}
/home/yypan/CDB-AV/.venv/bin/python driver.py --tag "reproduce_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --seed 42 --replay-host {int(retain)}
```

回退使用 `--replay-host 0`。
'''
    for folder in [RESULT, DEST]:
        (folder / 'FINAL_DECISION.json').write_text(json.dumps(result, indent=2) + '\n')
        (folder / 'FINAL_REPORT.md').write_text(report)
    env = dict(os.environ, PATH=str(ROOT / '.venv/bin') + ':' + os.environ['PATH'])
    paths = [path.relative_to(DOCS) for path in sorted(DEST.rglob('*')) if path.is_file()]
    args = ['bash', 'scripts/git_checkpoint.sh', '--message',
            'docs: archive Target Graph host replay comparison']
    for path in paths:
        args += ['--path', str(path)]
    with (RESULT / 'docs_final_checkpoint.log').open('w') as log:
        subprocess.run(args, cwd=DOCS, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=180)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=DOCS, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=DOCS, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'myfork', f'refs/heads/{branch}'],
                                     cwd=DOCS, text=True, timeout=30).split()[0]
    assert head == remote
    (RESULT / 'DELIVERED.json').write_text(json.dumps(dict(source_commit=sha, docs_commit=head,
                                                          remote_verified=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
