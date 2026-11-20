"""Archive the compilation diagnostic and all six paired runs."""
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
DEST = DOCS / 'docs/experiments/20261120_llama_compile_backbone'


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
    deadline = time.monotonic() + 10800
    while time.monotonic() < deadline:
        state = read(RESULT / 'STATUS.json')
        if state['state'] == 'completed':
            return
        if state['state'] == 'failed':
            raise RuntimeError('Compilation controller failed. Inspect archived STATUS.json and gate_compiled_warmup1024/server.log')
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
    refill = audit(RESULT / 'gate_compiled_warmup1024/server.log')
    assert refill['with_ready_and_surviving_active_rows'] > 0
    (RESULT / 'REFILL_FORWARD_AUDIT.json').write_text(json.dumps(refill, indent=2) + '\n')
    archive_run('gate_compiled_warmup1024')
    for run in data['runs']:
        archive_run(Path(run['directory']).name)
    for name in ['PLAN.json', 'SOURCE.json', 'RATIONALE.json', 'CRITERIA.json',
                 'FUNCTIONAL_ACCEPTED.json', 'INSTRUMENTATION_SHA256.json', 'COMPILER_WARMUP_RECOVERY.json', 'compile_audit.py', 'sitecustomize.py', 'test_compile_audit.py', 'driver.py',
                 'analyze.py', 'controller.py', 'pipeline.py', 'finalize.py',
                 'ACTUAL_MODEL_HEADERS.json', 'audit_refill_forward.py',
                 'REFILL_FORWARD_AUDIT.json', 'ANALYSIS.json']:
        if not name.endswith('.log') and (RESULT / name).exists():
            copy(RESULT / name, DEST / name)
        elif not (RESULT / name).exists():
            print('Missing historical archive input, not reconstructed:', name, flush=True)
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
    report = f'''# Llama Backbone 编译对照

源码 SHA {sha}。归档清单中的历史 RATIONALE.json 不存在；保留首次归档失败日志，没有重建该历史文件。{'保留为此负载的可选优化。' if retain else '保持默认关闭。'}保留门槛在首轮前登记，要求三组吞吐均为正、平均提升至少 0.5%、各轮 E2E P95 的均值最多增加 1%。

显式比较编译 mode=0/3，两个分支均使用 vLLM native Graph NONE 和 CDB 自己的 CUDA Graph。
编译配置只变更 mode，custom_ops 保持 all。模型权重、BF16、采样和 Scheduler 配置固定。首次门槛在 1028 的默认预热尺寸触发预算断言，两臂均以显式 compile_sizes=[1024] 复跑并保留记录。

2 项 CPU 测试验证可选诊断钩子。实测启动日志确认 Target 和 Draft 均编译。
GPU 功能门槛在 CDB 自带 Graph 校验里以 skip_compiled=True 运行原始 eager 参考。
通过 {gate['draft_graph_checks']}/{gate['target_graph_checks']} 次 Draft/Target Graph 检查，
{gate['cross_target_checks']} 次跨 Target Carry 字节检查和 {gate['verified']} 次唯一候选交接与验证。
确认 {refill['with_ready_and_surviving_active_rows']} 个有 Ready 请求且与存活行合批的补位事件。
通过是对实际校验输入的证据。正式请求集的完整 Token 对比附在分析中。


Llama 8B/1B、RTX 4090 GPU3、BF16/FA2、TP1、H8、HF 0.8、cap8、客户端 C16、Target 上限 4。完整上下文指纹缓存、C 类型检查、Draft/Target sequence floor 1024 在两组保持启用。每轮排除 128 请求预热，正式运行冻结的 512 个 ShareGPT 请求并输出 101003 个 token。

| Seed | OFF tok/s | ON tok/s | 吞吐变化 | E2E P95 变化 | 完整输出一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

三轮均值和样本标准差为 **{off['mean_tps']:.3f} ± {off['sd_tps']:.3f} → {on['mean_tps']:.3f} ± {on['sd_tps']:.3f} tok/s**（{gain:+.3f}%）。各轮 E2E P95 的均值为 {off['mean_run_e2e_p95']:.3f} → {on['mean_run_e2e_p95']:.3f} 秒，TTFT P95 的均值为 {off['mean_run_ttft_p95']:.4f} → {on['mean_run_ttft_p95']:.4f} 秒。六轮 3072/3072 请求成功，均有独立服务退出、端口关闭和 GPU 进程释放记录。

输出差异见 `ANALYSIS.json`，严格生成等价仍未证实。历史原生 SD4 为 341.073 ± 5.497 tok/s。历史原生使用编译与原生 Graph，CDB 使用外层 eager 和手工 Graph。历史数据保持原样。

每轮 `launch.sh` 保存完整命令。复测用新目录名，脚本结束时自动回收独立服务。

```bash
cd {RESULT}
/home/yypan/CDB-AV/.venv/bin/python driver.py --tag "reproduce_$(date +%Y%m%d_%H%M%S)" --mode hf --k-cap 8 --seed 42 --compile-backbone {int(retain)}
```

回退使用 `--compile-backbone 0`。
'''
    for folder in [RESULT, DEST]:
        (folder / 'FINAL_DECISION.json').write_text(json.dumps(result, indent=2) + '\n')
        (folder / 'FINAL_REPORT.md').write_text(report)
    env = dict(os.environ, PATH=str(ROOT / '.venv/bin') + ':' + os.environ['PATH'])
    paths = [path.relative_to(DOCS) for path in sorted(DEST.rglob('*')) if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc']
    args = ['bash', 'scripts/git_checkpoint.sh', '--message',
            'docs: archive Llama backbone compilation comparison']
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
