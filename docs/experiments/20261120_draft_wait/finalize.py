"""Archive every prescribed wait-queue comparison, including earlier failure."""
from pathlib import Path
import json
import os
import shutil
import statistics
import subprocess
import time

from audit_refill_forward import audit

B = Path('/home/yypan/CDB-AV')
R = Path(__file__).resolve().parent
W = B/'worktrees/cdb-sharegpt-delivery-20261119'
D = W/'docs/experiments/20261120_draft_wait'


def read(p):
    return json.loads(p.read_text())


def copy(p, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)


def main():
    deadline = time.monotonic()+10800
    while time.monotonic() < deadline:
        status = read(R/'STATUS.json')
        if status['state'] == 'completed':
            break
        if status['state'] == 'failed':
            raise RuntimeError('Controller failed; retain diagnostics and inspect STATUS.json')
        os.kill(status['pid'], 0)
        time.sleep(20)
    else:
        raise TimeoutError('Pair sequence has not completed')
    data = read(R/'ANALYSIS.json')
    assert len(data['runs']) == 6 and len(data['paired']) == 3
    gate = read(R/'FUNCTIONAL_ACCEPTED.json')
    assert gate['unclassified_ar_rows'] == 0
    refill = audit(R/'gate_wait/server.log')
    (R/'REFILL_FORWARD_AUDIT.json').write_text(json.dumps(refill,indent=2)+'\n')
    counts = {}
    for arm in [0,1]:
        group = [x for x in data['runs'] if x['draft_wait']==arm]
        assert len(group)==3
        counts[str(arm)] = {key:sum(x['observed'].get(key,0) for x in group)
                           for key in {key for x in group for key in x['observed']} if not key.endswith('_max_ms')}
    c = counts['1']
    reasons = {k:v for k,v in c.items() if k.startswith('cdb_ar_reason_')}
    assert sum(reasons.values()) == c['cdb_target_ar_rows']
    assert not reasons.get('cdb_ar_reason_unclassified_rows',0)
    for run in ['gate_wait']+[Path(x['directory']).name for x in data['runs']]:
        path = R/run
        release = read(path/'released.json')
        assert release['port_closed'] and not release['our_gpu_pids']
        for name in ['launch.sh','manifest.json','preflight.json','runtime_imports.json','summary.json','released.json']:
            copy(path/name,D/run/name)
    off,on = [data['groups'][str(x)] for x in [0,1]]
    gain = 100*(on['mean_tps']/off['mean_tps']-1)
    tail = 100*(on['mean_run_e2e_p95']/off['mean_run_e2e_p95']-1)
    keep = all(x['tps_change_pct']>0 for x in data['paired']) and gain>=.5 and tail<=1
    decision = dict(source_commit=read(R/'SOURCE.json')['commit'],
        implementation='implemented', cpu_tests='passed', gpu_functional='passed',
        performance_gate_passed=keep, recommended_wait_flag=int(keep), default_flag=0,
        groups=data['groups'], counts=counts, ar_fallback_reasons=reasons,
        mean_tps_gain_pct=gain, mean_run_e2e_p95_change_pct=tail,
        exact_output_requests=[p['exact_output_requests'] for p in data['paired']],
        strict_full_output_equivalence=False, historical_native_tps=341.0732791652744,
        historical_native_comparison_pct=100*(on['mean_tps']/341.0732791652744-1),
        completed_requests=3072, output_tokens=606018, own_services_released=True)
    table='\n'.join(f"| {p['seed']} | {off['tps'][i]:.3f} | {on['tps'][i]:.3f} | {p['tps_change_pct']:+.3f}% | {p['e2e_p95_change_pct']:+.3f}% | {p['exact_output_requests']}/512 |" for i,p in enumerate(data['paired']))
    report=f'''# 显式草稿等待队列调整结果

完成 Target Prefill 和首次生成的可推测请求进入 FIFO Draft 等待队列；每次验证提交正式 token 后重新取得等待资格。等待期间冻结正式前缀，不由普通 AR 推进；空位出现后转交 Draft lease。等待不预分配 Draft KV，也不占用已发布 Ready 的 Target 行名额。取消、抢占和结束清理等待状态；原冻结、KV 所有权和跨窗口 Carry 断言保留。必要的资源、长度、支持范围回退分别计数。开关 CDB_AV_DRAFT_WAIT，默认 0，负载推荐值 {int(keep)}。

源码 {decision['source_commit']}，独立分支 codex/cdb-draft-wait-20261120。这是用户授权的状态机调整，会改变 Target 行构成与生成轨迹，不能标成纯算子等价优化。H8、HF0.8、cap8、原 Ready FIFO 选择函数、BF16/FA2、两种 sequence floor1024 保持固定。编译和 Target replay host 选项关闭。

CPU：21 项等待状态专项；23 项原持久队列、14 项 KV 复用、11 项 KV 所有权回归通过，另有配置和预算回归及版本工具测试。GPU：64 请求预热和16短请求门槛通过，Draft/Target Graph 检查 {gate['draft_graph_checks']}/{gate['target_graph_checks']} 次，跨 Target Carry 字节检查 {gate['cross_target_checks']} 次，唯一验证 {gate['verified']} 次；等待入队/转交 {gate['draft_wait_enqueues']}/{gate['draft_wait_admissions']} 次，未分类 AR 行零。短门槛中的 logits、token 和 KV 校验是同输入证据，完整生成等价仍需区分。

首轮 GPU 门槛在旧 KV 复用 planner 绕过基类出队逻辑时触发状态重叠断言，服务安全退出并释放到15MiB。修复将两个 planner 的候选来源与等待转交共用，新增缓存存在/不存在的回归测试。失败日志和源码146a518保留在前一结果目录，正式配对统一使用修复后的提交。另一次控制脚本语法失败发生在 GPU 服务启动前，日志也保留。

正式比较：同一冻结512 ShareGPT 请求，每轮101003输出token，客户端C16，Target每次最多4行，GPU3。128请求、每个128token的预热排除。seed42 OFF/ON、43 ON/OFF、44 OFF/ON；只改变等待开关。准则在正式结果前登记：三对吞吐均为正、均值提升至少0.5%、逐轮E2E P95均值最多增加1%。

| Seed | OFF tok/s | ON tok/s | 吞吐变化 | E2E P95变化 | 完整输出相同 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

OFF **{off['mean_tps']:.3f}±{off['sd_tps']:.3f}**，ON **{on['mean_tps']:.3f}±{on['sd_tps']:.3f} tok/s**，均值变化{gain:+.3f}%。E2E P95逐轮均值 {off['mean_run_e2e_p95']:.3f}→{on['mean_run_e2e_p95']:.3f}秒；TTFT P95逐轮均值 {off['mean_run_ttft_p95']:.3f}→{on['mean_run_ttft_p95']:.3f}秒。性能准则{'通过' if keep else '未通过'}。完整3072请求成功，606018输出token，各轮独立服务与端口均释放。

ON 三轮 Target调用 {c['target_forward_calls']} 次，验证行 {c['cdb_target_verify_rows']}，AR行 {c['cdb_target_ar_rows']}，Prefill行 {c['cdb_target_prefill_rows']}。每次 Target 验证行数 {c['cdb_target_verify_rows']/c['target_forward_calls']:.3f}，总请求行数 {c['cdb_target_request_rows']/c['target_forward_calls']:.3f}；AR原因计数 {json.dumps(reasons,ensure_ascii=False)}。OFF完整计数与逐轮形状、Ready等待、接受率见 ANALYSIS 和 FINAL_DECISION；这些计数不直接构成耗时归因。

历史原生SD4为341.073±5.497 tok/s，ON均值相对历史为{decision['historical_native_comparison_pct']:+.3f}%。历史数据未重跑。完整输出不同的请求数及首处分歧均保存在ANALYSIS，不能因状态机看似合理就宣称全部输出严格等价。

复测在本结果目录运行 driver.py --tag 新标签 --mode hf --k-cap 8 --seed 42 --draft-wait {int(keep)}；只审查参数加 --dry-run。回退 --draft-wait 0。每轮 launch.sh 保存完整环境与实际命令。原始日志、请求响应、指标与采样保留服务器。
'''
    for name,value in [('FINAL_DECISION.json',json.dumps(decision,indent=2)+'\n'),('FINAL_REPORT.md',report)]:
        (R/name).write_text(value)
    names=['SOURCE.json','PLAN.json','RATIONALE.json','CRITERIA.json','RUNTIME_BINDINGS.json',
        'ACTUAL_MODEL_HEADERS.json','FUNCTIONAL_ACCEPTED.json','ANALYSIS.json','REFILL_FORWARD_AUDIT.json',
        'FINAL_DECISION.json','FINAL_REPORT.md','driver.py','controller.py','analyze.py','manual_observer.py',
        'sitecustomize.py','audit_refill_forward.py','finalize.py','INSTRUMENTATION_SHA256.json']
    for name in names:
        copy(R/name,D/name)
    previous=B/'results/draft_wait_20261120'
    for name in ['STATUS.json','controller_before_syntax_fix.log','gate_wait/released.json','gate_wait/FAILED.json','gate_wait/manifest.json']:
        if (previous/name).exists():
            copy(previous/name,D/'prior_failed_attempt'/name)
    env=dict(os.environ,PATH=str(B/'.venv/bin')+':'+os.environ['PATH'])
    args=['bash','scripts/git_checkpoint.sh','--message','docs: record explicit Draft wait correctness and paired performance']
    for p in sorted(D.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.log'):
            args+=['--path',str(p.relative_to(W))]
    with (R/'docs_checkpoint.log').open('w') as log:
        subprocess.run(args,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','myfork','refs/heads/codex/cdb-sharegpt-delivery-20261119'],cwd=W,text=True).split()[0]
    assert sha==remote
    (R/'DELIVERED.json').write_text(json.dumps(dict(source_commit=decision['source_commit'],docs_commit=sha,remote_verified=True),indent=2)+'\n')


if __name__=='__main__':
    main()
