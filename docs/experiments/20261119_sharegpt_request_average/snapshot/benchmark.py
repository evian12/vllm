"""Paired streaming client: independent open-loop submission, no hidden semaphore."""
import asyncio,time,json,pathlib,random,math,hashlib,os
import aiohttp

def pct(values,q):
 if not values:return None
 v=sorted(values); pos=(len(v)-1)*q;lo=int(pos);return v[lo]+(v[min(lo+1,len(v)-1)]-v[lo])*(pos-lo)

async def run(base,rows,family,kind,level,n,seed,dest,smoke=False,slo=None,arrival_window=60,open_requests=None):
 dest=pathlib.Path(dest); dest.mkdir(exist_ok=False)
 rng=random.Random(seed);indices=list(range(len(rows)));rng.shuffle(indices)
 if kind=='closed':plan=[dict(index=i,source=indices[i%len(indices)],at=None) for i in range(n)]
 else:
  plan=[]; at=0.
  while True:
   at+=rng.expovariate(level)
   if open_requests is None and at>=arrival_window:break
   i=len(plan);plan.append(dict(index=i,source=indices[i%len(indices)],at=at))
   if open_requests is not None and len(plan)>=open_requests:break
 (dest/'plan.json').write_text(json.dumps(plan))
 results=[];wall_start=time.time();start=time.perf_counter();cursor=0
 timeout=aiohttp.ClientTimeout(total=None,sock_connect=30)
 async with aiohttp.ClientSession(timeout=timeout,connector=aiohttp.TCPConnector(limit=0)) as session:
  async def req(p):
   source=rows[p['source']];dispatch=time.perf_counter(); first=None;events=[]; usage=None;error=None;texts=[];done=False;finish=None
   cap=min(source['max_tokens'],32) if smoke else source['max_tokens']
   body=dict(model='target',prompt=source['prompt_ids'][family],max_tokens=cap,temperature=0,seed=seed,ignore_eos=True,stream=True,stream_options={'include_usage':True},return_token_ids=True,top_p=1,top_k=-1,request_id='baseline-'+hashlib.sha256(str(dest).encode()).hexdigest()[:12]+'-'+str(p['index']))
   trace_key=dest.name+':'+str(source['id'])
   if os.getenv('CDB_TRACE_MISSING_FIRST')=='1' and p['index']==0:
    trace_key='intentional_missing:'+trace_key
   body['vllm_xargs']={'cdb_trace_key':trace_key}
   with (dest/'identities.jsonl').open('a') as identity_log:
    identity_log.write(json.dumps(dict(trace_key=trace_key,prompt_token_ids=body['prompt'],request_id=body['request_id'],source_id=source['id']))+'\n')
   with (dest/'http_requests.jsonl').open('a') as request_log:
    request_log.write(json.dumps(body)+'\n')
   token_events=[];response_id=None
   try:
    async with session.post(base+'/v1/completions',json=body) as resp:
     if resp.status!=200:raise RuntimeError(f'HTTP {resp.status}: '+(await resp.text())[:500])
     async for line in resp.content:
      line=line.strip()
      if not line.startswith(b'data:'):continue
      value=line[5:].strip()
      if value==b'[DONE]':done=True;break
      obj=json.loads(value); now=time.perf_counter();response_id=obj.get('id',response_id)
      if obj.get('usage'):usage=obj['usage']
      for ch in obj.get('choices',[]):
       if ch.get('finish_reason'):finish=ch['finish_reason']
       ts=ch.get('token_ids') or []
       if ts:
        if first is None:first=now
        events.append(now-dispatch);token_events.append(dict(t=now-dispatch,n=len(ts),tokens=ts))
       texts.append(ch.get('text',''))
     if not done or usage is None or usage['completion_tokens']!=cap or first is None:raise RuntimeError('Incomplete stream, missing usage, or token budget mismatch')
   except Exception as e:error=str(e)
   end=time.perf_counter();nt=usage['completion_tokens'] if usage else 0
   # Multi-token SD chunks have no observable within-chunk token timestamps.
   # Preserve chunks, report per-token ITL only if every event is one token.
   gaps=[events[i]-events[i-1] for i in range(1,len(events))]
   exact=all(e['n']==1 for e in token_events) and sum(e['n'] for e in token_events)==nt
   delivery_times=[e['t'] for e in token_events for _ in range(e['n'])]
   delivery_itl=[delivery_times[i]-delivery_times[i-1] for i in range(1,len(delivery_times))]
   r=dict(response_id=response_id,delivery_itl=delivery_itl,delivery_itl_p95=pct(delivery_itl,.95),completed_at=end-start,index=p['index'],source_id=source['id'],ok=error is None,error=error,scheduled=p['at'],dispatch=dispatch-start,dispatch_lag=None if p['at'] is None else dispatch-start-p['at'],scheduled_e2e=None if p['at'] is None else end-start-p['at'],scheduled_ttft=None if p['at'] is None or first is None else first-start-p['at'],e2e=end-dispatch,ttft=None if first is None else first-dispatch,output_tokens=nt,requested_tokens=cap,finish_reason=finish,tpot=None if nt<2 or first is None else (events[-1]-events[0])/(nt-1),itl_exact=exact,itl=gaps if exact else None,chunk_gaps=gaps,token_events=token_events,text=''.join(texts))
   results.append(r)
   with (dest/'requests.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  async def worker():
   nonlocal cursor
   while cursor<len(plan):
    p=plan[cursor];cursor+=1;await req(p)
  if kind=='closed':await asyncio.gather(*(worker() for _ in range(level)))
  else:
   pending=[]
   for p in plan:
    await asyncio.sleep(max(0,start+p['at']-time.perf_counter()));pending.append(asyncio.create_task(req(p)))
   await asyncio.gather(*pending)
 finished=time.perf_counter()-start
 first_dispatch=min((r['dispatch'] for r in results),default=0.)
 last_completion=max((r['completed_at'] for r in results),default=finished)
 elapsed=last_completion-first_dispatch; good=[r for r in results if r['ok']]
 summary=dict(requests=len(plan),completed=len(good),failed=len(results)-len(good),elapsed_s=elapsed,measurement_start_wall=wall_start,first_dispatch_wall=wall_start+first_dispatch,last_completion_wall=wall_start+last_completion,first_dispatch_s=first_dispatch,last_completion_s=last_completion,drain_s=max(0.,last_completion-(arrival_window if kind=='open' and open_requests is None else (plan[-1]['at'] if kind=='open' and plan else 0.))) if kind=='open' else None,output_tokens=sum(r['output_tokens'] for r in good),throughput_tok_s=sum(r['output_tokens'] for r in good)/elapsed,open_requests=open_requests if kind=='open' else None,arrival_window_s=(plan[-1]['at'] if open_requests is not None and plan else arrival_window) if kind=='open' else None,open_workload='finite_request_poisson' if kind=='open' and open_requests is not None else ('fixed_window_poisson' if kind=='open' else None),latency_s={k:{'p50':pct([r[k] for r in good if r[k] is not None],.5),'p95':pct([r[k] for r in good if r[k] is not None],.95)} for k in ['ttft','e2e','tpot','scheduled_ttft','scheduled_e2e']},itl_s={f'p{q}':pct([v for r in good for v in (r['itl'] or [])],q/100) for q in [50,95]},dispatch_lag_s={f'p{q}':pct([r['dispatch_lag'] for r in results if r['dispatch_lag'] is not None],q/100) for q in [50,95]},arrival_timing_qualified=kind=='closed' or (pct([r['dispatch_lag'] for r in results],.95) is not None and pct([r['dispatch_lag'] for r in results],.95)<=.01),delivery_itl_s={f'p{q}':pct([v for r in good for v in r['delivery_itl']],q/100) for q in [50,95]},itl_definition='Client delivery: all token IDs in one SSE chunk share its receive timestamp; zero intervals within a burst are retained. GPU-internal generation time is not inferred.',itl_coverage_requests=sum(r['itl_exact'] for r in good),slo_compliance=None,slo_note='No predefined thresholds; multi-token stream chunks do not expose exact token ITL.',plan_sha256=hashlib.sha256((dest/'plan.json').read_bytes()).hexdigest(),smoke=smoke)
 (dest/'summary.json').write_text(json.dumps(summary,indent=2));return summary
