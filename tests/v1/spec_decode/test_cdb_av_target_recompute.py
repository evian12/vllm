"""Exercise real scheduler admission, chunk continuation and Draft-wait handoff."""
import pytest
from transformers import OPTConfig

from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.spec_decode.test_cdb_av_draft_wait import queue  # noqa: F401
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize('prompt', [15, 1012, 1188, 2048, 3073])
@pytest.mark.parametrize('outputs', [1, 12])
def test_recompute_stops_before_last_confirmed_token(tmp_path, queue, prompt, outputs):
    OPTConfig(max_position_embeddings=4096).save_pretrained(tmp_path)
    scheduler=create_scheduler(model=str(tmp_path),max_num_seqs=4,
        max_num_batched_tokens=1024,max_model_len=4096,skip_tokenizer_init=True)
    request=create_requests(num_requests=1,num_tokens=prompt,max_tokens=512)[0]
    request.sampling_params.temperature=0
    request.append_output_token_ids([42]*outputs)
    request.status=RequestStatus.PREEMPTED
    request.num_preemptions=1
    scheduler.add_request(request)
    queue.s=scheduler
    queue.begin_target_schedule=lambda budget:queue.refresh_draft_wait()
    scheduler._cdb_p2=queue
    prior_output=list(request.output_token_ids)
    remaining=request.num_tokens-1
    chunks=[]
    while remaining:
        scheduled=scheduler.schedule()
        count=scheduled.num_scheduled_tokens[request.request_id]
        assert count==min(remaining,1024)
        assert not scheduled.scheduled_spec_decode_tokens
        chunks.append(count);remaining-=count
        # Every recompute call remains a non-sampling partial prefill.
        assert request.num_computed_tokens<request.num_tokens
        assert list(request.output_token_ids)==prior_output
    assert request.num_computed_tokens==request.num_tokens-1
    following=scheduler.schedule()
    assert not following.num_scheduled_tokens
    assert request.request_id in queue.draft_wait
    assert request.request_id in queue.frozen
    assert request.num_computed_tokens==request.num_tokens-1
    assert list(request.output_token_ids)==prior_output


@pytest.mark.parametrize('mode', ['initial', 'disabled', 'unsupported', 'candidate'])
def test_recompute_guard_preserves_other_paths(queue,mode):
    from tests.v1.spec_decode.test_cdb_av_draft_wait import add
    add(queue,'a');request=queue.s.requests['a']
    request.num_computed_tokens=0
    if mode=='initial':request.output_token_ids.clear()
    if mode=='disabled':queue.draft_wait_enabled=False
    if mode=='unsupported':request.sampling_params.temperature=1
    if mode=='candidate':request.spec_token_ids=[7]
    assert queue.target_recompute_limit(request,0) is None
