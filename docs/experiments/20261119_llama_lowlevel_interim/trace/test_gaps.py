from analyze_gaps import summarize


def e(name,start,duration,category='kernel',corr=None):
    return dict(ph='X',name=name,ts=start,dur=duration,cat=category,pid=1,tid=2,
                args={} if corr is None else {'correlation':corr})


def test_graph_gap_and_cpu_gap_are_distinct():
    rows=[e('CDB::target_forward::t8',0,8,'user_annotation'),
          e('cudaGraphLaunch',1,1,'cuda_runtime',1),
          e('a',10,5,corr=1),e('b',20,5,corr=1),
          e('CDB::draft_forward::r4',30,15,'user_annotation'),
          e('cudaLaunchKernel',32,1,'cuda_runtime',2),
          e('c',40,5,corr=2)]
    result=summarize(rows)
    assert result['total_uncovered_us']==20
    assert result['groups']['same_graph_launch']['sum_us']==5
    assert result['transitions']['target_forward -> draft_forward']['sum_us']==15


def test_overlap_uses_latest_end_not_latest_start():
    result=summarize([e('a',0,100),e('b',10,5),e('c',110,10)])
    assert result['total_uncovered_us']==10


def test_no_gpu_data_is_not_zero():
    assert summarize([])['total_uncovered_us'] is None
    assert summarize([e('a',0,5)])['total_uncovered_us']==0


def test_missing_correlation_is_not_accidentally_matched():
    result=summarize([e('cudaGraphLaunch',0,5,'cuda_runtime'),e('a',10,5),e('b',20,5)])
    assert 'same_graph_launch' not in result['groups']
