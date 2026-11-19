import json
import pytest
from repair_labels import repair


def test_bad_export_only_our_known_names_are_escaped():
    text='{"traceEvents":[{"name": "CDB::target_dispatch::replay::[8,4,9,512,64,0,"bounded"]", "pid":1,"dur":12.5},{"name": "CDB::draft_graph_forward::[8,1024,4,"dynamic8"]", "pid":1,"dur":2.5}]}'
    corrected,stats=repair(text)
    assert stats['changed_names']==2 and stats['events']==2
    data=json.loads(corrected)
    assert [e['dur'] for e in data['traceEvents']]==[12.5,2.5]
    assert json.loads(data['traceEvents'][0]['name'].split('::',3)[3])==[8,4,9,512,64,0,'bounded']


def test_already_valid_simple_marker_unchanged():
    text='{"traceEvents":[{"name": "CDB::target_dispatch::fallback_layout::null", "pid":1,"ts":4,"dur":1}]}'
    result,stats=repair(text)
    assert result==text and stats['changed_names']==0


def test_unrelated_bad_json_not_silently_repaired():
    with pytest.raises(json.JSONDecodeError):
        repair('{"traceEvents":[{"name":"other"bad","pid":1}]}')
