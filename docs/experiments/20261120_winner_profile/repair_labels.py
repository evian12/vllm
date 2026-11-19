"""Escape only our known raw Kineto annotation names; never edit raw traces.

The decoder verifies the suffix was valid JSON before re-escaping the enclosing
name. Numeric event fields, event ordering and all other bytes are unchanged.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

PATTERN=re.compile(r'("name":\s*)"(CDB::(?:target_dispatch::|draft_graph_forward::)[^\r\n]*?)"(?=,\s*"pid":)')


def repair(text):
    names=Counter();changed=Counter()
    def escaped(match):
        name=match[2]
        suffix=(name.split('::',3)[3] if name.startswith('CDB::target_dispatch::')
                else name.split('::',2)[2])
        key=json.loads(suffix)
        if key is not None and not isinstance(key,list):raise ValueError('Unknown annotation key')
        encoded=json.dumps(name,ensure_ascii=False)
        assert json.loads(encoded)==name
        replacement=match[1]+encoded
        names[name]+=1
        if replacement!=match[0]:changed[name]+=1
        return replacement
    output=PATTERN.sub(escaped,text)
    data=json.loads(output)
    decoded=Counter(e.get('name') for e in data['traceEvents'] if e.get('name') in names)
    assert decoded==names
    return output,dict(matched_names=sum(names.values()),changed_names=sum(changed.values()),
                       changed_labels=dict(changed),events=len(data['traceEvents']))


if __name__=='__main__':
    source=Path(sys.argv[1]);dest=Path(sys.argv[2]);manifest=Path(sys.argv[3])
    assert source.resolve()!=dest.resolve() and not dest.exists() and not manifest.exists()
    raw=source.read_bytes();text=gzip.decompress(raw).decode()
    result,stats=repair(text)
    packed=gzip.compress(result.encode(),mtime=0);dest.write_bytes(packed)
    stats.update(original=str(source),original_compressed_sha256=hashlib.sha256(raw).hexdigest(),
                 original_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                 repaired=str(dest),repaired_compressed_sha256=hashlib.sha256(packed).hexdigest(),
                 change_scope='Only JSON escaping of known CDB range name fields. Logical names preserved; all other event bytes untouched.')
    manifest.write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps({k:v for k,v in stats.items() if k!='changed_labels'},indent=2))
