# SPDX-License-Identifier: Apache-2.0
"""Optional memoized digest of an EXACT, typed immutable formal prefix.

Never cache by request ID/length alone. Every hit compares every token, and only
builtin ints can hit a previously validated prefix. All other inputs use the
original validator/hasher. No request, KV page, lease or mutable context is held.
"""
from dataclasses import dataclass
import json
import logging
import os
import threading
import weakref

logger=logging.getLogger(__name__)


def flag(name,env=None):
    value=(os.environ if env is None else env).get(name,'0')
    if value not in ('0','1'):raise ValueError(f'{name} must be 0 or 1')
    return value=='1'


@dataclass(frozen=True)
class Entry:
    tokens: tuple
    fingerprint: str
    hasher: object  # weak reference; callbacks cannot retain a Request here.


class ContextFingerprintCache:
    def __init__(self,*,max_entries=64,max_tokens=1024,validate=False):
        if (type(max_entries) is not int or not 1<=max_entries<=256
                or type(max_tokens) is not int or not 1<=max_tokens<=1024):
            raise ValueError('Unbounded context fingerprint memoization')
        self.max_entries=max_entries;self.max_tokens=max_tokens;self.validate=validate
        self.entries=weakref.WeakKeyDictionary();self.owner=threading.get_ident()
        self.calls=self.hits=self.misses=self.fallbacks=self.audits=0

    @staticmethod
    def native(tokens,hasher):
        return dict(context_tokens=len(tokens),fingerprint=hasher(tokens))

    def context(self,request,tokens,hasher):
        if threading.get_ident()!=self.owner:
            return self.native(tokens,hasher)  # no shared cache mutation
        self.calls+=1
        if (len(tokens)>self.max_tokens or type(request).__hash__ is not object.__hash__
                or type(request).__eq__ is not object.__eq__
                or not (_NATIVE_TYPES.all_builtin_ints(tokens)
                        if NATIVE_TYPES and type(tokens) in (list, tuple)
                        else all(type(token) is int for token in tokens))):
            self.fallbacks+=1
            return self.native(tokens,hasher)
        try:
            weakref.ref(request);hasher_ref=weakref.ref(hasher)
        except TypeError:
            self.fallbacks+=1
            return self.native(tokens,hasher)
        immutable=tuple(tokens)
        previous=self.entries.get(request)
        if (previous is not None and previous.hasher() is hasher
                and previous.tokens==immutable):
            # Immutable builtin-int equality to a validated old tuple proves
            # range/count/encoding identity, including same-length mutations.
            digest=previous.fingerprint;self.hits+=1
            if self.validate and self.audits<32:
                expected=hasher(tokens)
                exact=digest==expected
                logger.info('[CDB_CONTEXT_FP_VALIDATE] %s',json.dumps(dict(
                    sample=self.audits+1,request_id=getattr(request,'request_id',None),
                    context_tokens=len(tokens),fingerprint_exact=exact)))
                if not exact:raise RuntimeError('Fingerprint cache/native mismatch')
                self.audits+=1
        else:
            self.misses+=1
            digest=hasher(tokens)  # original validation/errors, never swallowed
            if isinstance(digest,str):
                if request not in self.entries and len(self.entries)>=self.max_entries:
                    del self.entries[next(iter(self.entries))]
                self.entries[request]=Entry(immutable,digest,hasher_ref)
        if self.calls==1 or self.calls%1024==0:
            logger.info('[CDB_CONTEXT_FP_CACHE] %s',json.dumps(dict(
                source=__file__,calls=self.calls,hits=self.hits,misses=self.misses,
                fallbacks=self.fallbacks,entries=len(self.entries),max_entries=self.max_entries,
                full_typed_prefix_equality=True,request_weak_references=True,
                owner_thread_only=True,native_exact_type_check=NATIVE_TYPES,
                native_type_calls=_NATIVE_TYPES.calls() if NATIVE_TYPES else 0,
                native_module=_NATIVE_TYPES.__file__ if NATIVE_TYPES else None)))
        # Do not share a mutable dict with a caller or stored Carry snapshot.
        return dict(context_tokens=len(tokens),fingerprint=digest)


ENABLED=flag('CDB_AV_CONTEXT_FINGERPRINT_CACHE')
NATIVE_TYPES=flag('CDB_AV_CONTEXT_NATIVE_TYPES')
_NATIVE_TYPES=None
if NATIVE_TYPES:
    if not ENABLED:raise ValueError('Native type predicate requires context fingerprint cache')
    from vllm.v1.spec_decode import _cdb_av_prefix_types as _NATIVE_TYPES
VALIDATE=flag('CDB_AV_CONTEXT_FINGERPRINT_VALIDATE')
_cache=None


def cached_context(request,tokens,hasher):
    global _cache
    if not ENABLED:raise RuntimeError('Context memoization called while disabled')
    if _cache is None:_cache=ContextFingerprintCache(validate=VALIDATE)
    return _cache.context(request,tokens,hasher)
