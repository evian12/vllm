"""CPU prototype only. No Queue wire-up, model or GPU acceptance claim."""
from dataclasses import replace
import gc
import importlib.util
from pathlib import Path
import sys
import threading
import unittest
import weakref


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


ROOT=Path(__file__).parent
SOURCE=(ROOT if (ROOT/'cdb_av_context_cache.py').exists() else
        ROOT.parents[2]/'vllm/v1/spec_decode')
m=load('cache_prototype',SOURCE/'cdb_av_context_cache.py')
reference=load('original_fingerprint',SOURCE/('cdb_av_fingerprint_reference.py'
               if (SOURCE/'cdb_av_fingerprint_reference.py').exists() else 'cdb_av_fingerprint.py'))


class Request:
    request_id='same_label'


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.cache=m.ContextFingerprintCache();self.req=Request();self.calls=0
        def digest(tokens):
            self.calls+=1;return reference._batched_token_fingerprint(tokens)
        self.digest=digest

    def context(self,tokens,request=None,hasher=None):
        return self.cache.context(self.req if request is None else request,tokens,
                                  self.digest if hasher is None else hasher)

    def test_exact_frozen_prefix_avoids_rehash(self):
        a=self.context([1,2,3]);b=self.context([1,2,3])
        self.assertEqual(a,b);self.assertIsNot(a,b)
        self.assertEqual((self.calls,self.cache.hits),(1,1))

    def test_same_length_mutation_is_detected(self):
        a=self.context([1,2,3]);b=self.context([1,8,3])
        self.assertNotEqual(a,b);self.assertEqual(self.calls,2)

    def test_mutable_input_cannot_change_saved_tuple(self):
        ids=[1,2];self.context(ids);ids[0]=7
        self.assertEqual(self.cache.entries[self.req].tokens,(1,2))
        self.assertEqual(self.context(ids)['fingerprint'],self.digest(ids))

    def test_growth_and_truncation_recompute(self):
        for ids in [[1],[1,2],[1,2,3],[1,2]]:
            value=self.context(ids)
            self.assertEqual(value['fingerprint'],reference._batched_token_fingerprint(ids))
        self.assertEqual(self.calls,4)

    def test_request_id_is_not_the_cache_identity(self):
        a=self.context([1]);b=self.context([2],Request())
        self.assertNotEqual(a,b);self.assertEqual(self.cache.hits,0)

    def test_different_hasher_cannot_hit(self):
        self.context([1])
        def changed(tokens):return 'new-personality'
        self.assertEqual(self.context([1],hasher=changed)['fingerprint'],'new-personality')

    def test_invalid_values_preserve_original_error(self):
        self.context([1])
        for invalid in [-1,2**32,1.0,'1']:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):self.context([invalid])

    def test_equal_numeric_non_int_never_hits(self):
        class EqualOne:
            def __eq__(self,other):return other==1
        self.context([1])
        with self.assertRaises(ValueError):self.context([EqualOne()])
        self.assertEqual(self.cache.hits,0)

    def test_bool_and_int_subclass_use_original(self):
        class Integer(int):pass
        self.context([1])
        for value in [True,Integer(1)]:
            self.assertEqual(self.context([value])['fingerprint'],reference._batched_token_fingerprint([value]))
        self.assertEqual(self.cache.hits,0)

    def test_unweakrefable_and_unhashable_requests_fall_back(self):
        class Slots:__slots__=()
        class NoHash:__hash__=None
        for request in [Slots(),NoHash()]:
            for _ in range(2):self.context([1],request)
        self.assertEqual(self.calls,4);self.assertEqual(len(self.cache.entries),0)

    def test_collected_request_releases_entry(self):
        req=Request();ref=weakref.ref(req);self.context([1],req)
        del req;gc.collect()
        self.assertIsNone(ref());self.assertEqual(len(self.cache.entries),0)

    def test_weak_hasher_cannot_retain_request(self):
        req=Request();ref=weakref.ref(req)
        def digest(tokens,captured=req):return reference._batched_token_fingerprint(tokens)
        self.context([1],req,digest)
        del req,digest;gc.collect()
        self.assertIsNone(ref());self.assertEqual(len(self.cache.entries),0)

    def test_entries_and_prefix_storage_are_bounded(self):
        self.cache=m.ContextFingerprintCache(max_entries=2,max_tokens=8)
        requests=[Request() for _ in range(3)]
        for req in requests:self.context([1],req)
        self.assertEqual(len(self.cache.entries),2)
        req=Request();self.context(list(range(9)),req)
        self.assertNotIn(req,self.cache.entries)

    def test_validation_recomputes_and_is_bounded(self):
        self.cache=m.ContextFingerprintCache(validate=True)
        for _ in range(40):self.context([1,2,3])
        self.assertEqual((self.cache.audits,self.calls),(32,33))

    def test_validation_does_not_silently_heal_corruption(self):
        self.cache=m.ContextFingerprintCache(validate=True)
        self.context([1]);entry=self.cache.entries[self.req]
        self.cache.entries[self.req]=replace(entry,fingerprint='wrong')
        with self.assertRaisesRegex(RuntimeError,'cache/native mismatch'):self.context([1])

    def test_returned_dict_is_not_an_alias(self):
        a=self.context([1]);a['fingerprint']='poison'
        self.assertNotEqual(self.context([1])['fingerprint'],'poison')

    def test_cross_thread_uses_native_without_touching_entries(self):
        result=[];before=self.cache.calls
        thread=threading.Thread(target=lambda:result.append(self.context([1])))
        thread.start();thread.join()
        self.assertEqual(result[0]['fingerprint'],reference._batched_token_fingerprint([1]))
        self.assertEqual((self.cache.calls,len(self.cache.entries)),(before,0))

    def test_boundaries_are_exact_against_original(self):
        for n in [0,1,255,256,257,1024,1025]:
            ids=[2**32-1]*n
            with self.subTest(n=n):
                for _ in range(2):
                    self.assertEqual(self.context(ids)['fingerprint'],reference._batched_token_fingerprint(ids))

    def test_bad_configuration_rejected(self):
        for kw in [dict(max_entries=0),dict(max_entries=True),dict(max_tokens=1025)]:
            with self.assertRaises(ValueError):m.ContextFingerprintCache(**kw)
        with self.assertRaises(ValueError):m.flag('X',{'X':'true'})


if __name__=='__main__':unittest.main()
