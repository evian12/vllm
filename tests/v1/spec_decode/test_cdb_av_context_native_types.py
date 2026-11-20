# SPDX-License-Identifier: Apache-2.0
"""CPU-only native predicate contract; original cache/frozen tests run too."""
import gc
import random
import weakref

import pytest

native = pytest.importorskip(
    'vllm.v1.spec_decode._cdb_av_prefix_types',
    reason='Optional CPython predicate has not been built',
    exc_type=ModuleNotFoundError,
)


@pytest.mark.parametrize('tokens',[[],(),[1,2,3],(0,2**32-1),[-1,2**256]])
def test_only_types_are_checked_not_ranges(tokens):
    assert native.all_builtin_ints(tokens) is True


@pytest.mark.parametrize('value',[True,False,1.0,'1',None,object()])
def test_non_builtin_int_cannot_match_cached_integer(value):
    assert native.all_builtin_ints([1,value]) is False


def test_integer_subclass_is_rejected_without_comparison_callback():
    class Special(int):
        def __eq__(self,other):raise AssertionError('Do not execute token callbacks')
    assert native.all_builtin_ints([Special(1)]) is False


def test_unsupported_sequences_are_not_consumed():
    class SpecialList(list):
        def __iter__(self):raise AssertionError('Native must not consume subclass')
    value=SpecialList([1])
    assert native.all_builtin_ints(value) is NotImplemented
    generator=iter([1,2])
    assert native.all_builtin_ints(generator) is NotImplemented
    assert next(generator)==1
    assert native.all_builtin_ints(None) is NotImplemented


def test_random_cases_match_original_builtin_type_expression():
    rng=random.Random(42)
    pool=[0,1,2**32-1,2**80,-3,False,1.0,None,'1']
    for n in [0,1,255,256,257,1024,1025]:
        for _ in range(20):
            values=[rng.choice(pool) for _ in range(n)]
            for tokens in [values,tuple(values)]:
                assert native.all_builtin_ints(tokens) is all(type(t) is int for t in tokens)


def test_no_sequence_or_token_is_retained():
    class Token:pass
    token=Token();ref=weakref.ref(token)
    assert native.all_builtin_ints([token]) is False
    del token;gc.collect();assert ref() is None


def test_counter_reports_actual_calls():
    before=native.calls();native.all_builtin_ints([1])
    assert native.calls()==before+1


def test_correct_argument_contract():
    with pytest.raises(TypeError):native.all_builtin_ints()
    with pytest.raises(TypeError):native.all_builtin_ints([1],[2])
