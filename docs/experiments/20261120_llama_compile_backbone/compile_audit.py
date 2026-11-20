"""Route existing diagnostic model comparisons through original eager forwards."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps


_target_reference = ContextVar('cdb_target_eager_reference', default=False)
_draft_validation = ContextVar('cdb_draft_validation', default=False)


def wrap_context(module, should_skip, model_name):
    original = module.set_forward_context

    @wraps(original)
    @contextmanager
    def forward_context(*args, **kwargs):
        force_eager = should_skip()
        if force_eager:
            kwargs['skip_compiled'] = True
        with original(*args, **kwargs) as context:
            yield context
        if force_eager:
            module.logger.info('[CDB_COMPILE_EAGER_REFERENCE] %s', model_name)

    module.set_forward_context = forward_context


def target(module):
    original = module.TargetFullGraph._validate
    wrap_context(module, _target_reference.get, 'target')

    @wraps(original)
    def validate(owner, entry, live, caches, model, *args, **kwargs):
        @wraps(model)
        def eager_model(*model_args, **model_kwargs):
            marker = _target_reference.set(True)
            try:
                return model(*model_args, **model_kwargs)
            finally:
                _target_reference.reset(marker)

        return original(owner, entry, live, caches, eager_model, *args, **kwargs)

    module.TargetFullGraph._validate = validate


def draft(module):
    import torch

    original = module.RefillSession._decode_graph_forward
    wrap_context(module, lambda: _draft_validation.get()
                 and not torch.cuda.is_current_stream_capturing(), 'draft')

    @wraps(original)
    def validate(session, *args, **kwargs):
        marker = _draft_validation.set(True)
        try:
            return original(session, *args, **kwargs)
        finally:
            _draft_validation.reset(marker)

    module.RefillSession._decode_graph_forward = validate


def install(observer):
    for module_name, patch in [
        ('vllm.v1.spec_decode.cdb_av_target_graph', target),
        ('vllm.v1.spec_decode.cdb_av_refill', draft),
    ]:
        previous = observer.PATCHES.get(module_name)

        def apply(module, previous=previous, patch=patch):
            if previous is not None:
                previous(module)
            patch(module)

        observer.PATCHES[module_name] = apply
