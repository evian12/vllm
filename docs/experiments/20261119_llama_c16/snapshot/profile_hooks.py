"""CPU range labels for a bounded private Torch trace; no tensor mutations."""
import functools

import manual_observer


def wrap(cls, name, label):
    import torch
    original = getattr(cls, name)

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        text = label(self, args, kwargs) if callable(label) else label
        with torch.profiler.record_function(text):
            return original(self, *args, **kwargs)

    setattr(cls, name, observed)


def augment(module_name, operation):
    old = manual_observer.PATCHES.get(module_name)

    def patch(module):
        if old is not None:
            old(module)
        operation(module)

    manual_observer.PATCHES[module_name] = patch


def refill(module):
    wrap(module.RefillSession, '_forward',
         lambda self, args, kwargs: 'CDB::draft_forward::' + str(args[5])
         + '::r' + str(len(args[4])) + '::t' + str(args[0].numel()))


def reuse(module):
    wrap(module.ReuseDraftSession, 'step', 'CDB::draft_step_with_sampling')
    wrap(module.ReuseDraftSession, 'prepare', 'CDB::draft_prepare')
    wrap(module.ReuseDraftSession, '_clear_slots',
         lambda self, args, kwargs: 'CDB::clear_slots::' + str(args[2] - args[1]))
    wrap(module.ReuseDraftQueue, 'before_target', 'CDB::kv_ownership_before_target')
    wrap(module.ReuseDraftQueue, 'plans', 'CDB::draft_plans')


def runner(module):
    def target_label(self, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        return 'CDB::target_forward::t' + str(ids.numel() if ids is not None else 'none')

    wrap(module.GPUModelRunner, '_model_forward', target_label)
    wrap(module.GPUModelRunner, 'sample_tokens', 'CDB::target_sampling')


def draft(module):
    original = module.DraftModelProposer.load_model

    @functools.wraps(original)
    def loaded(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        kind = type(self.model)
        if not getattr(kind, '_cdb_profile_logits_wrapped', False):
            wrap(kind, 'compute_logits', lambda instance, args, kwargs:
                 'CDB::draft_logits' if getattr(instance, '_baseline_observed_draft', False)
                 else 'CDB::target_logits')
            kind._cdb_profile_logits_wrapped = True
        return result

    module.DraftModelProposer.load_model = loaded


augment('vllm.v1.worker.gpu_model_runner', runner)
augment('vllm.v1.spec_decode.draft_model', draft)
augment('vllm.v1.spec_decode.cdb_av_refill', refill)
augment('vllm.v1.spec_decode.cdb_av_kv_reuse', reuse)
augment('vllm.v1.spec_decode.cdb_av_target_graph',
        lambda m: wrap(m.TargetFullGraph, '_storage', 'CDB::target_storage_audit'))
augment('vllm.v1.spec_decode.cdb_av_inputs',
        lambda m: wrap(m.PinnedDraftInputStager, 'pack', 'CDB::draft_metadata_pack'))
