"""Generation-only v2 constructions; numerical failure remains numerical failure.

The caller verifies the original failed result, frozen parent/source hashes and
all frozen validation tokens using the existing laundering input helpers before
calling this API. Real execution requires CUDA with TF32 disabled. This module
does not load source weights, download, change the original transform arithmetic,
replace formal inventory slots, or grant G1 credit.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from .artifacts import atomic_json, content_hash, write_checkpoint
from .schema import validate_id
from .transforms import _transform, check_equivalence

PROTOCOL_ID = 'public-transform-equivalence-v2'
COHORT_ID = 'public-transform-drift-v2'


def make_drift_recipe(frozen_recipe, original_failure):
    """Bind an immutable original recipe and portable failed-result reference."""
    if frozen_recipe.get('method') not in {'mlp_permutation', 'value_rotation'}:
        raise ValueError('v2 drift cohort permits only original permutations and rotations')
    original_id = frozen_recipe['model_id']
    validate_id(original_id)
    if original_id.endswith('-drift-v2'):
        raise ValueError('expected original failed recipe, not an existing v2 recipe')
    path = Path(original_failure.get('path', ''))
    if (not path.name or path.is_absolute() or '..' in path.parts
            or not re.fullmatch('[0-9a-f]{64}', original_failure.get('sha256', ''))):
        raise ValueError('portable original failure path and SHA256 required')
    recipe = copy.deepcopy(frozen_recipe)
    recipe['model_id'] = original_id + '-drift-v2'
    recipe['drift_protocol'] = dict(protocol_id=PROTOCOL_ID, cohort_id=COHORT_ID,
        original_recipe=copy.deepcopy(frozen_recipe), original_recipe_sha256=content_hash(frozen_recipe),
        original_failure=copy.deepcopy(original_failure), approval_scope='generation_only', quota_credit=0)
    return recipe


def _validate_recipe(recipe):
    protocol = recipe.get('drift_protocol', {})
    original = protocol.get('original_recipe', {})
    expected = make_drift_recipe(original, protocol.get('original_failure', {}))
    if recipe != expected:
        raise ValueError('v2 recipe mismatch with frozen construction')


def _execution(model, synthetic):
    devices = {p.device for p in model.parameters()}
    if synthetic:
        if any(d.type != 'cpu' for d in devices):
            raise ValueError('synthetic fixture execution requires CPU')
    elif (len(devices) != 1 or next(iter(devices)).type != 'cuda'
          or torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('real v2 execution requires one CUDA device and TF32 disabled')


def _finite_state(model):
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(name + ': nonfinite construction state')


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _same_state(expected, actual):
    first, second = expected.state_dict(), actual.state_dict()
    if set(first) != set(second):
        raise ValueError('reloaded state mismatch: tensor names')
    for name, value in first.items():
        other = second[name]
        if value.dtype != other.dtype or not torch.equal(value, other.to(value.device)):
            raise ValueError('reloaded state mismatch: ' + name)
    if expected.config.tie_word_embeddings:
        if actual.get_input_embeddings().weight.data_ptr() != actual.get_output_embeddings().weight.data_ptr():
            raise ValueError('reloaded tied embeddings are not tied')


class _FiniteLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train(model.training)

    def forward(self, tokens):
        result = self.model(tokens)
        if not torch.isfinite(result.logits).all():
            raise ValueError('nonfinite validation logits')
        return result


def _measure(first, second, batches):
    return check_equivalence(_FiniteLogits(first), _FiniteLogits(second), batches)


def _validation_batches(batches):
    # Separate public calls cannot reuse an exhausted iterator. Require a reusable
    # collection, then freeze its iteration order once for all checks in this call.
    if iter(batches) is batches:
        raise ValueError('validation batches must be a reusable nonempty collection')
    batches = tuple(batches)
    if (not batches or any(not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long
                           or tokens.ndim != 2 or not tokens.numel() for tokens in batches)):
        raise ValueError('validation batches require nonempty rank-two int64 tensors')
    return batches


def _verify_token_contract(batches, *reports):
    digest = hashlib.sha256()
    count = 0
    for tokens in batches:
        digest.update(str(tuple(tokens.shape)).encode())
        digest.update(tokens.detach().cpu().contiguous().numpy().tobytes())
        count += tokens.numel()
    if count <= 0 or any(report.get('token_count') != count
                         or report.get('tokens_sha256') != digest.hexdigest() for report in reports):
        raise ValueError('validation token mismatch across complete dtype/reload audits')


def _functional(*reports):
    return 'PASSED' if all(r['passed'] for r in reports) else 'FAILED'


def audit_drift(model, recipe, batches, *, synthetic=False):
    """Return (edited BF16 model, audit), retaining failed unchanged checks.

    Construction status is VERIFIED here; artifact completion requires the safe
    serialization/reload checks in save_drift. The original FP32 model stays at
    FP32, allowing the caller to retain its exact baseline for reload validation.
    """
    batches = _validation_batches(batches)
    _validate_recipe(recipe)
    _execution(model, synthetic)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('v2 audit requires the original FP32 source model')
    _finite_state(model)
    original_digest = _state_sha256(model)
    edited = copy.deepcopy(model).eval()
    state, audit = _transform(model.state_dict(), model.config.to_dict(), recipe)
    edited.load_state_dict(state, strict=True)
    # Check the actual loaded state, including tied aliases, before quantization.
    for name, value in state.items():
        if not torch.equal(value, edited.state_dict()[name]):
            raise ValueError('loaded construction state mismatch: ' + name)
    del state
    audit['fp32'] = _measure(model, edited, batches)
    baseline = copy.deepcopy(model).to(torch.bfloat16).eval()
    edited.to(torch.bfloat16)
    _finite_state(edited)
    audit['bfloat16'] = _measure(baseline, edited, batches)
    _verify_token_contract(batches, audit['fp32'], audit['bfloat16'])
    audit.update(protocol_id=PROTOCOL_ID, cohort_id=COHORT_ID, synthetic=synthetic,
        artifact_status='PENDING_SERIALIZATION', construction_status='VERIFIED',
        functional_equivalence=_functional(audit['fp32'], audit['bfloat16']),
        quota_credit=0, recipe_sha256=content_hash(recipe), baseline_state_sha256=original_digest,
        edited_state_sha256=_state_sha256(edited))
    return edited, audit


def verify_reloaded(baseline, edited, reloaded, batches, *, synthetic=False):
    """Require exact finite BF16 state/ties; rerun the unchanged BF16 check."""
    batches = _validation_batches(batches)
    for model in (baseline, edited, reloaded):
        _execution(model, synthetic)
        _finite_state(model)
    if any(p.dtype != torch.bfloat16 for p in edited.parameters()):
        raise ValueError('serialized v2 construction must be BF16')
    _same_state(edited, reloaded)
    baseline_bf = copy.deepcopy(baseline).to(torch.bfloat16).eval()
    report = _measure(baseline_bf, reloaded, batches)
    _verify_token_contract(batches, report)
    return report


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_files_sha256(output):
    return {p.relative_to(output).as_posix(): _file_sha256(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p != output / 'cohort.json'}


def save_drift(root, recipe, baseline, edited, audit, batches, *, synthetic=False, resume=False):
    """Atomically write a new immutable artifact and version-2 cohort sidecar.

    root must be a separate cohort destination, never the old formal-slot root.
    Reload uses a builtin local safetensors loader and the original audit device.
    Resume verifies sidecar, file hashes, exact reloaded state, and frozen tokens.
    """
    from transformers import AutoModelForCausalLM
    batches = _validation_batches(batches)
    _verify_token_contract(batches, audit['fp32'], audit['bfloat16'])
    _validate_recipe(recipe)
    for model in (baseline, edited):
        _execution(model, synthetic)
        _finite_state(model)
    if (audit.get('recipe_sha256') != content_hash(recipe) or audit.get('synthetic') != synthetic
            or audit.get('baseline_state_sha256') != _state_sha256(baseline)
            or audit.get('edited_state_sha256') != _state_sha256(edited)):
        raise ValueError('audit recipe or construction state mismatch')
    complete = None

    def reload_report(output):
        reloaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16).eval()
        reloaded.to(next(edited.parameters()).device)
        report = verify_reloaded(baseline, edited, reloaded, batches, synthetic=synthetic)
        _verify_token_contract(batches, audit['fp32'], audit['bfloat16'], report)
        return report

    def writer(output):
        nonlocal complete
        edited.save_pretrained(output, safe_serialization=True)
        report = reload_report(output)
        complete = dict(audit, schema_version='2', artifact_status='COMPLETE_VERIFIED_CONSTRUCTION',
            functional_equivalence=_functional(audit['fp32'], audit['bfloat16'], report),
            reloaded_bfloat16=report, reloaded_state_exact=True,
            original_failure=recipe['drift_protocol']['original_failure'],
            weight_sha256={p.name: _file_sha256(p) for p in sorted(output.glob('*.safetensors'))})
        atomic_json(output / 'cohort.json', complete)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / recipe['model_id']
    if not target.exists() and not target.is_symlink():
        # The shared writer adds recipe.json and lineage.json after writer(). Seal
        # those files too, inside an outer staging directory, before publication.
        with tempfile.TemporaryDirectory(prefix='.' + recipe['model_id'] + '-', dir=root) as staging:
            staged = write_checkpoint(recipe['model_id'], Path(staging), dict(recipe, audit=audit), writer)
            complete['checkpoint_files_sha256'] = _checkpoint_files_sha256(staged)
            atomic_json(staged / 'cohort.json', complete)
            if target.exists() or target.is_symlink():
                raise ValueError('v2 output appeared during write')
            os.rename(staged, target)
        return target
    target = write_checkpoint(recipe['model_id'], root, dict(recipe, audit=audit), writer, resume=resume)
    if complete is None:
        try:
            saved = json.loads((target / 'cohort.json').read_text())
            if (saved['schema_version'] != '2' or saved['protocol_id'] != PROTOCOL_ID
                    or saved['artifact_status'] != 'COMPLETE_VERIFIED_CONSTRUCTION'
                    or saved['cohort_id'] != COHORT_ID or not saved['reloaded_state_exact']
                    or saved['quota_credit'] != 0
                    or saved['original_failure'] != recipe['drift_protocol']['original_failure']
                    or saved['checkpoint_files_sha256'] != _checkpoint_files_sha256(target)
                    or saved['functional_equivalence'] != _functional(
                        audit['fp32'], audit['bfloat16'], saved['reloaded_bfloat16'])
                    or any(saved.get(k) != value for k, value in audit.items()
                           if k not in {'artifact_status', 'functional_equivalence'})
                    or saved['weight_sha256'] != {p.name: _file_sha256(p) for p in sorted(target.glob('*.safetensors'))}
                    or saved['reloaded_bfloat16'] != reload_report(target)):
                raise ValueError('cohort verification mismatch')
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError('v2 resume mismatch or incomplete cohort') from exc
    return target
