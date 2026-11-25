DEFAULT_WEIGHTS = {'image': 0.45, 'tabular': 0.40, 'nlp': 0.15}


def _validate(probs):
    for name, val in probs.items():
        if val is not None and not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} probability must be between 0 and 1, got {val}")


def weighted_fusion(image_prob=None, tabular_prob=None, nlp_prob=None,
                    weights=None):
    """
    Combine the available modality probabilities into one malignant probability.

    A modality passed as None (e.g. no scan uploaded, no symptoms typed) is
    DROPPED and the remaining weights are re-normalised - it is not replaced by
    a neutral 0.5, which would silently drag every result towards 'moderate'.

    Default weights are expert-set heuristics (image 0.45 / clinical 0.40 /
    text 0.15), not learned - there is no public dataset with all three
    modalities for the same patient.
    """
    probs = {'image': image_prob, 'tabular': tabular_prob, 'nlp': nlp_prob}
    _validate(probs)
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    avail = {k: v for k, v in probs.items() if v is not None}
    if not avail:
        raise ValueError("At least one modality is required for fusion.")
    total = sum(w[k] for k in avail)
    return round(float(sum(w[k] / total * v for k, v in avail.items())), 4)


def confidence_weighted_fusion(image_prob=None, tabular_prob=None, nlp_prob=None):
    """Weight each available modality by |p - 0.5| (how decisive it is)."""
    probs = {'image': image_prob, 'tabular': tabular_prob, 'nlp': nlp_prob}
    _validate(probs)
    avail = {k: v for k, v in probs.items() if v is not None}
    if not avail:
        raise ValueError("At least one modality is required for fusion.")
    conf = {k: abs(v - 0.5) * 2 for k, v in avail.items()}
    total = sum(conf.values())
    if total == 0:
        return round(sum(avail.values()) / len(avail), 4)
    return round(float(sum(conf[k] / total * v for k, v in avail.items())), 4)


def modality_agreement(image_prob=None, tabular_prob=None, nlp_prob=None):
    """
    Flags conflicting evidence: returns dict(agree: bool, spread: float,
    note: str). 'spread' = max - min of the available probabilities.
    """
    vals = [v for v in (image_prob, tabular_prob, nlp_prob) if v is not None]
    if len(vals) < 2:
        return {'agree': True, 'spread': 0.0,
                'note': 'Only one modality available - no cross-check possible.'}
    spread = round(max(vals) - min(vals), 4)
    sides = {v >= 0.5 for v in vals}
    agree = len(sides) == 1
    if agree and spread < 0.4:
        note = 'All available modalities point the same way.'
    elif agree:
        note = 'Modalities agree on direction but differ in strength.'
    else:
        note = 'CONFLICT: modalities disagree - manual clinical review advised.'
    return {'agree': agree, 'spread': spread, 'note': note}
