"""Shadow result decisions; acceptance is not permission to publish or actuate."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceDecision:
    reasons: tuple[str, ...] = ()
    reconcile: bool = False

    @property
    def accepted(self):
        return not self.reasons


def compare_revisions(consumed, current):
    changed = tuple(sorted(key for key in consumed.keys() | current.keys()
                           if consumed.get(key) != current.get(key)))
    if changed:
        # A process configuration change requires restart, not rapid reruns.
        return AcceptanceDecision(tuple(f'input_changed:{key}' for key in changed), 'config' not in changed)
    return AcceptanceDecision()
