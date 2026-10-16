"""Reserved V2 interface; full Purged K-Fold and embargo are intentionally absent."""


class PurgedKFold:
    """Explicit placeholder that cannot silently substitute a random split."""

    def split(self, X, y=None, groups=None):
        """Raise until a separate V2 design and leakage audit are authorized."""
        raise NotImplementedError('Full Purged K-Fold and embargo are reserved for V2')
