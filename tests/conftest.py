import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Dislikes (1 star) come first in time, likes (5 stars) later, so a chronological split holds out likes.
# Two taste groups: users 1-3 like sci-fi (items 1-4), users 4-6 like romance (items 5-8).
# User 7 has a single rating, user 1 has not rated item 4 (the obvious recommendation for them).
ITEMS = pd.DataFrame(
    {
        "title": [
            "Star Fleet (1999)",
            "Galaxy Run (2001)",
            "Robot Dawn (2003)",
            "Nebula Wars (2005)",
            "Love Letters (1999)",
            "Paris Summer (2001)",
            "Wedding Bells (2003)",
            "Kiss Me Twice (2005)",
            "Unwatched (2010)",
        ],
        "genres": ["Sci-Fi"] * 4 + ["Romance"] * 4 + ["Drama"],
        "tags": ["space"] * 4 + ["love"] * 4 + [""],
    },
    index=pd.Index(range(1, 10), name="item_id"),
)


def _ratings():
    rows = []
    scifi, romance = [1, 2, 3, 4], [5, 6, 7, 8]
    for user in (1, 2, 3):
        for t, item in enumerate(scifi):
            if user == 1 and item == 4:
                continue
            rows.append((user, item, 5.0, 100 + t))
        for t, item in enumerate(romance[:2]):
            rows.append((user, item, 1.0, 50 + t))
    for user in (4, 5, 6):
        for t, item in enumerate(romance):
            rows.append((user, item, 5.0, 100 + t))
        for t, item in enumerate(scifi[:2]):
            rows.append((user, item, 1.0, 50 + t))
    rows.append((7, 1, 4.0, 100))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


@pytest.fixture
def items():
    return ITEMS.copy()


@pytest.fixture
def ratings():
    return _ratings()
