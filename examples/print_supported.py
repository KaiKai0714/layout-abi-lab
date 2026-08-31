"""Print the declared support matrix. Does not import PyTorch."""

from __future__ import annotations

import json

from layoutabi import supported

if __name__ == "__main__":
    print(json.dumps(supported(), indent=2))
