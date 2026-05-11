from __future__ import annotations

import ast
import unittest
from pathlib import Path


FORBIDDEN = {"transcribe", "transcribe_chunked"}


class TestNoLegacyImports(unittest.TestCase):
    def test_no_legacy_module_imports(self) -> None:
        root = Path("meeting_v15")
        offenders: list[str] = []

        for file_path in root.rglob("*.py"):
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in FORBIDDEN:
                            offenders.append(f"{file_path}: import {alias.name}")
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top in FORBIDDEN:
                            offenders.append(f"{file_path}: from {node.module} import ...")

        self.assertEqual(offenders, [], msg="\n".join(offenders))


if __name__ == "__main__":
    unittest.main()

