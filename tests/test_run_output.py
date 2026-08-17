import json
import tempfile
import unittest
from pathlib import Path

from run_output import prepare_run_output


class RunOutputTests(unittest.TestCase):
    def test_explicit_run_id_creates_manifest_and_output_stem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = prepare_run_output(
                "sim.py", output_root=temp_dir, run_id="paper-baseline"
            )

            self.assertEqual(result.run_id, "paper-baseline")
            self.assertEqual(
                result.run_dir, Path(temp_dir).resolve() / "paper-baseline"
            )
            self.assertEqual(result.output_stem, str(result.run_dir / "output"))

            manifest = json.loads(
                (result.run_dir / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["run_id"], "paper-baseline")
            self.assertEqual(manifest["entrypoint"], "sim.py")
            self.assertEqual(manifest["output_stem"], result.output_stem)

    def test_existing_run_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepare_run_output("sim.py", output_root=temp_dir, run_id="same-run")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_run_output(
                    "sim.py", output_root=temp_dir, run_id="same-run"
                )

    def test_path_traversal_run_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "path-safe"):
                prepare_run_output(
                    "sim.py", output_root=temp_dir, run_id="../escape"
                )

    def test_generated_run_ids_are_unique(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = prepare_run_output("sim.py", output_root=temp_dir)
            second = prepare_run_output("sim.py", output_root=temp_dir)
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertTrue(first.run_dir.is_dir())
            self.assertTrue(second.run_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
