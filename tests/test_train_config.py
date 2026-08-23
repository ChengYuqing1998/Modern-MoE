import tempfile
import unittest
import os
import json
from pathlib import Path
from unittest.mock import patch

from scripts.train import (
    configure_wandb_network,
    resolve_experiment,
    save_managed_checkpoint,
)


class ExperimentResolutionTest(unittest.TestCase):
    def config(self, root: Path, resume: bool = False) -> dict:
        config = {
            "checkpoint_root": str(root),
            "resume_training": resume,
        }
        if resume:
            config["experiment_id"] = "exp_007"
        return config

    def test_new_experiment_generates_unique_id_and_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment_id, output_dir, resume, checkpoint = resolve_experiment(
                self.config(root), None
            )
            self.assertRegex(
                experiment_id,
                r"^exp_\d{8}_\d{6}_[0-9a-f]{8}$",
            )
            self.assertEqual(output_dir, root / experiment_id)
            self.assertTrue(output_dir.is_dir())
            self.assertFalse(resume)
            self.assertIsNone(checkpoint)

    def test_resume_requires_experiment_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                resolve_experiment(
                    self.config(Path(temporary), resume=True),
                    None,
                )

    def test_resume_selects_latest_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "exp_007"
            output_dir.mkdir()
            (output_dir / "step_0000010.pt").touch()
            latest = output_dir / "step_0000250.pt"
            latest.touch()
            _, _, resume, checkpoint = resolve_experiment(
                self.config(root, resume=True), None
            )
            self.assertTrue(resume)
            self.assertEqual(checkpoint, latest)

    def test_resume_prefers_newer_stage_step_over_old_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "exp_007"
            output_dir.mkdir()
            final = output_dir / "final.pt"
            final.touch()
            step = output_dir / "step_0006000.pt"
            step.touch()
            os.utime(final, ns=(1, 1))
            os.utime(step, ns=(2, 2))
            _, _, resume, checkpoint = resolve_experiment(
                self.config(root, resume=True), None
            )
            self.assertTrue(resume)
            self.assertEqual(checkpoint, step)

    def test_new_runs_get_different_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = resolve_experiment(self.config(root), None)
            second = resolve_experiment(self.config(root), None)
            self.assertNotEqual(first[0], second[0])
            self.assertTrue(first[1].is_dir())
            self.assertTrue(second[1].is_dir())

    def test_resume_requires_experiment_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), resume=True)
            del config["experiment_id"]
            with self.assertRaises(ValueError):
                resolve_experiment(config, None)

    def test_wandb_direct_mode_removes_proxy_for_training_process(self):
        environment = {
            "HTTP_PROXY": "http://proxy.invalid:1234",
            "https_proxy": "http://proxy.invalid:1234",
            "NO_PROXY": "localhost",
        }
        with patch.dict(os.environ, environment, clear=True):
            configure_wandb_network({"wandb_disable_proxy": True})
            self.assertNotIn("HTTP_PROXY", os.environ)
            self.assertNotIn("https_proxy", os.environ)
            self.assertIn("api.wandb.ai", os.environ["NO_PROXY"])
            self.assertIn("localhost", os.environ["NO_PROXY"])


class CheckpointRetentionTest(unittest.TestCase):
    def test_keeps_best_latest_and_two_most_recent_with_step_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "final.pt").touch()

            def fake_save(path, *_args, **_kwargs):
                path.touch()

            with patch("scripts.train.save_checkpoint", side_effect=fake_save):
                for step in range(1, 7):
                    save_managed_checkpoint(
                        output_dir,
                        None,
                        None,
                        None,
                        0,
                        step,
                        step,
                        step,
                        step,
                        {},
                        None,
                        validation_loss=0.5 if step == 2 else None,
                        best_validation_loss=0.5,
                        best_validation_step=2,
                        max_checkpoints=4,
                    )

            self.assertEqual(
                {path.name for path in output_dir.glob("step_*.pt")},
                {
                    "step_0000002.pt",
                    "step_0000004.pt",
                    "step_0000005.pt",
                    "step_0000006.pt",
                },
            )
            self.assertFalse((output_dir / "final.pt").exists())
            manifest = json.loads(
                (output_dir / "checkpoint_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["latest_step"], 6)
            self.assertEqual(manifest["best_step"], 2)
            by_step = {
                record["step"]: record for record in manifest["checkpoints"]
            }
            self.assertEqual(by_step[6]["roles"], ["latest"])
            self.assertEqual(by_step[2]["roles"], ["best"])
            self.assertTrue(all(record["saved_at"] for record in by_step.values()))

    def test_new_best_releases_old_best_and_keeps_strict_file_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)

            def fake_save(path, *_args, **_kwargs):
                path.touch()

            with patch("scripts.train.save_checkpoint", side_effect=fake_save):
                for step in range(1, 5):
                    save_managed_checkpoint(
                        output_dir,
                        None,
                        None,
                        None,
                        0,
                        step,
                        step,
                        step,
                        step,
                        {},
                        None,
                        validation_loss=0.5 if step == 1 else None,
                        best_validation_loss=0.5,
                        best_validation_step=1,
                        max_checkpoints=4,
                    )
                save_managed_checkpoint(
                    output_dir,
                    None,
                    None,
                    None,
                    0,
                    5,
                    5,
                    5,
                    5,
                    {},
                    None,
                    validation_loss=0.4,
                    best_validation_loss=0.4,
                    best_validation_step=5,
                    max_checkpoints=4,
                )

            self.assertEqual(
                {path.name for path in output_dir.glob("step_*.pt")},
                {
                    "step_0000002.pt",
                    "step_0000003.pt",
                    "step_0000004.pt",
                    "step_0000005.pt",
                },
            )
            manifest = json.loads(
                (output_dir / "checkpoint_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["best_step"], 5)
            self.assertEqual(manifest["latest_step"], 5)
            self.assertEqual(manifest["checkpoints"][0]["roles"], ["latest", "best"])


if __name__ == "__main__":
    unittest.main()
