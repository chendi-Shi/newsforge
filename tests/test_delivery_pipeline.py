"""Failure and provenance tests for orchestration, without research training."""
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from newsforge.delivery import pipeline as p


@contextmanager
def current_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class PipelinePlanTests(unittest.TestCase):
    def test_backbone_materializes_actual_internal_winner_not_seed17(self):
        candidates = {seed: {"checkpoint_sha256": f"checkpoint-{seed}", "state_sha256": f"state-{seed}"}
                      for seed in p.SEEDS}
        records = [{"seed": seed, **candidates[seed], "metrics": {"ndcg10": {"mean": score}}}
                   for seed, score in zip(p.SEEDS, (.20, .23, .25))]
        selection = {"candidates": records, "selected": {"seed": 43}}
        self.assertEqual(p.selected_backbone_seed(candidates, selection), 43)
        selection["selected"]["seed"] = 17
        with self.assertRaisesRegex(ValueError, "internal-only decision rule"):
            p.selected_backbone_seed(candidates, selection)
        selection["selected"]["seed"] = 43
        records[1]["state_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            p.selected_backbone_seed(candidates, selection)

    def test_backbone_requires_all_three_seeds_and_uses_frozen_tie_break(self):
        candidates = {seed: {"checkpoint_sha256": f"checkpoint-{seed}", "state_sha256": f"state-{seed}"}
                      for seed in p.SEEDS}
        records = [{"seed": seed, **candidates[seed], "metrics": {"ndcg10": {"mean": .2}}}
                   for seed in reversed(p.SEEDS)]
        selection = {"candidates": records, "selected": {"seed": 17}}
        self.assertEqual(p.selected_backbone_seed(candidates, selection), 17)
        with self.assertRaisesRegex(ValueError, "every fixed backbone seed"):
            p.selected_backbone_seed({17: candidates[17]}, selection)
        selection["candidates"] = records[:2]
        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            p.selected_backbone_seed(candidates, selection)

    def test_plan_contains_real_audit_dependencies_in_order(self):
        plan = p.make_plan()
        self.assertFalse(plan["executed"])
        names = [stage["name"] for stage in plan["stages"]]
        for stage in plan["stages"]:
            for dependency in stage["dependencies"]:
                self.assertLess(names.index(dependency), names.index(stage["name"]))
        chain = ["neural-train", "v1-full-ranking", "backbone-initial", "backbone-provisional",
                 "backbone-materialize", "backbone-final", "encoder", "v2-select",
                 "v3-baseline-gate", "v2-catalog", "v2-audit", "v3-dev", "v3-audit"]
        self.assertEqual([names.index(name) for name in chain], sorted(names.index(name) for name in chain))
        self.assertIn("v3-http", names)
        self.assertIn("v3-plot", names)
        self.assertIn("--skip-test", next(s for s in plan["stages"] if s["name"] == "neural-train")["argv"])

    def test_target_dependency_closure_and_invalid_graphs(self):
        self.assertEqual([s.name for s in p.ordered_stages("data")], ["environment", "unit-tests", "data"])
        a = p.Stage("a", (), ("-c", "pass"), ())
        b = p.Stage("b", ("a",), ("-c", "pass"), ())
        for definitions, target in [((a, a), "a"), ((a,), "missing"),
                                    ((b,), "b"), ((replace(a, dependencies=("b",)), b), "a")]:
            with self.subTest(definitions=definitions), self.assertRaises(ValueError):
                p.ordered_stages(target, definitions)

    def test_cli_plan_runs_without_data_or_third_party_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            with patch.object(sys, "argv", ["pipeline", "plan", "--output", str(output)]):
                p.main()
            self.assertFalse(json.loads(output.read_text())["executed"])
            with patch.object(sys, "argv", ["pipeline", "plan", "--output", str(output)]):
                with self.assertRaises(FileExistsError):
                    p.main()

    def test_cross_platform_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in ("", "../outside", "a/../outside", "./a", "/absolute", "C:/outside",
                         "C:outside", "\\\\server\\share\\x", "a//b", "a\\..\\outside"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    p.inside(directory, path)
            self.assertEqual(p.inside(directory, "a/b"), Path(directory).resolve() / "a/b")


class PipelineWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.workspace = self.base / "run"
        files = {"newsforge/delivery/pipeline.py": "# frozen orchestrator\n",
                 "newsforge/train.py": "# frozen train\n", "newsforge/v3/experiment.py": "# experiment\n",
                 "docs/V3_PROTOCOL.md": "fixed protocol\n", "requirements-windows.lock": "numpy==1.26.4\n",
                 "requirements-plot.txt": "matplotlib==3.10.9\n",
                 "requirements-delivery-windows.lock": "numpy==1.26.4 --hash=sha256:abc\n",
                 "reports/v3/dev.json": '{"historical": true}\n', "artifacts/old.pt": "old weights"}
        for name, value in files.items():
            self.write(self.source / name, value)
        self.python = Path(sys.executable).resolve()
        self.a = p.Stage("first", (), ("-c", "pass"), ("reports/first.json",))
        self.b = p.Stage("second", ("first",), ("-c", "pass"), ("reports/second.json",))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def initialize(self, raw=None):
        return p.initialize(self.source, self.workspace, raw)

    def execute(self, stage_names=("first", "second"), mutate=None, code=0):
        names = iter(stage_names)

        def executor(command, **kwargs):
            name = next(names)
            self.assertEqual(kwargs["cwd"], str(self.workspace.resolve()))
            self.assertFalse(kwargs["check"])
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            kwargs["stdout"].write(b"stage log\n")
            self.write(self.workspace / f"reports/{name}.json", '{"new": true}\n')
            if mutate:
                mutate(name)
            return subprocess.CompletedProcess(command, code)

        return executor

    def run_pipeline(self, target="second", definitions=None, executor=None):
        with redirect_stdout(io.StringIO()):
            return p.run(self.workspace, self.python, target,
                         definitions=(self.a, self.b) if definitions is None else definitions,
                         executor=self.execute() if executor is None else executor)

    def test_init_copies_sources_and_keeps_historical_reports_separate(self):
        manifest = self.initialize()
        self.assertIn("requirements-delivery-windows.lock", manifest["source_files"])
        self.assertEqual(manifest["raw_mode"], "download")
        self.assertFalse((self.workspace / "reports/v3/dev.json").exists())
        self.assertTrue((self.workspace / "reports-delivered/v3/dev.json").is_file())
        self.assertFalse((self.workspace / "artifacts/old.pt").exists())
        self.write(self.source / "newsforge/train.py", "changed source\n")
        self.assertEqual((self.workspace / "newsforge/train.py").read_text(), "# frozen train\n")
        p.verify_workspace(self.workspace)

    def test_init_rejects_existing_or_nested_destination_before_mutation(self):
        for path in (self.source, self.source / "nested", self.base):
            with self.subTest(path=path), self.assertRaises(ValueError):
                p.initialize(self.source, path)
        self.workspace.mkdir()
        with self.assertRaises(FileExistsError):
            self.initialize()
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_raw_files_are_copied_and_tampering_is_detected(self):
        raw = self.base / "raw"
        for split in ("train", "dev"):
            for filename in ("news.tsv", "behaviors.tsv"):
                self.write(raw / split / filename, split + filename)
        manifest = self.initialize(raw)
        self.assertEqual(len(manifest["raw_files"]), 4)
        self.write(raw / "train/news.tsv", "original changed")
        p.verify_workspace(self.workspace)
        self.write(self.workspace / "data/raw/train/news.tsv", "tampered")
        with self.assertRaisesRegex(ValueError, "raw data changed"):
            p.verify_workspace(self.workspace)

    def test_missing_raw_input_does_not_create_workspace(self):
        raw = self.base / "raw"
        raw.mkdir()
        with self.assertRaises(FileNotFoundError):
            self.initialize(raw)
        self.assertFalse(self.workspace.exists())

    def test_success_receipts_bind_sources_inputs_logs_and_outputs(self):
        self.initialize()
        result = self.run_pipeline()
        self.assertEqual(result["completed_stages"], ["first", "second"])
        receipt = p.read(self.workspace / p.STATE / "second/success.json")
        self.assertEqual(receipt["inputs"], {"reports/first.json": p.digest(self.workspace / "reports/first.json")})
        self.assertEqual(receipt["workspace_manifest_sha256"], p.digest(self.workspace / p.STATE / "workspace.json"))
        with patch.object(p.subprocess, "run", side_effect=AssertionError("must not rerun")):
            self.run_pipeline(executor=None)

    def test_completed_output_or_log_tampering_prevents_resume(self):
        self.initialize()
        self.run_pipeline()
        output = self.workspace / "reports/first.json"
        original = output.read_bytes()
        output.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "output changed"):
            self.run_pipeline()
        output.write_bytes(original)
        (self.workspace / p.STATE / "first/output.log").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "log changed"):
            self.run_pipeline()

    def test_source_or_workspace_manifest_tampering_prevents_resume(self):
        self.initialize()
        self.run_pipeline()
        source = self.workspace / "newsforge/train.py"
        original = source.read_bytes()
        source.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_pipeline()
        source.write_bytes(original)
        manifest = self.workspace / p.STATE / "workspace.json"
        value = p.read(manifest)
        value["created_utc"] = "changed"
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_pipeline()

    def test_failed_command_retains_partial_artifacts_and_refuses_retry(self):
        self.initialize()
        with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
            self.run_pipeline(executor=self.execute(code=7))
        failure = p.read(self.workspace / p.STATE / "first/failed.json")
        self.assertTrue(failure["partial_outputs_retained"])
        self.assertEqual(failure["returncode"], 7)
        self.assertTrue((self.workspace / "reports/first.json").exists())
        with self.assertRaisesRegex(FileExistsError, "started/failed"):
            self.run_pipeline()

    def test_zero_exit_without_output_is_a_recorded_failure(self):
        self.initialize()
        def executor(command, **kwargs):
            return subprocess.CompletedProcess(command, 0)
        with self.assertRaises(FileNotFoundError):
            self.run_pipeline(executor=executor)
        self.assertTrue((self.workspace / p.STATE / "first/failed.json").exists())
        self.assertFalse((self.workspace / p.STATE / "first/success.json").exists())

    def test_mutating_dependency_or_source_during_command_is_failure(self):
        self.initialize()
        def mutate(name):
            if name == "second":
                self.write(self.workspace / "reports/first.json", "rewritten")
        with self.assertRaisesRegex(ValueError, "mutated dependency input"):
            self.run_pipeline(executor=self.execute(mutate=mutate))
        self.assertTrue((self.workspace / p.STATE / "second/failed.json").exists())

    def test_existing_output_rejected_without_starting_stage(self):
        self.initialize()
        self.write(self.workspace / "reports/first.json", "do not overwrite")
        with self.assertRaisesRegex(FileExistsError, "reuse/overwrite"):
            self.run_pipeline()
        self.assertFalse((self.workspace / p.STATE / "first").exists())

    def test_declared_stage_definition_change_prevents_resume(self):
        self.initialize()
        self.run_pipeline()
        changed = replace(self.a, argv=("-c", "print('changed')"))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_pipeline(definitions=(changed, self.b))

    def test_symlink_escape_is_rejected(self):
        self.initialize()
        link = self.workspace / "escape"
        try:
            link.symlink_to(self.source, target_is_directory=True)
        except OSError:
            self.skipTest("host does not permit symlink creation")
        with self.assertRaises(ValueError):
            p.inside(self.workspace, "escape/newsforge/train.py")

    def test_baseline_gate_rejects_fresh_incompatible_winner(self):
        self.initialize()
        config = {"channels": ["popular"], "pool_size": 100, "route_size": 200,
                  "rank_constant": 60, "recent_hours": 24, "freeze_feedback": False}
        selection = {"serving": {"candidate_config": config, "use_reranker": False}}
        target = self.workspace / "artifacts/v2/selection.json"
        self.write(target, json.dumps(selection))
        with current_directory(self.workspace):
            p.baseline_gate()
            self.assertTrue(p.read("reports/delivery/v3_baseline_gate.json")["passed"])
            config["channels"] = ["popular", "dense"]
            self.write(target, json.dumps(selection))
            with self.assertRaisesRegex(RuntimeError, "never force the old winner"):
                p.baseline_gate()


if __name__ == "__main__":
    unittest.main()
