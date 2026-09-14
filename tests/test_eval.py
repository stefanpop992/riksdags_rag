"""Evaluation tests use scripted retrieval and API responses; no paid calls."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import eval as evaluation
import rag


def settings(**kwargs):
    return evaluation.korning_settings(argparse.Namespace(**kwargs))


def source():
    return {"text": "Vi föreslår mer vindkraft.", "likhet": 0.9,
            "meta": {"talare": "Anna (S)", "parti": "S", "datum": "2024-01-01",
                     "anforande_id": "a", "chunk_nr": 0, "debattrubrik": "Energi",
                     "url": "https://example.org/speech"}}


def verdict():
    return evaluation.Granskning(pastaenden=[evaluation.Pastaende(
        pastaende="Anna föreslår mer vindkraft", stods_av_kallorna=True,
        motivering="Står i utdraget")], avstod_helt=False)


def response(parsed=None, input_tokens=10, output_tokens=5):
    return SimpleNamespace(model="test-model", parsed_output=parsed,
                           usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                                                 cache_read_input_tokens=2, cache_creation_input_tokens=3))


class FakeStream:
    def __iter__(self):
        yield SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(
            type="text_delta", text="Anna föreslår mer vindkraft [Anna (S), 2024-01-01]."))

    def get_final_message(self):
        return response(input_tokens=20, output_tokens=8)


def client():
    @contextlib.contextmanager
    def stream(**kwargs):
        yield FakeStream()
    return SimpleNamespace(messages=SimpleNamespace(
        stream=Mock(side_effect=stream), parse=Mock(return_value=response(verdict()))))


def run_fixture(**kwargs):
    with patch.object(evaluation, "kodversion", return_value={"commit": "abc", "dirty": False, "sha256": {}}):
        run = evaluation.ny_korning(settings(limit=1, **kwargs))
    return run


class RoutingTests(unittest.TestCase):
    def test_all_four_configurations_use_the_correct_retrieval(self):
        index = object()
        for config in evaluation.CONFIGS:
            with self.subTest(config=config), \
                    patch.object(rag, "sok", return_value=[source()]) as vector, \
                    patch.object(rag, "sok_hybrid", return_value=[source()]) as hybrid, \
                    patch.object(rag, "agentisk_sokning", return_value=([source()], [], [])) as deep, \
                    patch.object(rag, "stromma_svar", return_value=iter(["Svar"])) as answer:
                opts = settings(answer_model="answer", planning_model="planner", per_speech=2, count=5)
                row = evaluation.kor_en(None, "db", "model", [], "fråga", config, opts, index)
                self.assertEqual(row["kallor"], [source()])
                self.assertIn(source()["text"], row["underlag"])
                answer.assert_called_once_with(None, "fråga", row["underlag"], modell="answer")
                if config.startswith("deep"):
                    deep.assert_called_once()
                    self.assertIs(deep.call_args.kwargs["bm25_index"], index if config.endswith("hybrid") else None)
                    self.assertEqual(deep.call_args.kwargs["planeringsmodell"], "planner")
                    self.assertEqual(deep.call_args.kwargs["per_anforande"], 2)
                    vector.assert_not_called()
                    hybrid.assert_not_called()
                else:
                    selected = hybrid if config.endswith("hybrid") else vector
                    selected.assert_called_once()
                    self.assertEqual(selected.call_args.kwargs, {"antal": 5, "per_anforande": 2})
                    (vector if config.endswith("hybrid") else hybrid).assert_not_called()
                    deep.assert_not_called()

    def test_hybrid_never_silently_falls_back(self):
        for config in ("quick-hybrid", "deep-hybrid"):
            with self.subTest(config=config), patch.object(rag, "sok") as search:
                with self.assertRaisesRegex(ValueError, "BM25"):
                    evaluation.kor_en(None, None, None, [], "fråga", config, settings())
                search.assert_not_called()

    def test_empty_results_skip_answer_and_judge_calls(self):
        run = run_fixture(configs=["quick-vector"])
        api = client()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(rag, "sok", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            evaluation.kor_jobb(run, directory, api, None, None, [], None)
            row = next(iter(evaluation.las_korning(directory)["results"].values()))
        self.assertEqual(row["status"], "complete")
        self.assertTrue(row["avstod_helt"])
        self.assertIsNone(row["citatprecision"])
        self.assertIsNone(row["groundedness"])
        self.assertEqual(row["api_anrop"], 0)
        api.messages.stream.assert_not_called()
        api.messages.parse.assert_not_called()


class CheckpointTests(unittest.TestCase):
    def test_judge_failure_resumes_without_generating_answer_again(self):
        run = run_fixture(configs=["quick-vector"])
        api = client()
        api.messages.parse.side_effect = [RuntimeError("temporary failure"), response(verdict())]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(rag, "sok", return_value=[source()]) as retrieve, \
                contextlib.redirect_stdout(io.StringIO()):
            evaluation.kor_jobb(run, directory, api, None, None, [], None)
            resumed = evaluation.las_korning(directory)
            row = next(iter(resumed["results"].values()))
            self.assertEqual(row["status"], "answer_ready")
            self.assertIn(source()["text"], row["underlag"])
            self.assertEqual(len(resumed["errors"]), 1)
            evaluation.kor_jobb(resumed, directory, api, None, None, [], None)
            final = evaluation.las_korning(directory)
            self.assertEqual(final["errors"], {})
            self.assertEqual(len(final["failed_attempts"]), 1)
            row = next(iter(final["results"].values()))
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["citatprecision"], 1)
            self.assertEqual(row["usage"][0]["input_tokens"], 20)
            self.assertEqual(row["judge_usage"][0]["input_tokens"], 10)
            self.assertEqual(row["api_anrop"], 1)
            self.assertEqual(row["judge_api_anrop"], 1)
            evaluation.kor_jobb(final, directory, api, None, None, [], None)
            retrieve.assert_called_once()
            api.messages.stream.assert_called_once()
            self.assertEqual(api.messages.parse.call_count, 2)

    def test_failed_generation_remains_pending(self):
        run = run_fixture(configs=["quick-vector"])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(rag, "sok", side_effect=RuntimeError("search failed")), \
                contextlib.redirect_stdout(io.StringIO()):
            evaluation.kor_jobb(run, directory, client(), None, None, [], None)
            saved = evaluation.las_korning(directory)
        self.assertEqual(saved["results"], {})
        self.assertEqual(next(iter(saved["errors"].values()))["phase"], "generation")
        self.assertIn("Ofullständig/obalanserad", evaluation.bygg_rapport(saved))

    def test_atomic_save_failure_preserves_previous_results(self):
        run = run_fixture()
        with tempfile.TemporaryDirectory() as directory:
            evaluation.spara_korning(run, directory)
            original = (Path(directory) / "results.json").read_bytes()
            run["errors"] = {"changed": {}}
            with patch.object(Path, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    evaluation.spara_korning(run, directory)
            self.assertEqual((Path(directory) / "results.json").read_bytes(), original)

    def test_settings_resume_defaults_and_reject_changed_options(self):
        saved = settings(configs=["deep-hybrid"], limit=2, max_varv=2)
        self.assertEqual(evaluation.korning_settings(argparse.Namespace(), saved), saved)
        with self.assertRaisesRegex(ValueError, "Inställningarna"):
            evaluation.korning_settings(argparse.Namespace(max_varv=3), saved)

    def test_question_config_keys_are_distinct(self):
        run = run_fixture()
        self.assertEqual(len(evaluation.jobb_for(run)), 4)
        self.assertEqual(len({evaluation.jobbnyckel(q, c) for _, q, c in evaluation.jobb_for(run)}), 4)


class CommandAndReportTests(unittest.TestCase):
    def test_existing_run_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(rag, "skapa_klient") as api, contextlib.redirect_stderr(io.StringIO()):
            marker = Path(directory) / "results.json"
            marker.write_text("existing results")
            with self.assertRaises(SystemExit) as error:
                evaluation.main(["--run-dir", directory])
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(marker.read_text(), "existing results")
            api.assert_not_called()

    def test_resume_rejects_changed_code_before_api_initialization(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(rag, "skapa_klient") as api, contextlib.redirect_stderr(io.StringIO()):
            evaluation.spara_korning(run_fixture(), directory)
            with self.assertRaises(SystemExit):
                evaluation.main(["--resume", "--run-dir", directory])
            api.assert_not_called()

    def test_report_only_needs_saved_results(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(rag, "skapa_klient") as api, \
                patch.object(rag, "ladda_modell") as model, \
                patch.object(bm25 := evaluation.bm25, "ladda") as index, \
                contextlib.redirect_stdout(io.StringIO()):
            evaluation.spara_korning(run_fixture(), directory)
            self.assertEqual(evaluation.main(["--rapport", "--run-dir", directory]), 0)
            report = (Path(directory) / "report.md").read_text()
            for config in evaluation.CONFIGS:
                self.assertIn(f"| {config} | 0 |", report)
            api.assert_not_called()
            model.assert_not_called()
            index.assert_not_called()

    def test_four_configurations_have_separate_report_rows(self):
        run = run_fixture()
        api = client()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(rag, "sok", return_value=[source()]), \
                patch.object(rag, "sok_hybrid", return_value=[source()]), \
                patch.object(rag, "agentisk_sokning", return_value=([source()], [], [])), \
                contextlib.redirect_stdout(io.StringIO()):
            evaluation.kor_jobb(run, directory, api, None, None, [], object())
        report = evaluation.bygg_rapport(run)
        self.assertIn("Slutförda: 4/4", report)
        self.assertNotIn("Ofullständig/obalanserad", report)
        for config in evaluation.CONFIGS:
            self.assertIn(f"| {config} | 1 | 100 % | 100 % |", report)
        self.assertEqual(len(run["results"]), 4)

    def test_legacy_report_does_not_change_historical_files(self):
        paths = [evaluation.ROOT / "eval_resultat.json", evaluation.ROOT / "eval_resultat.md"]
        before = [p.read_bytes() for p in paths]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(evaluation.main(["--rapport"]), 0)
        self.assertIn("quick-vector", output.getvalue())
        self.assertIn("deep-vector", output.getvalue())
        self.assertEqual([p.read_bytes() for p in paths], before)

    def test_invalid_limits_fail_before_loading_resources(self):
        for argv in (["--limit", "0"], ["--max-varv", "0"], ["--resume"]):
            with self.subTest(argv=argv), patch.object(rag, "ladda_modell") as model, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    evaluation.main(argv)
                model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
