"""Offline checks: real Chroma/BM25, fixed vectors, scripted LLM decisions."""
import contextlib
import io
import unittest
from uuid import uuid4
from unittest.mock import Mock, patch

import chromadb
import numpy as np
from chromadb.config import Settings

import bm25
import rag


def hit(speech, chunk=0, score=0.9):
    return {"text": "Ett källutdrag", "likhet": score,
            "meta": {"anforande_id": speech, "chunk_nr": chunk,
                     "talare": "Anna (S)", "parti": "S", "ar": 2024,
                     "datum": "2024-01-01", "debattrubrik": "Energi"}}


def question(speaker=None, party=None):
    return rag.Delfraga(sokfraga="energi", parti=party, talare=speaker,
                       motivering="Hämta underlag")


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.plan = patch.object(rag, "planera", return_value=[question()]).start()
        self.search = patch.object(rag, "sok", return_value=[]).start()
        self.review = patch.object(rag, "bedom", return_value=rag.Bedomning(
            racker_underlaget=True, saknas=[], nya_sokningar=[])).start()
        self.addCleanup(patch.stopall)

    def run_agent(self, **kwargs):
        return rag.agentisk_sokning(None, None, None, "Vad sägs om energi?",
                                    alla_talare=["Anna (S)"], **kwargs)

    def test_unknown_planned_speaker_never_broadens_search(self):
        self.plan.return_value = [question("Saknad Person")]
        hits, missing, _ = self.run_agent()
        self.search.assert_not_called()
        self.assertEqual(hits, [])
        self.assertTrue(any("Saknad Person" in gap for gap in missing))

    def test_unknown_user_speaker_fails_before_planning(self):
        with self.assertRaisesRegex(ValueError, "Ingen talare"):
            self.run_agent(anvandar_talare="Saknad Person")
        self.plan.assert_not_called()
        self.search.assert_not_called()

    def test_user_filters_override_planned_filters(self):
        self.plan.return_value = [question("Saknad Person", "M")]
        self.run_agent(anvandar_talare="Anna", anvandar_parti="S",
                       fran_ar=2023, till_ar=2024)
        where = self.search.call_args.args[4]
        self.assertTrue(rag.uppfyller(hit("a")["meta"], where))
        self.assertFalse(rag.uppfyller(dict(hit("a")["meta"], parti="M"), where))
        self.assertFalse(rag.uppfyller(dict(hit("a")["meta"], ar=2022), where))

    def test_excerpt_limit_applies_across_subqueries(self):
        self.plan.return_value = [question(), question()]
        self.search.side_effect = [[hit("a", 0, 0.8)],
                                   [hit("a", 1, 0.95), hit("b", 0, 0.7)]]
        hits, _, _ = self.run_agent(per_anforande=1)
        self.assertEqual([(h["meta"]["anforande_id"], h["meta"]["chunk_nr"])
                          for h in hits], [("a", 1), ("b", 0)])
        self.assertEqual(self.review.call_args.args[3], hits)

    def test_duplicate_chunks_are_returned_once(self):
        self.plan.return_value = [question(), question()]
        self.search.return_value = [hit("a")]
        hits, _, _ = self.run_agent()
        self.assertEqual(len(hits), 1)

    def test_unknown_speaker_gap_survives_later_review(self):
        self.plan.return_value = [question("Saknad Person")]
        self.review.side_effect = [
            rag.Bedomning(racker_underlaget=False, saknas=[], nya_sokningar=[question("Anna")]),
            rag.Bedomning(racker_underlaget=True, saknas=[], nya_sokningar=[]),
        ]
        self.search.return_value = [hit("a")]
        hits, missing, _ = self.run_agent()
        self.assertEqual(len(hits), 1)
        self.search.assert_called_once()
        self.assertTrue(any("Saknad Person" in gap for gap in missing))

    def test_hybrid_selection_uses_rrf_and_respects_both_limits(self):
        self.plan.return_value = [question(), question()]
        candidates = [dict(hit("a", 0, 0.99), rrf=0.01),
                      dict(hit("a", 1, 0.8), rrf=0.04),
                      dict(hit("a", 2, 0.7), rrf=0.03),
                      dict(hit("b", 0, 0.6), rrf=0.02),
                      dict(hit("c", 0, 0.5), rrf=0.005)]
        with patch.object(rag, "sok_hybrid", side_effect=[candidates[:2], candidates[2:]]):
            hits, _, _ = self.run_agent(bm25_index=object(), per_anforande=2, max_utdrag=3)
        self.assertEqual([(h["meta"]["anforande_id"], h["meta"]["chunk_nr"])
                          for h in hits], [("a", 1), ("a", 2), ("b", 0)])
        self.search.assert_not_called()

    def test_review_cannot_exceed_round_limit(self):
        self.review.return_value = rag.Bedomning(
            racker_underlaget=False, saknas=["Mer underlag behövs"],
            nya_sokningar=[question()])
        _, missing, trace = self.run_agent(max_varv=2)
        self.assertEqual(self.search.call_count, 2)
        self.assertEqual(self.review.call_count, 2)
        self.assertEqual(missing, ["Mer underlag behövs"])
        self.assertEqual(trace[-1]["typ"], "stopp")


class HybridIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        cls.collection = cls.client.create_collection(
            "test-" + uuid4().hex, metadata={"hnsw:space": "cosine"},
            embedding_function=None)
        cls.ids = ["a:0", "a:1", "b:0", "c:0"]
        documents = ["energi kärnkraft", "energi elpriser", "energi vindkraft", "energi skatter"]
        metas = [hit("a")["meta"], hit("a", 1)["meta"],
                 hit("b")["meta"], dict(hit("c")["meta"], parti="M", ar=2022)]
        cls.collection.add(ids=cls.ids, documents=documents, metadatas=metas,
                           embeddings=[[1., 0.], [0.99, 0.1], [0.8, 0.6], [0., 1.]])
        # Include a stale BM25 ID: it must not leak into the merged results.
        with contextlib.redirect_stdout(io.StringIO()):
            matrix, vocabulary = bm25.bygg(documents + ["energi gammalt dokument"])
        cls.index = bm25.Bm25(matrix, vocabulary, cls.ids + ["deleted:0"])

    @classmethod
    def tearDownClass(cls):
        cls.client.delete_collection(cls.collection.name)

    def setUp(self):
        self.model = Mock()
        self.model.encode.return_value = np.array([[1., 0.]])

    def test_hybrid_respects_filters_and_speech_diversity(self):
        where = rag.bygg_filter("S", 2024, 2024, ["Anna (S)"])
        hits = rag.sok_hybrid(self.collection, self.model, "energi", self.index,
                              antal=3, where=where, per_anforande=1)
        self.assertEqual({h["meta"]["anforande_id"] for h in hits}, {"a", "b"})
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(rag.uppfyller(h["meta"], where) for h in hits))
        self.model.encode.assert_called_once_with(["query: energi"], normalize_embeddings=True)

    def test_no_matching_filter_returns_no_hits_in_both_modes(self):
        where = rag.bygg_filter("V")
        self.assertEqual(rag.sok(self.collection, self.model, "energi", where=where), [])
        self.assertEqual(rag.sok_hybrid(self.collection, self.model, "energi",
                                      self.index, where=where), [])

    def test_unknown_bm25_words_fall_back_to_vector_results(self):
        vector = rag.sok(self.collection, self.model, "xyzunknown", antal=3)
        hybrid = rag.sok_hybrid(self.collection, self.model, "xyzunknown",
                                self.index, antal=3)
        self.assertEqual([h["meta"] for h in vector], [h["meta"] for h in hybrid])


if __name__ == "__main__":
    unittest.main()
