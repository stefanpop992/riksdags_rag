import unittest

from index_data import batcha_hela_anforanden, dela_i_chunks


class IndexingTests(unittest.TestCase):
    def test_overlap_cannot_push_next_paragraph_over_chunk_limit(self):
        paragraphs = [" ".join(f"first{i}" for i in range(50)),
                      " ".join(f"second{i}" for i in range(240))]
        chunks = dela_i_chunks(paragraphs, max_ord=250, overlapp_ord=60)
        self.assertTrue(all(len(chunk.split()) <= 250 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), " ".join(paragraphs).split())

    def test_overlap_is_preserved_when_it_fits(self):
        paragraphs = ["ett " * 200, "två " * 40, "tre " * 30]
        chunks = dela_i_chunks(paragraphs, max_ord=250, overlapp_ord=60)
        self.assertEqual([len(c.split()) for c in chunks], [240, 70])
        self.assertEqual(chunks[1].split()[:40], ["två"] * 40)

    def test_long_sentence_is_split_without_losing_words(self):
        words = [f"ord{i}" for i in range(701)]
        chunks = dela_i_chunks([" ".join(words)], max_ord=250)
        self.assertEqual([word for c in chunks for word in c.split()], words)
        self.assertTrue(all(len(c.split()) <= 250 for c in chunks))

    def test_batches_keep_each_speech_whole_for_resume(self):
        speeches = [["a0", "a1"], ["b0", "b1", "b2", "b3"], ["c0"]]
        self.assertEqual(list(batcha_hela_anforanden(speeches, max_storlek=3)), speeches)


if __name__ == "__main__":
    unittest.main()
