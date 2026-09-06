import unittest
from unittest.mock import MagicMock, patch

import server


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.rag = MagicMock()
        self.rag.talarvarianter.return_value = []
        self.rag.formatera_talare.return_value = 'Test (S)'
        self.hits = [{'meta': {'datum': '2025-01-01', 'debattrubrik': 'Debatt'}, 'text': 'Utdrag'}]
        self.rag.sok.return_value = self.hits
        self.rag.sok_hybrid.return_value = self.hits
        self.rag.agentisk_sokning.return_value = (self.hits, [], [])
        self.rag.stromma_svar.return_value = iter(['Ett ', 'svar.'])
        self.events = []
        self.data = server.validate({'question': ' Test? ', 'mode': 'quick'})

    def run_search(self, index=None):
        with patch.object(server, 'resources', return_value=(self.rag, 'db', 'client', 'model', [], index, {})):
            server.search(self.data, lambda kind, value: self.events.append((kind, value)))

    def test_quick_streams_sources_then_answer(self):
        self.run_search()
        self.rag.sok.assert_called_once()
        self.assertEqual([e[1] for e in self.events if e[0] == 'token'], ['Ett ', 'svar.'])
        self.assertEqual(self.events[-1], ('done', None))
        self.assertLess([e[0] for e in self.events].index('sources'), [e[0] for e in self.events].index('token'))

    def test_hybrid_and_filters(self):
        self.data.update(party='S', from_year=2020, to_year=2025)
        self.run_search('index')
        self.rag.bygg_filter.assert_called_once_with('S', 2020, 2025, [])
        self.rag.sok_hybrid.assert_called_once()

    def test_deep_mode(self):
        self.data['mode'] = 'deep'
        self.run_search()
        self.rag.agentisk_sokning.assert_called_once()
        self.rag.bygg_underlag_med_luckor.assert_called_once_with(self.hits, [])

    def test_empty_results_do_not_generate_answer(self):
        self.rag.sok.return_value = []
        self.run_search()
        self.rag.stromma_svar.assert_not_called()
        self.assertIn(('sources', []), self.events)

    def test_unknown_speaker(self):
        self.data['speaker'] = 'Unknown'
        with self.assertRaises(ValueError):
            self.run_search()
        self.rag.sok.assert_not_called()

    def test_invalid_inputs(self):
        for update in [{'question': ' '}, {'count': 99}, {'from_year': 2025, 'to_year': 2020},
                       {'mode': 'invalid'}, {'speaker': []}, {'hybrid': 'yes'}, {'party': 'invalid'}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                server.validate(dict(self.data, **update))


if __name__ == '__main__':
    unittest.main()
