from __future__ import annotations

import unittest

from src.cost_tracker import CostTracker


class CostTrackerTests(unittest.TestCase):
    def test_fresh_tracker_starts_at_zero_with_empty_events(self) -> None:
        tracker = CostTracker()
        self.assertEqual(tracker.total_units, 0)
        self.assertEqual(tracker.events, [])

    def test_record_single_event(self) -> None:
        tracker = CostTracker()
        tracker.record('inference', 42)
        self.assertEqual(tracker.total_units, 42)
        self.assertEqual(len(tracker.events), 1)

    def test_record_multiple_events_accumulates_totals(self) -> None:
        tracker = CostTracker()
        tracker.record('inference', 10)
        tracker.record('embedding', 20)
        tracker.record('search', 30)
        self.assertEqual(tracker.total_units, 60)
        self.assertEqual(len(tracker.events), 3)

    def test_record_zero_units(self) -> None:
        tracker = CostTracker()
        tracker.record('noop', 0)
        self.assertEqual(tracker.total_units, 0)
        self.assertEqual(len(tracker.events), 1)
        self.assertIn('noop:0', tracker.events)

    def test_record_large_units(self) -> None:
        tracker = CostTracker()
        large = 10**9
        tracker.record('bulk', large)
        self.assertEqual(tracker.total_units, large)
        self.assertEqual(tracker.events, [f'bulk:{large}'])

    def test_event_format_is_label_colon_units(self) -> None:
        tracker = CostTracker()
        tracker.record('inference', 42)
        self.assertEqual(tracker.events[0], 'inference:42')

    def test_events_are_ordered_chronologically(self) -> None:
        tracker = CostTracker()
        labels = ['first', 'second', 'third']
        for i, label in enumerate(labels):
            tracker.record(label, i)
        self.assertEqual(tracker.events, ['first:0', 'second:1', 'third:2'])
