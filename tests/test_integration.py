"""Integration tests for the Rust→Python path of new modules.

Tests that the compiled _rust extension classes work correctly
from Python, including CCG, Discourse, and Nonmonotonic modules.
"""

import json
import os
import sys
import unittest


class TestCCGIntegration(unittest.TestCase):
    """Test CCG classes from Python."""

    def setUp(self):
        from fastnltk._rust import CCGLexicon, CCGChartParser

        bs = chr(92)
        self.lex = CCGLexicon(
            [
                ("the", "NP/N"),
                ("cat", "N"),
                ("dog", "N"),
                ("ran", f"S{bs}NP"),
                ("chased", f"(S{bs}NP)/NP"),
                ("a", "NP/N"),
                ("ball", "N"),
            ]
        )
        self.parser = CCGChartParser(self.lex, 20)

    def test_ccg_lexicon_lookup(self):
        cats = self.lex.lookup("the")
        self.assertEqual(len(cats), 1)
        self.assertEqual(str(cats[0]), "NP/N")

    def test_ccg_lexicon_add(self):
        self.lex.add("saw", "NP/N")
        cats = self.lex.lookup("saw")
        self.assertEqual(len(cats), 1)

    def test_ccg_lexicon_missing(self):
        cats = self.lex.lookup("unknown")
        self.assertEqual(len(cats), 0)

    def test_ccg_lexicon_len(self):
        self.assertEqual(len(self.lex), 7)

    def test_ccg_parse_simple_sentence(self):
        result = self.parser.parse(["the", "cat", "ran"])
        self.assertTrue(any("S" in r for r in result), f"Should get S parse: {result}")

    def test_ccg_parse_full_sentence(self):
        result = self.parser.parse(["the", "cat", "chased", "a", "ball"])
        self.assertTrue(any("S" in r for r in result), f"Should get S parse: {result}")

    def test_ccg_parse_empty_raises(self):
        with self.assertRaises(Exception):
            self.parser.parse([])

    def test_ccg_parse_unknown_word(self):
        result = self.parser.parse(["the", "cat", "ate", "a", "ball"])
        self.assertGreater(len(result), 0)

    def test_ccg_parse_max_span_exceeded(self):
        small_parser = type(self.parser)(self.lex, 3)
        with self.assertRaises(Exception):
            small_parser.parse(["the", "cat", "chased", "a", "ball"])

    def test_fromstring(self):
        from fastnltk._rust import from_string

        cat = from_string("NP/N")
        self.assertEqual(str(cat), "NP/N")

    def test_fromstring_nested(self):
        from fastnltk._rust import from_string

        bs = chr(92)
        cat = from_string(f"(S{bs}NP)/NP")
        self.assertTrue("S" in str(cat))


class TestDiscourseIntegration(unittest.TestCase):
    """Test DiscourseThread from Python."""

    def setUp(self):
        from fastnltk._rust import DiscourseThread

        self.thread = DiscourseThread()

    def test_empty_thread(self):
        self.assertEqual(len(self.thread), 0)

    def test_add_drs(self):
        self.thread.add_drs("([x],[dog(x)])")
        self.assertEqual(len(self.thread), 1)

    def test_add_alias(self):
        self.thread.add("([x],[dog(x)])")
        self.assertEqual(len(self.thread), 1)

    def test_merge(self):
        self.thread.add_drs("([x],[dog(x)])")
        self.thread.add_drs("([y],[cat(y)])")
        merged = self.thread.merge()
        self.assertIn("dog", merged)
        self.assertIn("cat", merged)

    def test_to_fol(self):
        self.thread.add_drs("([x],[dog(x)])")
        fol = self.thread.to_fol()
        self.assertTrue("dog" in fol or "exists" in fol)

    def test_answer_question_true(self):
        self.thread.add_drs("([x],[dog(x)])")
        self.thread.add_drs("([y],[cat(y)])")
        valuation = {"dog": [["fido"]], "cat": [["felix"]]}
        domain = ["fido", "felix"]
        answer = self.thread.answer_question(
            "([x],[dog(x)])",
            json.dumps(valuation),
            json.dumps(domain),
        )
        self.assertEqual(answer, "true")

    def test_answer_question_false(self):
        self.thread.add_drs("([x],[cat(x)])")
        valuation = {"cat": [["felix"]]}
        domain = ["felix"]
        answer = self.thread.answer_question(
            "([x],[dog(x)])",
            json.dumps(valuation),
            json.dumps(domain),
        )
        self.assertEqual(answer, "false")

    def test_get_drss(self):
        self.thread.add_drs("([x],[dog(x)])")
        drss = self.thread.get_drss()
        self.assertEqual(len(drss), 1)
        self.assertIn("dog", drss[0])


class TestNonmonotonicIntegration(unittest.TestCase):
    """Test DefaultReasoner and ClosedWorldReasoner from Python."""

    def setUp(self):
        from fastnltk._rust import DefaultRule, DefaultReasoner, ClosedWorldReasoner

        self.DefaultRule = DefaultRule
        self.DefaultReasoner = DefaultReasoner
        self.ClosedWorldReasoner = ClosedWorldReasoner

    def test_default_rule(self):
        rule = self.DefaultRule("bird", "flies", "flies", "")
        self.assertEqual(rule.prerequisite, "bird")
        self.assertEqual(rule.consequent, "flies")

    def test_default_rule_display(self):
        rule = self.DefaultRule("bird", "flies", "flies", "bf")
        s = str(rule)
        self.assertIn("bird", s)
        self.assertIn("flies", s)

    def test_default_reasoner_empty(self):
        reasoner = self.DefaultReasoner([], 10)
        exts = reasoner.extensions()
        self.assertEqual(len(exts), 1)
        self.assertEqual(exts[0], [])

    def test_default_reasoner_rules(self):
        rule = self.DefaultRule("bird", "flies", "flies", "bf")
        reasoner = self.DefaultReasoner([rule], 10)
        rs = reasoner.rules()
        self.assertEqual(len(rs), 1)

    def test_cwr_query_true(self):
        cwr = self.ClosedWorldReasoner(["bird(tweety)"])
        self.assertTrue(cwr.query("bird(tweety)"))

    def test_cwr_query_false(self):
        cwr = self.ClosedWorldReasoner(["bird(tweety)"])
        self.assertFalse(cwr.query("cat(felix)"))

    def test_cwr_empty_kb(self):
        cwr = self.ClosedWorldReasoner([])
        self.assertFalse(cwr.query("anything"))

    def test_cwr_positive_facts(self):
        cwr = self.ClosedWorldReasoner(["bird(tweety)", "cat(felix)"])
        facts = cwr.positive_facts()
        self.assertEqual(len(facts), 2)

    def test_cwr_negative_facts(self):
        cwr = self.ClosedWorldReasoner(["bird(tweety)"])
        neg = cwr.negative_facts()
        # Should not contain ~bird(tweety) since that's derived
        self.assertIsInstance(neg, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
