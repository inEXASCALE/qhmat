"""Guard the block terminology and the first-use quantization budget definition."""
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def active_source(name):
    text = (ROOT / name).read_text()
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", text, flags=re.S)
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


class ManuscriptTerminologyTests(unittest.TestCase):
    def test_blocks_are_not_called_leaves(self):
        for name in ("main.tex", "ex_supplement.tex"):
            with self.subTest(name=name):
                text = active_source(name)
                # Internal reference keys and archived filenames are not prose.
                text = re.sub(r"\\(?:label|ref|eqref|texttt)\{[^}]*\}", "", text)
                # These are verbs, not names for blocks.
                text = text.replace(r"QD^{-1}$ leaves the", "")
                text = text.replace("bit widths leaves storage", "")
                self.assertNotRegex(text, r"\b(?:leaf|leafwise|leaves)\b")

    def test_budget_is_defined_before_intro_bound(self):
        text = active_source("main.tex")
        abstract = text.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
        self.assertIn("max-norm quantization error budget", abstract)
        intro = text.split(r"\section{Introduction}", 1)[1].split(r"\section{", 1)[0]
        intro = re.sub(r"\s+", " ", intro)
        definition = (r"Let $B^{\mathrm q}\ge0$ denote the prescribed absolute max-norm "
                      "error budget for factor quantization")
        self.assertIn(definition, intro)
        self.assertLess(intro.index(definition), intro.index(r"E_{\mathrm{cert}}\le B^{\mathrm q}"))
        self.assertIn(r"\eqref{eq:quant-budget}", intro)

    def test_partition_and_diagonal_only_policy_remain_distinct(self):
        text = re.sub(r"\s+", " ", active_source("main.tex"))
        self.assertIn("non-diagonal inadmissible blocks use quantized low-rank factors", text)
        self.assertIn("only diagonal blocks remain dense", text)
        self.assertIn("Admissible blocks may terminate subdivision above the terminal cluster level", text)
        self.assertIn("not a synonym for non-diagonality or geometric inadmissibility", text)
        self.assertIn("the blocks form the disjoint partition", text)
        self.assertIn(r"\mathop{\dot\bigcup}_{k\in\mathcal L}", text)


if __name__ == "__main__":
    unittest.main()
