from src.prompts import extract_code_block
import unittest

class CodeExtractTests(unittest.TestCase):
    def test_preserves_relative_indent(self):
        text = "```python\n    def f(x):\n        return x + 1\n    print(f(2))\n```"
        code = extract_code_block(text)
        self.assertIsNotNone(code)
        self.assertIn("def f(x):", code)
        self.assertIn("    return x + 1", code)
        # first line should not keep residual outer indent after sanitize
        self.assertTrue(code.splitlines()[0].startswith("def "))

    def test_rejects_latex_equations(self):
        latex = r"""
\[ P(x) = 2x^2 + ax + b \]
\[ 54 = 2(16)^2 + 16a + b \]
"""
        self.assertIsNone(extract_code_block(latex))

if __name__ == "__main__":
    unittest.main()
