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

if __name__ == "__main__":
    unittest.main()
