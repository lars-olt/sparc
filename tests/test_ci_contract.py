import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SETUP_UV_ACTION = (
    'astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9'
)


# Keep the workflow on an action reference that GitHub can actually resolve.
class ContinuousIntegrationContractTests(unittest.TestCase):
    def test_uv_action_uses_an_immutable_release(self):
        workflow = (ROOT / '.github' / 'workflows' / 'ci.yml').read_text(
            encoding='utf-8',
        )

        self.assertIn(f'uses: {SETUP_UV_ACTION}', workflow)
        self.assertNotIn('astral-sh/setup-uv@v', workflow)


if __name__ == '__main__':
    unittest.main()
