import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SETUP_UV_ACTION = (
    'astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9'
)


# Keep the workflow on an action reference that GitHub can actually resolve.
class ContinuousIntegrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (ROOT / '.github' / 'workflows' / 'ci.yml').read_text(
            encoding='utf-8',
        )
        cls.project = tomllib.loads(
            (ROOT / 'pyproject.toml').read_text(encoding='utf-8'),
        )

    def test_uv_action_uses_an_immutable_release(self):
        self.assertIn(f'uses: {SETUP_UV_ACTION}', self.workflow)
        self.assertNotIn('astral-sh/setup-uv@v', self.workflow)

    def test_public_git_dependencies_do_not_require_an_ssh_secret(self):
        dependencies = self.project['project']['dependencies']
        million_concepts = [
            dependency
            for dependency in dependencies
            if 'github.com/MillionConcepts/' in dependency
        ]

        self.assertEqual(5, len(million_concepts))
        for dependency in million_concepts:
            self.assertIn(
                '@ git+https://github.com/MillionConcepts/',
                dependency,
            )
            self.assertNotIn('git+ssh://', dependency)

        self.assertNotIn('SSH_PRIVATE_KEY', self.workflow)
        self.assertNotIn('webfactory/ssh-agent', self.workflow)


if __name__ == '__main__':
    unittest.main()
