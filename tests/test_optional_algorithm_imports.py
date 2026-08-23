"""Regression tests for SPARC's lightweight installation boundary."""

from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib
import unittest


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
ALGORITHM_MODULES = (
    "kneed",
    "psutil",
    "segment_anything",
    "sklearn",
    "torch",
    "torchvision",
)


def run_with_algorithm_imports_blocked(code: str) -> subprocess.CompletedProcess[str]:
    """Run code in a clean interpreter where algorithm packages cannot import."""
    blocked = repr(ALGORITHM_MODULES)
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        BLOCKED = {blocked}

        class BlockAlgorithmImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(
                    fullname == name or fullname.startswith(name + ".")
                    for name in BLOCKED
                ):
                    error = ModuleNotFoundError(
                        f"blocked optional algorithm import: {{fullname}}"
                    )
                    error.name = fullname
                    raise error
                return None

        sys.meta_path.insert(0, BlockAlgorithmImports())
        sys.path.insert(0, {str(SRC)!r})

        {textwrap.indent(textwrap.dedent(code), '        ').lstrip()}
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


# The base package should work without silently pulling in the algorithm stack.
class OptionalAlgorithmImportTests(unittest.TestCase):
    def assert_script_succeeds(self, code: str) -> None:
        completed = run_with_algorithm_imports_blocked(code)
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_lightweight_public_apis_do_not_import_algorithm_stack(self):
        # This blocks every heavy package and then imports the real public APIs.
        self.assert_script_succeeds(
            """
            import sys
            import sparc

            from sparc import SparcConfig, SparcState
            from sparc.core.constants import get_instrument_config
            from sparc.data.loading import create_rgb_stretch
            from sparc.spectral.metrics import compute_roi_spectra
            from sparc.utils.geometry import right_rect_to_left_inscribed
            from sparc.utils.sel_writer import export_sel
            from sparc.visualization.plotting import plot_spectra_with_error

            assert SparcConfig
            assert SparcState
            assert get_instrument_config
            assert create_rgb_stretch
            assert compute_roi_spectra
            assert right_rect_to_left_inscribed
            assert export_sel
            assert plot_spectra_with_error
            assert not any(
                loaded == blocked or loaded.startswith(blocked + ".")
                for loaded in sys.modules
                for blocked in BLOCKED
            )
            """
        )

    def test_algorithm_api_explains_how_to_install_the_extra(self):
        # Asking for the algorithm should fail with an instruction the user can act on.
        self.assert_script_succeeds(
            """
            import sparc

            try:
                sparc.Sparc
            except ModuleNotFoundError as error:
                assert "sparc[algorithm]" in str(error), str(error)
            else:
                raise AssertionError("algorithm API imported without its dependencies")
            """
        )

    def test_algorithm_dependencies_are_only_in_the_algorithm_extra(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        base = metadata["project"]["dependencies"]
        algorithm = metadata["project"]["optional-dependencies"]["algorithm"]

        base_names = {
            dependency.split("@")[0].split("[")[0].split("<")[0]
            .split(">=")[0].strip().lower()
            for dependency in base
        }
        algorithm_names = {
            dependency.split("@")[0].split("[")[0].split("<")[0]
            .split(">=")[0].strip().lower()
            for dependency in algorithm
        }

        expected = {
            "scikit-learn",
            "kneed",
            "psutil",
            "torch",
            "torchvision",
            "segment-anything",
        }
        self.assertTrue(expected.isdisjoint(base_names))
        self.assertTrue(expected.issubset(algorithm_names))


if __name__ == "__main__":
    unittest.main()
