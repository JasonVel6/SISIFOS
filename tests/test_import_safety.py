import os
import subprocess
import sys


class TestImportSafety:
    def test_config_import_does_not_load_native_binaries(self):
        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from modules.config import SceneConfig; "
                "assert 'trimesh' not in sys.modules; "
                "print('OK')",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
