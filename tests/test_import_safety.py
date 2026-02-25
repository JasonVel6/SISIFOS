import subprocess
import sys

class TestImportSafety:
    def test_config_import_does_not_load_native_binaries(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "from modules.config import SceneConfig; "
             "assert 'trimesh' not in sys.modules; "
             "print('OK')"], capture_output=True, text=True
        )
        assert result.returncode == 0