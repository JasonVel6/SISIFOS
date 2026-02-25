import sys
import os
import pytest


def main():
    # Ensure project root is on sys.path so `modules` can be imported.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)

    args = ["-v", "tests/"]
    # Run pytest in-process so Blender's `bpy` and `mathutils` are available.
    raise SystemExit(pytest.main(args))


if __name__ == "__main__":
    main()
