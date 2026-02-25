import numpy as np
from modules.trajectory.motion_cases import detect_degeneracy, repair_hill

class TestDegeneracyAndRepair:
    def test_radial_velocity_is_degenerate(self):
        r0, v0 = np.array([30.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        bad, _ = detect_degeneracy(r0, v0, f_px=1000.0, dt=1.0, px_min=1.0, rho_max=0.9)
        assert bad

    def test_repair_hill_boosts_velocity(self):
        r0, v0 = np.array([10.0, 0.0, 0.0]), np.array([0.0, 1e-6, 0.0])
        fix = repair_hill(r0, v0, 0.001, f_px=1000.0, dt=1.0, px_min=3.0)
        assert fix is not None
        assert abs(fix["vy0"]) > abs(v0[1])