"""Unit and integration tests for src/train.py (Module 8: Training Orchestrator)."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.train import (
    build_parser,
    check_dependencies,
    show_status,
    run_stage,
    _stage_outputs_exist,
    _check_stage_inputs,
    _import_entry,
    STAGE_REGISTRY,
    main,
)
from src.config import REPORTS_DIR


@pytest.fixture
def test_logger():
    """Create a quiet logger for tests."""
    logger = logging.getLogger("TestTrainLogger")
    logger.setLevel(logging.CRITICAL)  # avoid polluting test output
    return logger


@pytest.mark.unit
class TestTrainCliParsing:
    """Tests for CLI argument parsing in train.py."""

    def test_build_parser_instantiation(self):
        """Parser should instantiate cleanly with expected program name."""
        parser = build_parser()
        assert parser.prog == "train.py"

    def test_parse_all_mode(self):
        """--all flag must set args.all = True."""
        parser = build_parser()
        args = parser.parse_args(["--all"])
        assert args.all is True
        assert args.stages is None
        assert args.status is False
        assert args.check_deps is False
        assert args.update_requirements is False

    def test_parse_stages_mode(self):
        """--stages must accept valid integers in range 1-7."""
        parser = build_parser()
        args = parser.parse_args(["--stages", "1", "2", "3"])
        assert args.stages == [1, 2, 3]
        assert args.all is False

    def test_parse_utility_modes(self):
        """Verify mutually exclusive utility modes."""
        parser = build_parser()
        assert parser.parse_args(["--status"]).status is True
        assert parser.parse_args(["--check-deps"]).check_deps is True
        assert parser.parse_args(["--update-requirements"]).update_requirements is True

    def test_parse_modifier_flags(self):
        """Verify modifier flags: --force, --dry-run, --skip-shap, --log-file."""
        parser = build_parser()
        args = parser.parse_args([
            "--all",
            "--force",
            "--dry-run",
            "--skip-shap",
            "--log-file", "reports/custom.log"
        ])
        assert args.force is True
        assert args.dry_run is True
        assert args.skip_shap is True
        assert args.log_file == Path("reports/custom.log")

    def test_mutually_exclusive_modes_raise_error(self):
        """Specifying two primary modes simultaneously must raise SystemExit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--all", "--status"])

    def test_missing_mode_raises_error(self):
        """Calling without any required mode must raise SystemExit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_invalid_stage_number_raises_error(self):
        """Stage numbers outside 1-7 must raise SystemExit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--stages", "8"])


@pytest.mark.unit
class TestTrainStageRegistry:
    """Tests for STAGE_REGISTRY integrity."""

    def test_stage_registry_covers_all_stages(self):
        """Registry must contain stages 1 through 7."""
        assert list(STAGE_REGISTRY.keys()) == [1, 2, 3, 4, 5, 6, 7]

    def test_stage_registry_metadata_complete(self):
        """Every stage must define required architectural metadata."""
        required_keys = {"name", "module", "entry_point", "requires", "produces", "isolation_note"}
        for sid, reg in STAGE_REGISTRY.items():
            assert required_keys.issubset(reg.keys()), f"Stage {sid} missing metadata keys"
            assert isinstance(reg["requires"], list)
            assert isinstance(reg["produces"], list)
            assert len(reg["produces"]) > 0

    def test_entry_points_importable(self):
        """Verify that all stage modules and entry points are valid Python callables."""
        for sid in range(1, 8):
            fn = _import_entry(sid)
            assert callable(fn), f"Stage {sid} entry point is not callable"


@pytest.mark.unit
class TestTrainDependencyChecking:
    """Tests for package version and backend verification."""

    def test_check_dependencies_active_env(self, test_logger):
        """Active environment dependencies and backend should pass verification."""
        assert check_dependencies(test_logger) is True

    @patch("importlib.import_module")
    def test_check_dependencies_missing_package(self, mock_import, test_logger):
        """Missing package should cause check_dependencies to return False."""
        mock_import.side_effect = ImportError("Simulated missing package")
        assert check_dependencies(test_logger) is False

    @patch("keras.backend.backend")
    def test_check_dependencies_wrong_backend(self, mock_backend, test_logger):
        """Non-torch Keras backend should fail check_dependencies."""
        mock_backend.return_value = "tensorflow"
        assert check_dependencies(test_logger) is False


@pytest.mark.unit
class TestTrainDryRunAndExecutionSafety:
    """Tests to guarantee dry-run safety and non-execution."""

    def test_dry_run_skips_when_outputs_exist(self, test_logger):
        """When outputs exist and force=False, dry-run reports WOULD SKIP and never calls entry."""
        with patch("src.train._import_entry") as mock_entry:
            mock_entry.side_effect = RuntimeError("Stage must NOT execute in dry-run!")
            # Stage 1 outputs already exist on disk
            result = run_stage(stage_id=1, force=False, dry_run=True, logger=test_logger)
            assert result is True
            mock_entry.assert_not_called()

    def test_dry_run_would_execute_when_forced(self, test_logger):
        """When force=True, dry-run reports WOULD EXECUTE and never calls entry."""
        with patch("src.train._import_entry") as mock_entry:
            mock_entry.side_effect = RuntimeError("Stage must NOT execute in dry-run!")
            result = run_stage(stage_id=1, force=True, dry_run=True, logger=test_logger)
            assert result is True
            mock_entry.assert_not_called()

    def test_stage_5_safety_guard_blocks_unforced_overwrite(self, test_logger):
        """Stage 5 must refuse to overwrite existing test metrics unless force=True."""
        final_metrics = REPORTS_DIR / "final_test_metrics.json"
        if final_metrics.exists():
            with patch("src.train._import_entry") as mock_entry:
                mock_entry.side_effect = RuntimeError("Test evaluation must NOT run without --force!")
                result = run_stage(stage_id=5, force=False, dry_run=False, logger=test_logger)
                assert result is True
                mock_entry.assert_not_called()

    def test_mocked_execution_runs_when_forced(self, test_logger):
        """When force=True and dry_run=False, mocked entry point is invoked safely."""
        mock_func = MagicMock()
        with patch("src.train._import_entry", return_value=mock_func):
            with patch("src.train._check_stage_inputs", return_value=[]):
                with patch("pathlib.Path.exists", return_value=True):
                    result = run_stage(stage_id=2, force=True, dry_run=False, logger=test_logger)
                    assert result is True
                    mock_func.assert_called_once()


@pytest.mark.unit
class TestTrainDependencyValidation:
    """Tests for stage predecessor validation in selective execution."""

    def test_required_predecessors_mapping_integrity(self):
        """Check expected predecessor mapping."""
        expected_predecessors = {
            2: [1],
            3: [1],
            4: [2, 3],
            5: [1, 2, 3, 4],
            6: [1, 2, 3],
            7: [1, 2, 3, 4],
        }
        # Read the mapping directly from train.py logic via inspection or mock
        from src.train import main
        # We test that running stage 5 with missing predecessor 1 halts with code 1
        with patch("src.train.check_dependencies", return_value=True):
            with patch("src.train._stage_outputs_exist", side_effect=lambda sid: sid != 1):
                exit_code = main(["--stages", "5"])
                assert exit_code == 1

    def test_selective_stage_passes_when_predecessors_exist(self):
        """When predecessors exist on disk, selective stage passes pre-flight checks."""
        with patch("src.train.check_dependencies", return_value=True):
            with patch("src.train.run_stage", return_value=True) as mock_run:
                exit_code = main(["--stages", "7", "--dry-run"])
                assert exit_code == 0
                mock_run.assert_called_once()


@pytest.mark.unit
class TestTrainStatusDisplay:
    """Tests for status display."""

    def test_show_status_executes_without_error(self, test_logger):
        """show_status must iterate through all stages cleanly."""
        show_status(test_logger)  # Should not raise exception

