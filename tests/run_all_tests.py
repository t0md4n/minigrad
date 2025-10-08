"""
Master test runner - executes all tests and training scripts.
Verifies complete functionality of the vibe-nn library.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, description, project_root):
    """Run a Python script and report results."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*70)

    # Add project root to PYTHONPATH so vibe_nn can be imported
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        env=env
    )

    if result.returncode == 0:
        print(result.stdout)
        print(f"✓ {description} - PASSED")
        return True
    else:
        print(result.stdout)
        print(result.stderr)
        print(f"✗ {description} - FAILED")
        return False

def main():
    """Run all tests and training scripts."""
    print("\n" + "="*70)
    print(" VIBE-NN COMPREHENSIVE TEST SUITE")
    print("="*70)

    # Get paths relative to this script
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent
    examples_dir = project_root / "examples"

    tests = [
        (str(tests_dir / "test_tensor.py"), "Tensor Operations & Gradients Test Suite"),
        (str(examples_dir / "train_xor.py"), "XOR Problem Training"),
        (str(examples_dir / "train_classification.py"), "Binary Classification (Moons Dataset)"),
        (str(examples_dir / "train_regression.py"), "Linear Regression Training"),
    ]

    results = []
    for script, description in tests:
        success = run_script(script, description, project_root)
        results.append((description, success))

    # Print summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for description, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:12} - {description}")

    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Library is fully functional.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
