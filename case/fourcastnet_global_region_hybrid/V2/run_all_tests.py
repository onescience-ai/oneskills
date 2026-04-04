import sys
import subprocess
from pathlib import Path
import os

# 添加当前 case 目录到路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def run_test_file(test_file_path, test_name):
    """
    运行单个测试文件。

    Args:
        test_file_path: 测试文件路径
        test_name: 测试名称

    Returns:
        success: 是否成功
    """
    print("\n" + "=" * 80)
    print(f"Running: {test_name}")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, str(test_file_path)],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=300,
        )

        print(result.stdout)

        if result.returncode != 0:
            print(result.stderr)
            print(f"\n✗ {test_name} FAILED")
            return False
        else:
            print(f"\n✓ {test_name} PASSED")
            return True

    except subprocess.TimeoutExpired:
        print(f"\n✗ {test_name} TIMEOUT")
        return False
    except Exception as e:
        print(f"\n✗ {test_name} ERROR: {str(e)}")
        return False


def main():
    """
    运行所有测试。
    """
    print("\n" + "=" * 80)
    print("FourCastNet Global Region Hybrid - Test Suite")
    print("=" * 80)

    tests = [
        ("tests/test_data.py", "Data Layer Tests"),
        ("tests/test_components.py", "Component Layer Tests"),
        ("tests/test_models.py", "Model Layer Tests"),
        ("tests/test_utils.py", "Utility Layer Tests"),
        ("tests/test_integration.py", "Integration Tests"),
    ]

    results = []

    for test_file, test_name in tests:
        success = run_test_file(test_file, test_name)
        results.append((test_name, success))

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<50} {status}")

    print("\n" + "-" * 80)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
