#!/usr/bin/env python3
"""
Comprehensive Test Runner for Arxiver Application

This script runs all test suites and provides a complete validation
report of the application's integrity and functionality.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class TestRunner:
    """Comprehensive test runner with reporting."""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_suites": {},
            "overall_status": "UNKNOWN",
            "summary": {
                "total_suites": 0,
                "passed_suites": 0,
                "failed_suites": 0,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
            },
        }

    def run_test_suite(self, test_file: str, description: str) -> dict:
        """Run a single test suite and return results."""
        print(f"\n{'='*60}")
        print(f"🧪 Running {description}")
        print(f"{'='*60}")

        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return {
                "status": "SKIPPED",
                "reason": "Test file not found",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "output": "",
                "errors": "",
            }

        try:
            # Run pytest with detailed output
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    test_file,
                    "-v",
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Parse pytest JSON report if available
            test_stats = self._parse_pytest_output(result.stdout)

            suite_result = {
                "status": "PASSED" if result.returncode == 0 else "FAILED",
                "exit_code": result.returncode,
                "tests_run": test_stats.get("total", 0),
                "tests_passed": test_stats.get("passed", 0),
                "tests_failed": test_stats.get("failed", 0),
                "output": result.stdout[-2000:],  # Last 2000 chars
                "errors": result.stderr[-1000:] if result.stderr else "",
            }

            # Print summary
            if result.returncode == 0:
                print(f"✅ {description} - ALL TESTS PASSED")
                print(f"   Tests run: {test_stats.get('total', 0)}")
            else:
                print(f"❌ {description} - TESTS FAILED")
                print(f"   Tests run: {test_stats.get('total', 0)}")
                print(f"   Passed: {test_stats.get('passed', 0)}")
                print(f"   Failed: {test_stats.get('failed', 0)}")

                # Show first few lines of error output
                if result.stdout:
                    error_lines = [
                        line
                        for line in result.stdout.split("\n")
                        if "FAILED" in line or "ERROR" in line
                    ]
                    if error_lines:
                        print("   Key errors:")
                        for line in error_lines[:3]:
                            print(f"     {line}")

            return suite_result

        except subprocess.TimeoutExpired:
            print(f"⏰ {description} - TIMEOUT (exceeded 5 minutes)")
            return {
                "status": "TIMEOUT",
                "reason": "Test execution exceeded 5 minutes",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "output": "",
                "errors": "Test execution timeout",
            }
        except Exception as e:
            print(f"💥 {description} - EXECUTION ERROR: {e}")
            return {
                "status": "ERROR",
                "reason": str(e),
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "output": "",
                "errors": str(e),
            }

    def _parse_pytest_output(self, output: str) -> dict:
        """Parse pytest output to extract test statistics."""
        stats = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}

        # Look for pytest summary line
        lines = output.split("\n")
        for line in lines:
            if " passed" in line or " failed" in line:
                # Parse lines like "2 passed, 1 failed in 0.05s"
                if " passed" in line:
                    try:
                        passed = int(line.split()[0])
                        stats["passed"] = passed
                        stats["total"] += passed
                    except (ValueError, IndexError):
                        pass

                if " failed" in line:
                    try:
                        # Find the number before "failed"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "failed" and i > 0:
                                failed = int(parts[i - 1])
                                stats["failed"] = failed
                                stats["total"] += failed
                                break
                    except (ValueError, IndexError):
                        pass

        return stats

    def run_production_validation(self) -> dict:
        """Run production database validation."""
        print(f"\n{'='*60}")
        print("🔍 Running Production Database Validation")
        print(f"{'='*60}")

        try:
            # Check if production database exists
            db_path = "./data/arxiv_papers.db"
            if not os.path.exists(db_path):
                return {
                    "status": "FAILED",
                    "reason": "Production database not found",
                    "details": {},
                }

            # Run basic validation queries
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get basic stats
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]

            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN authors IS NOT NULL AND authors != '' THEN 1 ELSE 0 END) as has_authors,
                    SUM(CASE WHEN published IS NOT NULL AND published != '' THEN 1 ELSE 0 END) as has_published,
                    SUM(CASE WHEN categories IS NOT NULL AND categories != '' THEN 1 ELSE 0 END) as has_categories,
                    SUM(CASE WHEN concise_summary IS NOT NULL AND concise_summary != '' THEN 1 ELSE 0 END) as has_summaries
                FROM papers
            """)
            metadata_stats = cursor.fetchone()

            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN paper_id LIKE 'http%' THEN 1 ELSE 0 END) as url_format,
                    SUM(CASE WHEN paper_id NOT LIKE 'http%' THEN 1 ELSE 0 END) as id_format
                FROM papers
            """)
            id_format_stats = cursor.fetchone()

            conn.close()

            validation_details = {
                "total_papers": total_papers,
                "metadata_completeness": {
                    "authors": f"{metadata_stats[0]}/{total_papers} ({metadata_stats[0]/total_papers*100:.1f}%)",
                    "published": f"{metadata_stats[1]}/{total_papers} ({metadata_stats[1]/total_papers*100:.1f}%)",
                    "categories": f"{metadata_stats[2]}/{total_papers} ({metadata_stats[2]/total_papers*100:.1f}%)",
                    "concise_summaries": f"{metadata_stats[3]}/{total_papers} ({metadata_stats[3]/total_papers*100:.1f}%)",
                },
                "id_format_consistency": {
                    "url_format": id_format_stats[0],
                    "id_format": id_format_stats[1],
                    "is_consistent": id_format_stats[0] == total_papers
                    or id_format_stats[1] == total_papers,
                },
            }

            # Determine overall status
            issues = []
            if metadata_stats[0] == 0:
                issues.append("No papers have author metadata")
            if metadata_stats[1] == 0:
                issues.append("No papers have publication dates")
            if metadata_stats[2] == 0:
                issues.append("No papers have category metadata")
            if not validation_details["id_format_consistency"]["is_consistent"]:
                issues.append("Inconsistent paper ID formats")

            status = "PASSED" if not issues else "FAILED"

            print(f"📊 Production Database Status: {status}")
            print(f"   Total papers: {total_papers:,}")
            print(f"   Metadata completeness:")
            for field, stats in validation_details["metadata_completeness"].items():
                print(f"     {field}: {stats}")
            print(
                f"   ID format consistency: {'✅' if validation_details['id_format_consistency']['is_consistent'] else '❌'}"
            )

            if issues:
                print("   Issues found:")
                for issue in issues:
                    print(f"     ❌ {issue}")

            return {"status": status, "details": validation_details, "issues": issues}

        except Exception as e:
            print(f"💥 Production validation failed: {e}")
            return {"status": "ERROR", "reason": str(e), "details": {}}

    def generate_report(self):
        """Generate a comprehensive test report."""
        print(f"\n{'='*80}")
        print("📋 COMPREHENSIVE TEST REPORT")
        print(f"{'='*80}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Status: {self.results['overall_status']}")

        print(f"\n📊 Summary:")
        summary = self.results["summary"]
        print(
            f"   Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed"
        )
        print(
            f"   Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed"
        )

        print(f"\n📝 Test Suite Details:")
        for suite_name, suite_result in self.results["test_suites"].items():
            status_emoji = "✅" if suite_result["status"] == "PASSED" else "❌"
            print(f"   {status_emoji} {suite_name}: {suite_result['status']}")
            if suite_result["status"] in ["FAILED", "ERROR", "TIMEOUT"]:
                print(f"      Reason: {suite_result.get('reason', 'Test failures')}")

        # Save detailed report to file
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Detailed report saved to: {report_file}")

        print(f"\n🎯 Recommendations:")
        if self.results["overall_status"] == "PASSED":
            print(
                "   ✅ All tests passed! Application appears to be functioning correctly."
            )
            print("   🚀 Ready to proceed with FastMCP enhancements!")
        else:
            print("   ❌ Some tests failed. Priority actions:")
            print("   1. Fix MCP protocol compliance issues first (CRITICAL)")
            print("   2. Address FastMCP server integration problems")
            print("   3. Resolve JSON response validation errors")
            print("   4. Fix MCP resource functionality issues")
            print("   5. Review failed test details in the report file")
            print("   6. Address API endpoint failures")
            print("   7. Implement missing error handling")
            print("   8. Re-run tests after fixes")
            print("")
            print(
                "   ⚠️  Do NOT proceed with FastMCP enhancements until MCP tests pass!"
            )

    def run_all_tests(self):
        """Run all test suites and generate comprehensive report."""
        print("🚀 Starting Comprehensive Arxiver Application Testing")
        print(f"Test execution started at: {self.results['timestamp']}")

        # Define test suites
        test_suites = [
            # Core MCP Tests (High Priority)
            ("tests/test_mcp_protocol.py", "MCP Protocol Compliance Tests"),
            (
                "tests/test_mcp_server_integration.py",
                "FastMCP Server Integration Tests",
            ),
            ("tests/test_mcp_resources.py", "MCP Resource Functionality Tests"),
            ("tests/test_mcp_json_validation.py", "MCP JSON Response Validation Tests"),
            # MCP Integration Tests (Medium Priority)
            ("tests/test_mcp_client_simulation.py", "MCP Client Simulation Tests"),
            ("tests/test_mcp_tools.py", "MCP Tools Integration Tests"),
            ("tests/test_mcp_error_handling.py", "MCP Error Handling Tests"),
            # Application Tests (Existing)
            ("tests/test_data_integrity.py", "Data Integrity Tests"),
            ("tests/test_ingestion_workflow.py", "Ingestion Workflow Tests"),
            ("tests/test_api_endpoints.py", "API Endpoint Tests"),
            # Legacy Module Tests
            ("arxiver/test_arxiv.py", "ArXiv Module Tests (Legacy)"),
            ("arxiver/test_database.py", "Database Module Tests (Legacy)"),
        ]

        # Run production validation first
        prod_validation = self.run_production_validation()
        self.results["production_validation"] = prod_validation

        # Run each test suite
        all_passed = True
        total_tests = 0
        total_passed = 0
        total_failed = 0

        for test_file, description in test_suites:
            suite_result = self.run_test_suite(test_file, description)
            self.results["test_suites"][description] = suite_result

            if suite_result["status"] not in ["PASSED", "SKIPPED"]:
                all_passed = False

            self.results["summary"]["total_suites"] += 1
            if suite_result["status"] == "PASSED":
                self.results["summary"]["passed_suites"] += 1
            else:
                self.results["summary"]["failed_suites"] += 1

            total_tests += suite_result.get("tests_run", 0)
            total_passed += suite_result.get("tests_passed", 0)
            total_failed += suite_result.get("tests_failed", 0)

        # Update summary
        self.results["summary"]["total_tests"] = total_tests
        self.results["summary"]["passed_tests"] = total_passed
        self.results["summary"]["failed_tests"] = total_failed

        # Determine overall status
        if all_passed and prod_validation["status"] == "PASSED":
            self.results["overall_status"] = "PASSED"
        elif all_passed and prod_validation["status"] in ["FAILED", "ERROR"]:
            self.results["overall_status"] = "PASSED_WITH_WARNINGS"
        else:
            self.results["overall_status"] = "FAILED"

        # Generate final report
        self.generate_report()

        return self.results["overall_status"] == "PASSED"


def main():
    """Main test execution."""
    runner = TestRunner()
    success = runner.run_all_tests()

    if success:
        print("\n🎉 All tests passed! Application is ready for use.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please review the report and fix issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
