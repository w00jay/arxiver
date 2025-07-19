#!/usr/bin/env python3
"""
Data Recovery Strategy for Arxiver Missing Metadata

This script provides multiple strategies to recover the missing
authors, published dates, categories, and URLs for existing papers.
"""

import json
import logging
import os
import pickle
import signal
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Add arxiver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "arxiver"))

from arxiv import parse_arxiv_entry
from database import create_connection, update_paper_metadata

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataRecoveryManager:
    """Manages data recovery operations for missing metadata.

    Features:
    - Idempotent: Can be interrupted and resumed safely
    - Rate-limited: Respects arXiv's 3-second minimum delay
    - Progress tracking: Saves state and allows resumption
    - Single connection: Uses only one connection as required by arXiv
    """

    def __init__(self, db_path: str = "./data/arxiv_papers.db"):
        self.db_path = db_path
        self.rate_limit_delay = (
            3.1  # ArXiv API limit: 3 seconds minimum between requests
        )
        self.batch_size = 50  # Smaller batches for better resumability
        self.max_retries = 3
        self.state_file = "recovery_state.pkl"
        self.progress_file = "recovery_progress.json"
        self.session = requests.Session()  # Single connection as required
        self.interrupted = False

        # Set up signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Ensure state directory exists
        Path("recovery_state").mkdir(exist_ok=True)
        self.state_file = "recovery_state/recovery_state.pkl"
        self.progress_file = "recovery_state/recovery_progress.json"

    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        print(f"\n🛑 Recovery interrupted by signal {signum}. Saving state...")
        self.interrupted = True

    def analyze_missing_data(self) -> Dict:
        """Analyze the current state of missing data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get overall statistics
        cursor.execute("SELECT COUNT(*) FROM papers")
        total_papers = cursor.fetchone()[0]

        # Check metadata completeness
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN authors IS NULL OR authors = '' THEN 1 ELSE 0 END) as missing_authors,
                SUM(CASE WHEN published IS NULL OR published = '' THEN 1 ELSE 0 END) as missing_published,
                SUM(CASE WHEN categories IS NULL OR categories = '' THEN 1 ELSE 0 END) as missing_categories,
                SUM(CASE WHEN arxiv_url IS NULL OR arxiv_url = '' THEN 1 ELSE 0 END) as missing_arxiv_url,
                SUM(CASE WHEN pdf_url IS NULL OR pdf_url = '' THEN 1 ELSE 0 END) as missing_pdf_url
            FROM papers
        """)

        missing_stats = cursor.fetchone()

        # Get sample of papers missing metadata
        cursor.execute("""
            SELECT paper_id, title, updated 
            FROM papers 
            WHERE authors IS NULL OR authors = ''
            ORDER BY updated DESC 
            LIMIT 10
        """)
        sample_papers = cursor.fetchall()

        conn.close()

        analysis = {
            "total_papers": total_papers,
            "missing_data": {
                "authors": missing_stats[0],
                "published": missing_stats[1],
                "categories": missing_stats[2],
                "arxiv_url": missing_stats[3],
                "pdf_url": missing_stats[4],
            },
            "completeness_percentage": {
                "authors": (total_papers - missing_stats[0]) / total_papers * 100,
                "published": (total_papers - missing_stats[1]) / total_papers * 100,
                "categories": (total_papers - missing_stats[2]) / total_papers * 100,
                "arxiv_url": (total_papers - missing_stats[3]) / total_papers * 100,
                "pdf_url": (total_papers - missing_stats[4]) / total_papers * 100,
            },
            "sample_papers": sample_papers,
        }

        return analysis

    def save_recovery_state(self, state: Dict):
        """Save recovery state for resumption."""
        try:
            with open(self.state_file, "wb") as f:
                pickle.dump(state, f)

            # Also save human-readable progress
            progress = {
                "timestamp": datetime.now().isoformat(),
                "completed_papers": state.get("completed_papers", []),
                "failed_papers": state.get("failed_papers", []),
                "progress": {
                    "total": state.get("total_papers", 0),
                    "completed": len(state.get("completed_papers", [])),
                    "failed": len(state.get("failed_papers", [])),
                    "remaining": state.get("total_papers", 0)
                    - len(state.get("completed_papers", []))
                    - len(state.get("failed_papers", [])),
                },
            }

            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save recovery state: {e}")

    def load_recovery_state(self) -> Optional[Dict]:
        """Load previous recovery state if exists."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load recovery state: {e}")
        return None

    def clear_recovery_state(self):
        """Clear recovery state files."""
        for file_path in [self.state_file, self.progress_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def get_papers_needing_recovery(
        self, limit: Optional[int] = None, exclude_completed: List[str] = None
    ) -> List[Dict]:
        """Get list of papers that need metadata recovery, excluding already completed ones."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build exclusion condition
        exclusion_condition = ""
        params = []
        if exclude_completed:
            placeholders = ",".join(["?" for _ in exclude_completed])
            exclusion_condition = f"AND paper_id NOT IN ({placeholders})"
            params.extend(exclude_completed)

        query = f"""
            SELECT paper_id, title, summary, updated
            FROM papers 
            WHERE ((authors IS NULL OR authors = '')
               OR (published IS NULL OR published = '')
               OR (categories IS NULL OR categories = '')
               OR (arxiv_url IS NULL OR arxiv_url = '')
               OR (pdf_url IS NULL OR pdf_url = ''))
               {exclusion_condition}
            ORDER BY updated DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        papers = cursor.fetchall()
        conn.close()

        return [
            {
                "paper_id": paper[0],
                "title": paper[1],
                "summary": paper[2],
                "updated": paper[3],
            }
            for paper in papers
        ]

    def is_paper_metadata_complete(self, paper_id: str) -> bool:
        """Check if a paper already has complete metadata (idempotent check)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT authors, published, categories, arxiv_url, pdf_url
            FROM papers WHERE paper_id = ?
        """,
            (paper_id,),
        )

        result = cursor.fetchone()
        conn.close()

        if not result:
            return False

        # Check if all critical metadata fields are present
        authors, published, categories, arxiv_url, pdf_url = result
        return all(
            [
                authors and authors.strip(),
                published and published.strip(),
                categories and categories.strip(),
                arxiv_url and arxiv_url.strip(),
                pdf_url and pdf_url.strip(),
            ]
        )

    def extract_arxiv_id_from_paper_id(self, paper_id: str) -> str:
        """Extract clean arXiv ID from paper_id."""
        # Handle both URL format and clean ID format
        if "arxiv.org/abs/" in paper_id:
            return paper_id.split("/")[-1]
        else:
            return paper_id

    def fetch_paper_metadata_from_arxiv(
        self, arxiv_id: str, retry_count: int = 0
    ) -> Optional[Dict]:
        """Fetch metadata for a single paper from arXiv API with proper rate limiting."""
        base_url = "http://export.arxiv.org/api/query"
        params = {"id_list": arxiv_id}

        try:
            # Use single session connection as required by arXiv
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            # Find the entry
            entry = root.find("{http://www.w3.org/2005/Atom}entry")
            if entry is None:
                logger.warning(f"No entry found for {arxiv_id}")
                return None

            # Parse the entry using existing function
            parsed_data = parse_arxiv_entry(entry)

            logger.debug(f"Successfully fetched metadata for {arxiv_id}")
            return parsed_data

        except requests.RequestException as e:
            if retry_count < self.max_retries:
                logger.warning(
                    f"Request failed for {arxiv_id} (attempt {retry_count + 1}): {e}. Retrying..."
                )
                time.sleep(
                    self.rate_limit_delay * (retry_count + 1)
                )  # Exponential backoff
                return self.fetch_paper_metadata_from_arxiv(arxiv_id, retry_count + 1)
            else:
                logger.error(
                    f"Request failed for {arxiv_id} after {self.max_retries} retries: {e}"
                )
                return None
        except ET.ParseError as e:
            logger.error(f"XML parsing failed for {arxiv_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {arxiv_id}: {e}")
            return None

    def update_paper_with_metadata(self, paper_id: str, metadata: Dict) -> bool:
        """Update a paper with recovered metadata."""
        conn = create_connection(self.db_path)

        try:
            # Update metadata using existing function
            success = update_paper_metadata(conn, paper_id, metadata)
            conn.close()
            return success

        except Exception as e:
            logger.error(f"Failed to update paper {paper_id}: {e}")
            conn.close()
            return False

    def recover_single_paper(self, paper_id: str) -> Dict:
        """Recover metadata for a single paper (idempotent operation)."""
        result = {
            "paper_id": paper_id,
            "success": False,
            "error": None,
            "skipped": False,
        }

        # Check if paper already has complete metadata (idempotent)
        if self.is_paper_metadata_complete(paper_id):
            result["skipped"] = True
            result["success"] = True
            logger.debug(f"⏭️ Skipping {paper_id} - metadata already complete")
            return result

        arxiv_id = self.extract_arxiv_id_from_paper_id(paper_id)
        logger.info(f"🔄 Recovering metadata for {paper_id} (arXiv ID: {arxiv_id})")

        # Fetch metadata from arXiv with rate limiting
        metadata = self.fetch_paper_metadata_from_arxiv(arxiv_id)

        if metadata:
            # Update paper in database
            success = self.update_paper_with_metadata(paper_id, metadata)
            if success:
                result["success"] = True
                logger.info(f"✅ Successfully updated {paper_id}")
            else:
                result["error"] = "Database update failed"
                logger.error(f"❌ Database update failed for {paper_id}")
        else:
            result["error"] = "ArXiv fetch failed"
            logger.error(f"❌ ArXiv fetch failed for {paper_id}")

        return result

    def progressive_recovery(self, max_papers: int = 1000, resume: bool = True) -> Dict:
        """Perform progressive metadata recovery with full resumability and rate limiting.

        Features:
        - Idempotent: Can be interrupted and resumed at any time
        - Rate-limited: Respects arXiv's 3-second minimum delay
        - Single connection: Uses only one connection as required
        - Progress tracking: Saves state after each paper
        """
        logger.info(f"🚀 Starting progressive recovery for up to {max_papers} papers")

        # Load previous state if resuming
        recovery_state = None
        if resume:
            recovery_state = self.load_recovery_state()
            if recovery_state:
                logger.info(f"📂 Resuming previous recovery session")
                logger.info(
                    f"   Completed: {len(recovery_state.get('completed_papers', []))}"
                )
                logger.info(
                    f"   Failed: {len(recovery_state.get('failed_papers', []))}"
                )

        # Initialize or update state
        if not recovery_state:
            recovery_state = {
                "start_time": datetime.now().isoformat(),
                "max_papers": max_papers,
                "completed_papers": [],
                "failed_papers": [],
                "last_paper_index": 0,
            }

        # Get papers needing recovery, excluding already completed ones
        completed_papers = recovery_state.get("completed_papers", [])
        failed_papers = recovery_state.get("failed_papers", [])
        exclude_list = completed_papers + failed_papers

        papers = self.get_papers_needing_recovery(
            limit=max_papers, exclude_completed=exclude_list
        )

        total_papers = len(papers)
        previously_completed = len(completed_papers)
        previously_failed = len(failed_papers)

        if total_papers == 0:
            logger.info("✅ No papers need metadata recovery")
            self.clear_recovery_state()
            return {
                "status": "complete",
                "message": "No papers need recovery",
                "completed": previously_completed,
                "failed": previously_failed,
            }

        logger.info(f"📊 Recovery status:")
        logger.info(f"   Total papers to process: {total_papers}")
        logger.info(f"   Previously completed: {previously_completed}")
        logger.info(f"   Previously failed: {previously_failed}")
        logger.info(f"   Remaining: {total_papers}")

        recovery_state["total_papers"] = (
            total_papers + previously_completed + previously_failed
        )

        overall_results = {
            "total_papers": recovery_state["total_papers"],
            "attempted": previously_completed + previously_failed,
            "successful": previously_completed,
            "failed": previously_failed,
            "skipped": 0,
            "errors": [],
            "estimated_time_remaining": 0,
            "interrupted": False,
        }

        start_time = datetime.now()

        # Process papers one by one for maximum resumability
        for i, paper in enumerate(papers):
            if self.interrupted:
                logger.info("🛑 Recovery interrupted by user. Saving state...")
                overall_results["interrupted"] = True
                break

            paper_id = paper["paper_id"]
            paper_num = i + 1

            logger.info(
                f"📄 Processing paper {paper_num}/{total_papers}: {paper['title'][:60]}..."
            )

            # Apply rate limiting BEFORE each request (arXiv requirement)
            if i > 0:  # Don't delay before first request
                logger.debug(
                    f"⏰ Rate limiting: waiting {self.rate_limit_delay} seconds..."
                )
                time.sleep(self.rate_limit_delay)

            # Recover metadata for this paper
            result = self.recover_single_paper(paper_id)

            # Update state based on result
            if result["success"]:
                if result["skipped"]:
                    overall_results["skipped"] += 1
                else:
                    recovery_state["completed_papers"].append(paper_id)
                    overall_results["successful"] += 1
            else:
                recovery_state["failed_papers"].append(paper_id)
                overall_results["failed"] += 1
                if result["error"]:
                    overall_results["errors"].append(f"{paper_id}: {result['error']}")

            overall_results["attempted"] += 1
            recovery_state["last_paper_index"] = i

            # Save state after each paper for resumability
            self.save_recovery_state(recovery_state)

            # Update time estimates every 10 papers
            if paper_num % 10 == 0 or paper_num == total_papers:
                elapsed = datetime.now() - start_time
                if (
                    overall_results["attempted"]
                    > previously_completed + previously_failed
                ):
                    rate = (
                        overall_results["attempted"]
                        - previously_completed
                        - previously_failed
                    ) / elapsed.total_seconds()
                    remaining_papers = total_papers - paper_num
                    estimated_seconds = remaining_papers / rate if rate > 0 else 0
                    overall_results["estimated_time_remaining"] = estimated_seconds

                    hours = int(estimated_seconds // 3600)
                    minutes = int((estimated_seconds % 3600) // 60)

                    logger.info(
                        f"📈 Progress: {paper_num}/{total_papers} papers processed"
                    )
                    logger.info(
                        f"   Success rate: {overall_results['successful']}/{overall_results['attempted']} ({overall_results['successful']/overall_results['attempted']*100:.1f}%)"
                    )
                    if estimated_seconds > 0:
                        logger.info(f"   Estimated time remaining: {hours}h {minutes}m")

        # Final results
        total_time = datetime.now() - start_time
        overall_results["total_time_seconds"] = total_time.total_seconds()

        if not overall_results["interrupted"]:
            overall_results["status"] = "complete"
            logger.info(f"🎉 Progressive recovery complete!")
            self.clear_recovery_state()  # Clean up state files
        else:
            overall_results["status"] = "interrupted"
            logger.info(f"⏸️ Recovery interrupted - state saved for resumption")

        logger.info(f"📊 Final Results:")
        logger.info(f"   Total time: {total_time}")
        logger.info(
            f"   Papers processed: {overall_results['attempted']}/{overall_results['total_papers']}"
        )
        logger.info(f"   Successfully recovered: {overall_results['successful']}")
        logger.info(f"   Failed: {overall_results['failed']}")
        logger.info(f"   Skipped (already complete): {overall_results['skipped']}")

        return overall_results

    def priority_recovery(self, priority_count: int = 100) -> Dict:
        """Recover metadata for most recent papers first (idempotent)."""
        logger.info(
            f"🎯 Starting priority recovery for {priority_count} most recent papers"
        )
        return self.progressive_recovery(max_papers=priority_count, resume=False)

    def validate_recovery(self) -> Dict:
        """Validate the recovery results."""
        logger.info("Validating recovery results...")

        analysis_before = self.analyze_missing_data()

        return {
            "validation_timestamp": datetime.now().isoformat(),
            "current_state": analysis_before,
            "recommendations": self._generate_recommendations(analysis_before),
        }

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on current data state."""
        recommendations = []

        total = analysis["total_papers"]
        missing = analysis["missing_data"]

        if missing["authors"] > total * 0.9:
            recommendations.append(
                "CRITICAL: Over 90% of papers missing author data - full recovery recommended"
            )
        elif missing["authors"] > total * 0.5:
            recommendations.append(
                "HIGH: Over 50% of papers missing author data - progressive recovery recommended"
            )

        if missing["categories"] > total * 0.9:
            recommendations.append(
                "CRITICAL: Over 90% of papers missing category data - full recovery recommended"
            )

        if missing["published"] > total * 0.9:
            recommendations.append(
                "CRITICAL: Over 90% of papers missing publication dates - full recovery recommended"
            )

        if missing["arxiv_url"] > total * 0.5:
            recommendations.append(
                "MEDIUM: Many papers missing arXiv URLs - consider recovery"
            )

        if not recommendations:
            recommendations.append(
                "LOW: Metadata is mostly complete - spot recovery may be sufficient"
            )

        return recommendations


def main():
    """Main recovery execution with user interaction."""
    print("🔧 Arxiver Data Recovery Strategy - IDEMPOTENT & RESUMABLE")
    print("=" * 70)
    print("✅ Respects arXiv API limits (3+ second delays)")
    print("✅ Can be interrupted and resumed at any time")
    print("✅ Single connection as required by arXiv")
    print("✅ Progress saved after each paper")

    recovery_manager = DataRecoveryManager()

    # Check for previous recovery session
    previous_state = recovery_manager.load_recovery_state()
    if previous_state:
        print(f"\n📂 Previous recovery session found:")
        print(f"   Completed papers: {len(previous_state.get('completed_papers', []))}")
        print(f"   Failed papers: {len(previous_state.get('failed_papers', []))}")
        print(f"   Started: {previous_state.get('start_time', 'Unknown')}")

        resume_choice = input("\nResume previous session? (Y/n): ").strip().lower()
        if resume_choice in ["", "y", "yes"]:
            print("\n🔄 Resuming previous recovery session...")
            results = recovery_manager.progressive_recovery(
                max_papers=previous_state.get("max_papers", 1000), resume=True
            )
        else:
            print("\n🗑️ Clearing previous session...")
            recovery_manager.clear_recovery_state()
    else:
        # Analyze current state
        print("\n📊 Analyzing current data state...")
        analysis = recovery_manager.analyze_missing_data()

        print(f"\nCurrent Database State:")
        print(f"  Total papers: {analysis['total_papers']:,}")
        print(f"  Missing metadata:")
        for field, count in analysis["missing_data"].items():
            percentage = (count / analysis["total_papers"]) * 100
            print(f"    {field}: {count:,} papers ({percentage:.1f}%)")

        print(f"\nData Completeness:")
        for field, percentage in analysis["completeness_percentage"].items():
            print(f"    {field}: {percentage:.1f}% complete")

        print(f"\nRecommendations:")
        recommendations = recovery_manager._generate_recommendations(analysis)
        for rec in recommendations:
            print(f"  • {rec}")

        # Offer recovery options
        print(f"\n🛠️ Recovery Options (All are resumable):")
        print("1. Priority Recovery (100 most recent papers) - ~5 minutes")
        print("2. Progressive Recovery (1000 papers) - ~1 hour")
        print("3. Full Recovery (all papers) - ~20 hours")
        print("4. Custom Recovery (specify number)")
        print("5. Validation Only (no recovery)")
        print("6. Exit")

        while True:
            try:
                choice = input("\nSelect option (1-6): ").strip()

                if choice == "1":
                    print("\n🚀 Starting priority recovery...")
                    print("⏰ Estimated time: 5-10 minutes (100 papers × 3.1s delay)")
                    print("💡 You can interrupt with Ctrl+C and resume anytime")
                    results = recovery_manager.priority_recovery(100)
                    break
                elif choice == "2":
                    print("\n🚀 Starting progressive recovery...")
                    print("⏰ Estimated time: ~1 hour (1000 papers × 3.1s delay)")
                    print("💡 You can interrupt with Ctrl+C and resume anytime")
                    results = recovery_manager.progressive_recovery(1000)
                    break
                elif choice == "3":
                    total_papers = analysis["total_papers"]
                    estimated_hours = (total_papers * 3.1) / 3600
                    print(f"\n⚠️  Full recovery details:")
                    print(f"   Papers to process: {total_papers:,}")
                    print(f"   Estimated time: {estimated_hours:.1f} hours")
                    print(f"   API calls: {total_papers:,} (respects rate limits)")
                    print(f"   Resumable: Yes (can interrupt/resume anytime)")

                    confirm = (
                        input(f"\nProceed with full recovery? (y/N): ").strip().lower()
                    )
                    if confirm == "y":
                        print("\n🚀 Starting full recovery...")
                        print("💡 You can interrupt with Ctrl+C and resume anytime")
                        results = recovery_manager.progressive_recovery(999999)
                        break
                    else:
                        print("Full recovery cancelled.")
                        continue
                elif choice == "4":
                    try:
                        count = int(
                            input("Enter number of papers to recover: ").strip()
                        )
                        if count > 0:
                            estimated_minutes = (count * 3.1) / 60
                            print(f"\n📋 Custom recovery details:")
                            print(f"   Papers: {count}")
                            print(f"   Estimated time: {estimated_minutes:.1f} minutes")
                            print(f"   Resumable: Yes")
                            print(f"\n🚀 Starting recovery for {count} papers...")
                            print("💡 You can interrupt with Ctrl+C and resume anytime")
                            results = recovery_manager.progressive_recovery(count)
                            break
                        else:
                            print("Please enter a positive number.")
                    except ValueError:
                        print("Please enter a valid number.")
                elif choice == "5":
                    print("\n🔍 Running validation only...")
                    results = recovery_manager.validate_recovery()
                    break
                elif choice == "6":
                    print("Exiting without recovery.")
                    return
                else:
                    print("Please select a valid option (1-6).")
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user. No recovery started.")
                return

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"recovery_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    if results.get("status") == "interrupted":
        print(f"\n⏸️ Recovery was interrupted but progress is saved.")
        print(f"   Completed: {results.get('successful', 0)} papers")
        print(f"   Failed: {results.get('failed', 0)} papers")
        print(f"   Run the script again to resume from where you left off.")
    elif "successful" in results:
        print(f"\n✅ Recovery complete!")
        print(
            f"   Successfully recovered: {results['successful']}/{results.get('attempted', 0)} papers"
        )
        if results.get("skipped", 0) > 0:
            print(f"   Skipped (already complete): {results['skipped']} papers")

    print(f"\n💡 You can run this script anytime - it's completely idempotent!")
    print(f"   Papers already processed will be automatically skipped.")


if __name__ == "__main__":
    main()
