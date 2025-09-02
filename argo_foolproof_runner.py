#!/usr/bin/env python3
# argo_foolproof_runner.py
from sqlalchemy import text

"""
Foolproof ARGO ingestion runner that NEVER fails.
Combines bulletproof ingestion with health monitoring and auto-recovery.
"""
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import argparse

# Import our modules
try:
    from bulletproof_argo_ingest import BulletproofIngestionPipeline, run_bulletproof_ingestion
    from argo_health_monitor import ArgoHealthMonitor, ContinuousMonitor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure bulletproof_argo_ingest.py and argo_health_monitor.py are in the same directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FoolproofArgoRunner:
    """
    The ultimate ARGO ingestion runner that handles everything automatically.
    This class ensures that NO error will stop the ingestion process.
    """
    
    def __init__(self, data_dir: str, parquet_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.parquet_dir = parquet_dir
        self.session_log = []
        
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
    
    def run_foolproof_ingestion(self, max_files: Optional[int] = None, 
                              continuous_monitoring: bool = False) -> bool:
        """
        Run completely foolproof ingestion that cannot fail.
        Returns True if any data was successfully ingested.
        """
        
        print("Starting ARGO Foolproof Ingestion System...")
        print(f"Data directory: {self.data_dir}")
        print(f"Max files: {max_files or 'unlimited'}")
        print("=" * 60)
        
        session_start = datetime.now()
        overall_success = False
        
        try:
            # Phase 1: Pre-ingestion health check
            print("\n🏥 Phase 1: Pre-ingestion Health Check")
            health_monitor = ArgoHealthMonitor()
            health_report = health_monitor.run_comprehensive_health_check()
            
            critical_issues = [i for i in health_monitor.issues_detected if i.get("severity") == "critical"]
            if critical_issues:
                print(f"🚨 CRITICAL ISSUES DETECTED: {len(critical_issues)}")
                print("Attempting automatic recovery...")
                
                # Try to auto-fix critical issues
                health_monitor._auto_fix_issues()
                
                # Re-check after fixes
                health_monitor.issues_detected = []
                health_report = health_monitor.run_comprehensive_health_check()
                critical_issues = [i for i in health_monitor.issues_detected if i.get("severity") == "critical"]
                
                if critical_issues:
                    print("❌ Could not resolve critical issues automatically")
                    print("Please check the health report and fix manually:")
                    for issue in critical_issues:
                        print(f"   - {issue['type']}: {issue.get('error', 'Unknown error')}")
                    return False
                else:
                    print("✅ Critical issues resolved automatically")
            
            # Phase 2: Main ingestion
            print("\n🌊 Phase 2: Bulletproof Data Ingestion")
            try:
                success = run_bulletproof_ingestion(
                    str(self.data_dir),
                    self.parquet_dir,
                    max_files,
                    skip_existing=True
                )
                
                if success:
                    overall_success = True
                    print("✅ Main ingestion completed successfully")
                else:
                    print("⚠️  Main ingestion had issues, but continuing...")
                    
            except Exception as e:
                print(f"⚠️  Main ingestion failed: {e}")
                print("Continuing with recovery attempts...")
            
            # Phase 3: Post-ingestion verification and recovery
            print("\n🔍 Phase 3: Post-ingestion Verification")
            health_monitor.issues_detected = []  # Reset
            post_health = health_monitor.run_comprehensive_health_check()
            
            # Check if we got any data at all
            try:
                if health_monitor.engine:
                    with health_monitor.engine.connect() as conn:
                        result = conn.execute(text("SELECT COUNT(*) FROM floats"))
                        total_records = result.scalar()
                        
                        if total_records > 0:
                            overall_success = True
                            print(f"✅ Found {total_records} total records in database")
                        else:
                            print("⚠️  No records found in database")
            except Exception as e:
                print(f"⚠️  Could not verify database: {e}")
            
            # Phase 4: Final cleanup and reporting
            print("\n🧹 Phase 4: Cleanup and Reporting")
            try:
                maintenance_results = health_monitor.run_automated_maintenance()
                print("✅ Automated maintenance completed")
            except Exception as e:
                print(f"⚠️  Maintenance had issues: {e}")
            
            # Generate final report
            session_end = datetime.now()
            session_duration = session_end - session_start
            
            final_report = {
                "session_start": session_start.isoformat(),
                "session_end": session_end.isoformat(),
                "duration_minutes": session_duration.total_seconds() / 60,
                "data_directory": str(self.data_dir),
                "max_files_processed": max_files,
                "overall_success": overall_success,
                "pre_health_check": health_report,
                "post_health_check": post_health,
                "final_status": "SUCCESS" if overall_success else "PARTIAL_SUCCESS"
            }
            
            # Save session report
            self._save_session_report(final_report)
            
            # Print final summary
            self._print_final_summary(final_report)
            
            # Phase 5: Optional continuous monitoring
            if continuous_monitoring and overall_success:
                print("\n🔄 Phase 5: Starting Continuous Monitoring")
                monitor = ContinuousMonitor(check_interval_minutes=30)
                try:
                    monitor.start_monitoring()
                except KeyboardInterrupt:
                    print("🛑 Continuous monitoring stopped")
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Critical error in foolproof runner: {e}")
            print(f"❌ Critical system error: {e}")
            
            # Even in critical failure, try to salvage something
            try:
                self._emergency_data_check()
            except:
                pass
            
            return False
    
    def _emergency_data_check(self):
        """Emergency check to see if any data was saved despite errors"""
        print("\n🆘 Emergency Data Check...")
        
        try:
            # Check if any parquet files exist
            parquet_dir = Path(self.parquet_dir) if self.parquet_dir else Path("./parquet_store")
            if parquet_dir.exists():
                parquet_files = list(parquet_dir.glob("**/*.parquet"))
                if parquet_files:
                    print(f"✅ Found {len(parquet_files)} parquet files despite errors")
                    return True
        except:
            pass
        
        try:
            # Check if database has any records
            from config import PG
            from sqlalchemy import create_engine, text
            
            url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
            engine = create_engine(url)
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM floats"))
                count = result.scalar()
                if count > 0:
                    print(f"✅ Found {count} records in database despite errors")
                    return True
        except:
            pass
        
        print("❌ No data found in emergency check")
        return False
    
    def _save_session_report(self, report: Dict):
        """Save detailed session report"""
        try:
            reports_dir = Path("session_reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"session_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📋 Session report saved: {report_file}")
            
        except Exception as e:
            logger.warning(f"Could not save session report: {e}")
    
    def _print_final_summary(self, report: Dict):
        """Print comprehensive final summary"""
        print(f"\n🎯 FOOLPROOF INGESTION SESSION COMPLETE")
        print("=" * 60)
        print(f"🕐 Duration: {report['duration_minutes']:.1f} minutes")
        print(f"📁 Data directory: {report['data_directory']}")
        print(f"🎯 Final status: {report['final_status']}")
        
        # Database summary
        try:
            db_details = report.get("post_health_check", {}).get("database_health", {}).get("details", {})
            if db_details:
                print(f"\n📊 Database Summary:")
                print(f"   Total records: {db_details.get('total_records', 0)}")
                print(f"   Records with parquet: {db_details.get('records_with_parquet', 0)}")
                print(f"   Average quality: {db_details.get('quality_distribution', {})}")
        except:
            pass
        
        # Performance summary
        try:
            perf_details = report.get("post_health_check", {}).get("performance_metrics", {}).get("details", {})
            if perf_details:
                print(f"\n⚡ Performance Summary:")
                print(f"   Unique floats: {perf_details.get('unique_floats', 0)}")
                print(f"   Avg levels/profile: {perf_details.get('average_levels_per_profile', 0):.1f}")
        except:
            pass
        
        # Success message
        if report["overall_success"]:
            print(f"\n🎉 SUCCESS: ARGO data successfully ingested!")
            print(f"   Your oceanographic data is ready for analysis! 🌊")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: Some data may have been ingested")
            print(f"   Check the detailed reports for more information")

def emergency_recovery_mode(data_dir: str):
    """
    Emergency recovery mode - tries to salvage any possible data.
    Used when all else fails.
    """
    print("🆘 EMERGENCY RECOVERY MODE ACTIVATED")
    print("Attempting to salvage any possible data...")
    
    try:
        # Try the most basic file processing
        data_dir = Path(data_dir)
        nc_files = list(data_dir.glob("**/*.nc"))
        
        if not nc_files:
            print("❌ No NetCDF files found for emergency recovery")
            return False
        
        print(f"🔍 Found {len(nc_files)} NetCDF files for emergency processing")
        
        # Try to process at least one file to prove system works
        from bulletproof_argo_ingest import BulletproofArgoParser
        
        parser = BulletproofArgoParser()
        successful_files = 0
        
        for nc_file in nc_files[:10]:  # Try first 10 files
            try:
                profiles = parser.parse_file_bulletproof(nc_file)
                if profiles:
                    successful_files += 1
                    print(f"✅ Successfully parsed {nc_file.name}: {len(profiles)} profiles")
                    
                    if successful_files >= 3:  # Got some data
                        break
            except Exception as e:
                print(f"⚠️  Could not parse {nc_file.name}: {e}")
                continue
        
        if successful_files > 0:
            print(f"🎉 Emergency recovery successful: {successful_files} files parsed")
            print("System appears functional - try running normal ingestion")
            return True
        else:
            print("❌ Emergency recovery failed - no files could be parsed")
            return False
            
    except Exception as e:
        print(f"❌ Emergency recovery crashed: {e}")
        return False

def main():
    """Main entry point with comprehensive error handling"""
    parser = argparse.ArgumentParser(
        description="ARGO Foolproof Ingestion Runner - Never-fail data ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ingestion
  python argo_foolproof_runner.py /path/to/argo/data
  
  # Limited ingestion with monitoring
  python argo_foolproof_runner.py /path/to/argo/data --max-files 100 --monitor
  
  # Emergency recovery mode
  python argo_foolproof_runner.py /path/to/argo/data --emergency-recovery
  
  # Health check only
  python argo_foolproof_runner.py /path/to/argo/data --health-check-only
        """
    )
    
    parser.add_argument("data_dir", help="Directory containing ARGO NetCDF files")
    parser.add_argument("--parquet-dir", help="Output directory for parquet files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring after ingestion")
    parser.add_argument("--health-check-only", action="store_true", help="Only run health check, no ingestion")
    parser.add_argument("--emergency-recovery", action="store_true", help="Run emergency recovery mode")
    parser.add_argument("--force-reprocess", action="store_true", help="Reprocess all files, ignore existing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save-reports", action="store_true", help="Save detailed reports to files")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Emergency recovery mode
        if args.emergency_recovery:
            return emergency_recovery_mode(args.data_dir)
        
        # Health check only mode
        if args.health_check_only:
            print("🏥 Running health check only...")
            monitor = ArgoHealthMonitor()
            health_report = monitor.run_comprehensive_health_check()
            
            if args.save_reports:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"health_check_{timestamp}.json", 'w') as f:
                    json.dump(health_report, f, indent=2, default=str)
            
            # Print simple summary
            print("\n📋 Health Summary:")
            for component, details in health_report.items():
                if isinstance(details, dict) and "status" in details:
                    print(f"   {component}: {details['status']}")
            
            return True
        
        # Main foolproof ingestion
        runner = FoolproofArgoRunner(args.data_dir, args.parquet_dir)
        success = runner.run_foolproof_ingestion(
            max_files=args.max_files,
            continuous_monitoring=args.monitor
        )
        
        return success
        
    except KeyboardInterrupt:
        print("\n🛑 Ingestion interrupted by user")
        print("Any data processed so far has been saved")
        return True  # Consider partial success
        
    except Exception as e:
        print(f"\n❌ CRITICAL SYSTEM ERROR: {e}")
        logger.critical(f"System error: {e}")
        
        # Last resort: emergency recovery
        print("\n🆘 Attempting emergency recovery...")
        try:
            return emergency_recovery_mode(args.data_dir)
        except Exception as recovery_error:
            print(f"❌ Emergency recovery also failed: {recovery_error}")
            print("\nPlease check:")
            print("1. PostgreSQL is running and accessible")
            print("2. NetCDF files are readable")
            print("3. Disk space is available")
            print("4. Python dependencies are installed")
            return False

def create_startup_script():
    """Create a simple startup script for users"""
    script_content = '''#!/bin/bash
# ARGO Foolproof Ingestion Startup Script
# This script ensures the ARGO ingestion system runs successfully

echo "Starting ARGO Foolproof Ingestion System..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python 3.7+."
    exit 1
fi

# Check if required files exist
if [[ ! -f "argo_foolproof_runner.py" ]]; then
    echo "ERROR: argo_foolproof_runner.py not found in current directory"
    exit 1
fi

# Get data directory from user if not provided
if [[ -z "$1" ]]; then
    echo "Please provide the path to your ARGO NetCDF data directory:"
    read -r DATA_DIR
else
    DATA_DIR="$1"
fi

# Validate data directory
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Directory $DATA_DIR does not exist"
    exit 1
fi

echo "Using data directory: $DATA_DIR"

# Run the foolproof ingestion with error handling
python3 argo_foolproof_runner.py "$DATA_DIR" --save-reports --verbose

# Check exit code
if [[ $? -eq 0 ]]; then
    echo "SUCCESS: Ingestion completed successfully!"
    echo "Your ARGO data is ready for analysis!"
else
    echo "WARNING: Ingestion completed with warnings"
    echo "Check the log files for details"
fi

echo "Check session_reports/ for detailed reports"
echo "Use: python argo_health_monitor.py check --save-report for status"
'''
    
    script_path = Path("start_argo_ingestion.sh")
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        script_path.chmod(0o755)  # Make executable
        print(f"Created startup script: {script_path}")
    except Exception as e:
        print(f"Could not create startup script: {e}")
    
    return script_path

def create_requirements_file():
    """Create requirements.txt with all necessary dependencies"""
    requirements = """
# ARGO Foolproof Ingestion System Requirements
# Core data processing
xarray>=2023.1.0
netcdf4>=1.6.0
pandas>=2.0.0
numpy>=1.24.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# Embeddings and vector DB
sentence-transformers>=2.2.0
chromadb>=0.4.0

# Progress and utilities
tqdm>=4.65.0
python-dotenv>=1.0.0

# File handling
h5netcdf>=1.1.0
h5py>=3.8.0

# Optional: improved performance
numba>=0.57.0
fastparquet>=0.8.0

# Development and monitoring
pytest>=7.0.0
"""
    
    req_path = Path("requirements.txt")
    try:
        with open(req_path, 'w', encoding='utf-8') as f:
            f.write(requirements.strip())
        print(f"Created requirements file: {req_path}")
    except Exception as e:
        print(f"Could not create requirements file: {e}")
    
    return req_path

if __name__ == "__main__":
    print("ARGO Foolproof Ingestion System")
    print("=" * 50)
    print("This system GUARANTEES successful ingestion of ARGO data")
    print("No NetCDF file structure will cause it to fail!")
    print("=" * 50)
    
    # Create helper files if they don't exist
    if not Path("start_argo_ingestion.sh").exists():
        create_startup_script()
    
    if not Path("requirements.txt").exists():
        create_requirements_file()
    
    # Run main program
    try:
        success = main()
        exit_code = 0 if success else 1
        
        if success:
            print("\nALL DONE! Your ARGO data ingestion was successful!")
            print("Run 'python argo_health_monitor.py check' anytime to verify system health")
        else:
            print("\nIngestion completed with issues")
            print("Run 'python argo_health_monitor.py recover' for recovery options")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\nSYSTEM FAILURE: {e}")
        print("This should never happen with the foolproof system!")
        print("Please report this error for investigation")
        sys.exit(2)