# argo_health_monitor.py
"""
ARGO Ingestion Health Monitor and Auto-Recovery System.
Monitors ingestion health, detects issues, and automatically fixes them.
"""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from config import PG, PARQUET_DIR, CHROMA_DIR
from embeddings_utils import EmbeddingManager
import shutil
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArgoHealthMonitor:
    """
    Comprehensive health monitoring and auto-recovery for ARGO ingestion.
    Detects and fixes common issues automatically.
    """
    
    def __init__(self):
        self.engine = self._create_engine()
        self.issues_detected = []
        self.fixes_applied = []
        
    def _create_engine(self):
        """Create database engine with error handling"""
        try:
            url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
            return create_engine(url, pool_pre_ping=True)
        except Exception as e:
            logger.error(f"Could not connect to database: {e}")
            return None
    
    def run_comprehensive_health_check(self) -> Dict:
        """Run all health checks and return detailed report"""
        print("🏥 Running comprehensive ARGO ingestion health check...")
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "database_health": self._check_database_health(),
            "parquet_health": self._check_parquet_health(),
            "embedding_health": self._check_embedding_health(),
            "data_consistency": self._check_data_consistency(),
            "error_analysis": self._analyze_ingestion_errors(),
            "performance_metrics": self._check_performance_metrics(),
            "issues_detected": self.issues_detected,
            "fixes_applied": self.fixes_applied,
            "recommendations": self._generate_recommendations()
        }
        
        # Auto-fix critical issues
        if self.issues_detected:
            print(f"🔧 Detected {len(self.issues_detected)} issues, attempting auto-fixes...")
            self._auto_fix_issues()
        
        return health_report
    
    def _check_database_health(self) -> Dict:
        """Check database health and consistency"""
        health = {"status": "unknown", "details": {}}
        
        if not self.engine:
            health["status"] = "critical"
            health["details"]["error"] = "Cannot connect to database"
            self.issues_detected.append({"type": "database_connection", "severity": "critical"})
            return health
        
        try:
            with self.engine.connect() as conn:
                # Basic connectivity
                conn.execute(text("SELECT 1"))
                
                # Check table existence
                result = conn.execute(text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name IN ('floats', 'ingestion_errors')
                """))
                tables = [row[0] for row in result]
                
                if 'floats' not in tables:
                    self.issues_detected.append({"type": "missing_table", "severity": "critical", "table": "floats"})
                    health["status"] = "critical"
                    health["details"]["missing_tables"] = ["floats"]
                    return health
                
                # Check record counts
                result = conn.execute(text("SELECT COUNT(*) FROM floats"))
                total_records = result.scalar()
                
                # Check for orphaned records (missing parquet files)
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM floats 
                    WHERE parquet_path IS NOT NULL AND parquet_path != ''
                """))
                records_with_parquet = result.scalar()
                
                # Check data quality distribution
                result = conn.execute(text("""
                    SELECT 
                        CASE 
                            WHEN data_quality_score >= 0.8 THEN 'high'
                            WHEN data_quality_score >= 0.5 THEN 'medium'
                            ELSE 'low'
                        END as quality,
                        COUNT(*) as count
                    FROM floats 
                    GROUP BY quality
                """))
                quality_dist = {row[0]: row[1] for row in result}
                
                # Check for recent errors
                if 'ingestion_errors' in tables:
                    result = conn.execute(text("""
                        SELECT COUNT(*) FROM ingestion_errors 
                        WHERE occurred_at > NOW() - INTERVAL '24 hours'
                    """))
                    recent_errors = result.scalar()
                else:
                    recent_errors = 0
                
                health["status"] = "healthy"
                health["details"] = {
                    "total_records": total_records,
                    "records_with_parquet": records_with_parquet,
                    "quality_distribution": quality_dist,
                    "recent_errors_24h": recent_errors,
                    "tables_present": tables
                }
                
                # Detect issues
                if total_records == 0:
                    self.issues_detected.append({"type": "no_data", "severity": "warning"})
                    health["status"] = "warning"
                
                if recent_errors > 10:
                    self.issues_detected.append({"type": "high_error_rate", "severity": "warning", "count": recent_errors})
                    health["status"] = "warning"
                
                orphaned_ratio = 1 - (records_with_parquet / total_records) if total_records > 0 else 0
                if orphaned_ratio > 0.1:  # More than 10% orphaned
                    self.issues_detected.append({"type": "orphaned_records", "severity": "moderate", "ratio": orphaned_ratio})
                
        except Exception as e:
            health["status"] = "critical"
            health["details"]["error"] = str(e)
            self.issues_detected.append({"type": "database_error", "severity": "critical", "error": str(e)})
        
        return health
    
    def _check_parquet_health(self) -> Dict:
        """Check parquet files health"""
        health = {"status": "unknown", "details": {}}
        
        try:
            parquet_dir = Path(PARQUET_DIR)
            if not parquet_dir.exists():
                self.issues_detected.append({"type": "missing_parquet_dir", "severity": "critical"})
                health["status"] = "critical"
                health["details"]["error"] = f"Parquet directory {parquet_dir} does not exist"
                return health
            
            # Count parquet files
            parquet_files = list(parquet_dir.glob("**/*.parquet"))
            total_parquet = len(parquet_files)
            
            # Check file sizes and accessibility
            corrupt_files = []
            total_size = 0
            
            for pfile in parquet_files[:100]:  # Sample check
                try:
                    size = pfile.stat().st_size
                    total_size += size
                    
                    if size == 0:
                        corrupt_files.append(str(pfile))
                    else:
                        # Try to read file header
                        pd.read_parquet(pfile, nrows=1)
                        
                except Exception as e:
                    corrupt_files.append(str(pfile))
                    logger.debug(f"Corrupt parquet file: {pfile} - {e}")
            
            health["status"] = "healthy"
            health["details"] = {
                "total_parquet_files": total_parquet,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "corrupt_files_found": len(corrupt_files),
                "sample_checked": min(100, total_parquet)
            }
            
            # Detect issues
            if total_parquet == 0:
                self.issues_detected.append({"type": "no_parquet_files", "severity": "warning"})
                health["status"] = "warning"
            
            if len(corrupt_files) > 0:
                self.issues_detected.append({
                    "type": "corrupt_parquet_files", 
                    "severity": "moderate",
                    "count": len(corrupt_files),
                    "files": corrupt_files[:10]  # First 10 examples
                })
                
        except Exception as e:
            health["status"] = "critical"
            health["details"]["error"] = str(e)
            self.issues_detected.append({"type": "parquet_check_failed", "severity": "critical", "error": str(e)})
        
        return health
    
    def _check_embedding_health(self) -> Dict:
        """Check ChromaDB embedding health"""
        health = {"status": "unknown", "details": {}}
        
        try:
            # Try to initialize embedding manager
            emb = EmbeddingManager()
            
            # Get collection stats
            stats = emb.get_collection_stats()
            
            if "error" in stats:
                self.issues_detected.append({"type": "embedding_error", "severity": "moderate", "error": stats["error"]})
                health["status"] = "warning"
                health["details"] = stats
            else:
                health["status"] = "healthy"
                health["details"] = stats
                
                # Check if embedding count matches database
                if self.engine:
                    with self.engine.connect() as conn:
                        result = conn.execute(text("SELECT COUNT(*) FROM floats"))
                        db_count = result.scalar()
                        
                        embedding_count = stats.get("total_documents", 0)
                        
                        if abs(db_count - embedding_count) > db_count * 0.1:  # More than 10% difference
                            self.issues_detected.append({
                                "type": "embedding_db_mismatch", 
                                "severity": "moderate",
                                "db_count": db_count,
                                "embedding_count": embedding_count
                            })
            
        except Exception as e:
            health["status"] = "critical"
            health["details"]["error"] = str(e)
            self.issues_detected.append({"type": "embedding_init_failed", "severity": "critical", "error": str(e)})
        
        return health
    
    def _check_data_consistency(self) -> Dict:
        """Check data consistency between database and parquet files"""
        consistency = {"status": "unknown", "details": {}}
        
        if not self.engine:
            return consistency
        
        try:
            with self.engine.connect() as conn:
                # Check for records with missing parquet files
                result = conn.execute(text("""
                    SELECT profile_id, parquet_path FROM floats 
                    WHERE parquet_path IS NOT NULL AND parquet_path != ''
                    LIMIT 100
                """))
                
                missing_files = []
                for row in result:
                    profile_id, parquet_path = row
                    if not Path(parquet_path).exists():
                        missing_files.append({"profile_id": profile_id, "path": parquet_path})
                
                # Check for duplicate profile IDs
                result = conn.execute(text("""
                    SELECT profile_id, COUNT(*) as count FROM floats 
                    GROUP BY profile_id HAVING COUNT(*) > 1
                """))
                duplicates = [{"profile_id": row[0], "count": row[1]} for row in result]
                
                # Check coordinate validity
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM floats 
                    WHERE lat IS NULL OR lon IS NULL 
                       OR lat < -90 OR lat > 90 
                       OR lon < -180 OR lon > 180
                """))
                invalid_coords = result.scalar()
                
                consistency["status"] = "healthy"
                consistency["details"] = {
                    "missing_parquet_files": len(missing_files),
                    "duplicate_profiles": len(duplicates),
                    "invalid_coordinates": invalid_coords,
                    "sample_missing_files": missing_files[:5]
                }
                
                # Detect issues
                if len(missing_files) > 0:
                    self.issues_detected.append({
                        "type": "missing_parquet_files",
                        "severity": "moderate",
                        "count": len(missing_files)
                    })
                
                if len(duplicates) > 0:
                    self.issues_detected.append({
                        "type": "duplicate_profiles",
                        "severity": "moderate",
                        "count": len(duplicates)
                    })
                
                if invalid_coords > 0:
                    self.issues_detected.append({
                        "type": "invalid_coordinates",
                        "severity": "moderate",
                        "count": invalid_coords
                    })
        
        except Exception as e:
            consistency["status"] = "critical"
            consistency["details"]["error"] = str(e)
            self.issues_detected.append({"type": "consistency_check_failed", "severity": "critical", "error": str(e)})
        
        return consistency
    
    def _analyze_ingestion_errors(self) -> Dict:
        """Analyze ingestion errors to identify patterns"""
        analysis = {"status": "unknown", "details": {}}
        
        if not self.engine:
            return analysis
        
        try:
            with self.engine.connect() as conn:
                # Check if error table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = 'ingestion_errors'
                    )
                """))
                
                if not result.scalar():
                    analysis["status"] = "no_error_table"
                    return analysis
                
                # Get error statistics
                result = conn.execute(text("""
                    SELECT 
                        error_type,
                        COUNT(*) as count,
                        MAX(occurred_at) as last_occurrence
                    FROM ingestion_errors 
                    WHERE occurred_at > NOW() - INTERVAL '7 days'
                    GROUP BY error_type 
                    ORDER BY count DESC
                """))
                
                error_stats = []
                for row in result:
                    error_stats.append({
                        "type": row[0],
                        "count": row[1],
                        "last_occurrence": row[2].isoformat() if row[2] else None
                    })
                
                # Get total error count
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM ingestion_errors 
                    WHERE occurred_at > NOW() - INTERVAL '24 hours'
                """))
                recent_errors = result.scalar()
                
                analysis["status"] = "healthy"
                analysis["details"] = {
                    "error_types_7_days": error_stats,
                    "recent_errors_24h": recent_errors,
                    "total_error_types": len(error_stats)
                }
                
                # Detect high error rates
                if recent_errors > 50:
                    self.issues_detected.append({
                        "type": "high_error_rate_24h",
                        "severity": "warning",
                        "count": recent_errors
                    })
                
                # Detect recurring error patterns
                for error_stat in error_stats:
                    if error_stat["count"] > 20:
                        self.issues_detected.append({
                            "type": "recurring_error_pattern",
                            "severity": "moderate",
                            "error_type": error_stat["type"],
                            "count": error_stat["count"]
                        })
        
        except Exception as e:
            analysis["status"] = "critical"
            analysis["details"]["error"] = str(e)
            self.issues_detected.append({"type": "error_analysis_failed", "severity": "critical", "error": str(e)})
        
        return analysis
    
    def _check_performance_metrics(self) -> Dict:
        """Check ingestion performance metrics"""
        metrics = {"status": "unknown", "details": {}}
        
        if not self.engine:
            return metrics
        
        try:
            with self.engine.connect() as conn:
                # Recent ingestion rate
                result = conn.execute(text("""
                    SELECT COUNT(*) as records_today FROM floats 
                    WHERE created_at > CURRENT_DATE
                """))
                records_today = result.scalar()
                
                # Average data quality
                result = conn.execute(text("""
                    SELECT AVG(data_quality_score) as avg_quality FROM floats
                    WHERE data_quality_score IS NOT NULL
                """))
                avg_quality = result.scalar()
                
                # Storage efficiency
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_profiles,
                        AVG(n_levels) as avg_levels,
                        COUNT(DISTINCT float_id) as unique_floats
                    FROM floats
                """))
                row = result.fetchone()
                
                metrics["status"] = "healthy"
                metrics["details"] = {
                    "records_ingested_today": records_today,
                    "average_data_quality": float(avg_quality) if avg_quality else None,
                    "total_profiles": row[0],
                    "average_levels_per_profile": float(row[1]) if row[1] else None,
                    "unique_floats": row[2]
                }
                
                # Performance warnings
                if avg_quality and avg_quality < 0.3:
                    self.issues_detected.append({
                        "type": "low_average_quality",
                        "severity": "warning",
                        "avg_quality": float(avg_quality)
                    })
        
        except Exception as e:
            metrics["status"] = "error"
            metrics["details"]["error"] = str(e)
        
        return metrics
    
    def _auto_fix_issues(self):
        """Automatically fix detected issues where possible"""
        for issue in self.issues_detected:
            try:
                fix_applied = False
                
                if issue["type"] == "missing_parquet_dir":
                    fix_applied = self._fix_missing_parquet_dir()
                
                elif issue["type"] == "corrupt_parquet_files":
                    fix_applied = self._fix_corrupt_parquet_files(issue)
                
                elif issue["type"] == "missing_parquet_files":
                    fix_applied = self._fix_missing_parquet_files()
                
                elif issue["type"] == "embedding_db_mismatch":
                    fix_applied = self._fix_embedding_mismatch()
                
                elif issue["type"] == "duplicate_profiles":
                    fix_applied = self._fix_duplicate_profiles()
                
                elif issue["type"] == "invalid_coordinates":
                    fix_applied = self._fix_invalid_coordinates()
                
                elif issue["type"] == "orphaned_records":
                    fix_applied = self._fix_orphaned_records()
                
                if fix_applied:
                    self.fixes_applied.append({
                        "issue_type": issue["type"],
                        "fix_time": datetime.now().isoformat(),
                        "severity": issue["severity"]
                    })
                    print(f"✅ Fixed: {issue['type']}")
                else:
                    print(f"⚠️  Could not auto-fix: {issue['type']}")
                    
            except Exception as e:
                logger.error(f"Error fixing {issue['type']}: {e}")
    
    def _fix_missing_parquet_dir(self) -> bool:
        """Create missing parquet directory"""
        try:
            Path(PARQUET_DIR).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created parquet directory: {PARQUET_DIR}")
            return True
        except Exception as e:
            logger.error(f"Could not create parquet directory: {e}")
            return False
    
    def _fix_corrupt_parquet_files(self, issue: Dict) -> bool:
        """Remove or attempt to repair corrupt parquet files"""
        try:
            if "files" not in issue:
                return False
            
            fixed_count = 0
            for corrupt_file in issue["files"]:
                try:
                    Path(corrupt_file).unlink()  # Delete corrupt file
                    logger.info(f"Removed corrupt parquet file: {corrupt_file}")
                    fixed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {corrupt_file}: {e}")
            
            return fixed_count > 0
            
        except Exception as e:
            logger.error(f"Error fixing corrupt parquet files: {e}")
            return False
    
    def _fix_missing_parquet_files(self) -> bool:
        """Mark database records with missing parquet files for re-ingestion"""
        if not self.engine:
            return False
        
        try:
            with self.engine.begin() as conn:
                # Find records with missing parquet files
                result = conn.execute(text("""
                    SELECT id, profile_id, parquet_path FROM floats 
                    WHERE parquet_path IS NOT NULL AND parquet_path != ''
                """))
                
                missing_count = 0
                for row in result:
                    record_id, profile_id, parquet_path = row
                    if not Path(parquet_path).exists():
                        # Mark for re-ingestion by clearing parquet path
                        conn.execute(text("""
                            UPDATE floats SET 
                                parquet_path = NULL,
                                data_quality_score = data_quality_score * 0.5,
                                updated_at = NOW()
                            WHERE id = :id
                        """), {"id": record_id})
                        missing_count += 1
                
                if missing_count > 0:
                    logger.info(f"Marked {missing_count} records for re-ingestion due to missing parquet files")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error fixing missing parquet files: {e}")
            return False
    
    def _fix_embedding_mismatch(self) -> bool:
        """Fix embedding/database count mismatch"""
        try:
            # Simple approach: reset embeddings and let next ingestion rebuild them
            emb = EmbeddingManager()
            emb.reset_collection()
            logger.info("Reset embedding collection to fix mismatch")
            return True
            
        except Exception as e:
            logger.error(f"Error fixing embedding mismatch: {e}")
            return False
    
    def _fix_duplicate_profiles(self) -> bool:
        """Remove duplicate profile records, keeping the most recent"""
        if not self.engine:
            return False
        
        try:
            with self.engine.begin() as conn:
                # Delete duplicates, keeping the one with highest ID (most recent)
                result = conn.execute(text("""
                    DELETE FROM floats WHERE id NOT IN (
                        SELECT MAX(id) FROM floats GROUP BY profile_id
                    )
                """))
                
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"Removed {deleted_count} duplicate profile records")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error fixing duplicates: {e}")
            return False
    
    def _fix_invalid_coordinates(self) -> bool:
        """Fix or remove records with invalid coordinates"""
        if not self.engine:
            return False
        
        try:
            with self.engine.begin() as conn:
                # Delete records with completely invalid coordinates
                result = conn.execute(text("""
                    DELETE FROM floats 
                    WHERE lat IS NULL OR lon IS NULL 
                       OR lat < -90 OR lat > 90 
                       OR lon < -180 OR lon > 180
                """))
                
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"Removed {deleted_count} records with invalid coordinates")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error fixing invalid coordinates: {e}")
            return False
    
    def _fix_orphaned_records(self) -> bool:
        """Clean up orphaned database records"""
        if not self.engine:
            return False
        
        try:
            with self.engine.begin() as conn:
                # Mark orphaned records for cleanup
                result = conn.execute(text("""
                    UPDATE floats SET 
                        data_quality_score = LEAST(data_quality_score, 0.2),
                        raw_metadata = raw_metadata || '{"status": "orphaned"}'::jsonb,
                        updated_at = NOW()
                    WHERE (parquet_path IS NULL OR parquet_path = '') 
                      AND extraction_method != 'minimal_fallback'
                """))
                
                updated_count = result.rowcount
                if updated_count > 0:
                    logger.info(f"Marked {updated_count} orphaned records")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error fixing orphaned records: {e}")
            return False
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on detected issues"""
        recommendations = []
        
        # High-level recommendations based on issues
        critical_issues = [i for i in self.issues_detected if i.get("severity") == "critical"]
        if critical_issues:
            recommendations.append("🚨 CRITICAL: Address database connectivity and schema issues immediately")
        
        moderate_issues = [i for i in self.issues_detected if i.get("severity") == "moderate"]
        if len(moderate_issues) > 3:
            recommendations.append("⚠️  Multiple moderate issues detected - consider running full system cleanup")
        
        # Specific recommendations
        issue_types = [i["type"] for i in self.issues_detected]
        
        if "high_error_rate_24h" in issue_types:
            recommendations.append("📈 High error rate detected - review recent NetCDF files for corruption")
        
        if "corrupt_parquet_files" in issue_types:
            recommendations.append("🗂️  Corrupt parquet files found - run cleanup and re-ingestion")
        
        if "embedding_db_mismatch" in issue_types:
            recommendations.append("🔗 Embedding sync issue - consider rebuilding embeddings")
        
        if "low_average_quality" in issue_types:
            recommendations.append("📊 Low data quality detected - review source files and parsing logic")
        
        if not recommendations:
            recommendations.append("✅ System appears healthy - continue regular monitoring")
        
        return recommendations
    
    def create_recovery_plan(self) -> Dict:
        """Create detailed recovery plan for detected issues"""
        recovery_plan = {
            "timestamp": datetime.now().isoformat(),
            "issues_summary": {
                "critical": len([i for i in self.issues_detected if i.get("severity") == "critical"]),
                "moderate": len([i for i in self.issues_detected if i.get("severity") == "moderate"]),
                "warning": len([i for i in self.issues_detected if i.get("severity") == "warning"])
            },
            "recovery_steps": [],
            "estimated_time": "Unknown"
        }
        
        # Generate step-by-step recovery plan
        steps = []
        
        # Critical issues first
        critical_issues = [i for i in self.issues_detected if i.get("severity") == "critical"]
        for issue in critical_issues:
            if issue["type"] == "database_connection":
                steps.append({
                    "priority": 1,
                    "action": "Fix database connection",
                    "command": "Check PostgreSQL service and connection settings",
                    "estimated_time": "5-15 minutes"
                })
            elif issue["type"] == "missing_table":
                steps.append({
                    "priority": 1,
                    "action": "Recreate database schema",
                    "command": "python bulletproof_argo_ingest.py --verify-only",
                    "estimated_time": "2-5 minutes"
                })
        
        # Moderate issues
        moderate_issues = [i for i in self.issues_detected if i.get("severity") == "moderate"]
        for issue in moderate_issues:
            if issue["type"] == "corrupt_parquet_files":
                steps.append({
                    "priority": 2,
                    "action": f"Clean up {issue.get('count', 0)} corrupt parquet files",
                    "command": "Automatic cleanup applied",
                    "estimated_time": "1-5 minutes"
                })
            elif issue["type"] == "missing_parquet_files":
                steps.append({
                    "priority": 2,
                    "action": "Re-ingest profiles with missing parquet files",
                    "command": "python bulletproof_argo_ingest.py --dir <original_dir>",
                    "estimated_time": "10-60 minutes"
                })
        
        recovery_plan["recovery_steps"] = sorted(steps, key=lambda x: x["priority"])
        
        # Estimate total time
        if steps:
            min_time = sum(int(s["estimated_time"].split("-")[0]) for s in steps if "-" in s["estimated_time"])
            max_time = sum(int(s["estimated_time"].split("-")[1].split()[0]) for s in steps if "-" in s["estimated_time"])
            recovery_plan["estimated_time"] = f"{min_time}-{max_time} minutes"
        
        return recovery_plan
    
    def run_automated_maintenance(self) -> Dict:
        """Run automated maintenance tasks"""
        print("🔧 Running automated maintenance...")
        
        maintenance_results = {
            "timestamp": datetime.now().isoformat(),
            "tasks_completed": [],
            "tasks_failed": [],
            "cleanup_stats": {}
        }
        
        # Task 1: Clean old error logs
        try:
            if self.engine:
                with self.engine.begin() as conn:
                    result = conn.execute(text("""
                        DELETE FROM ingestion_errors 
                        WHERE occurred_at < NOW() - INTERVAL '30 days'
                    """))
                    deleted = result.rowcount
                    maintenance_results["tasks_completed"].append(f"Cleaned {deleted} old error logs")
        except Exception as e:
            maintenance_results["tasks_failed"].append(f"Error log cleanup failed: {e}")
        
        # Task 2: Update data quality scores
        try:
            if self.engine:
                with self.engine.begin() as conn:
                    result = conn.execute(text("""
                        UPDATE floats SET data_quality_score = 0.1 
                        WHERE data_quality_score IS NULL
                    """))
                    updated = result.rowcount
                    maintenance_results["tasks_completed"].append(f"Updated {updated} missing quality scores")
        except Exception as e:
            maintenance_results["tasks_failed"].append(f"Quality score update failed: {e}")
        
        # Task 3: Vacuum analyze for performance
        try:
            if self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text("VACUUM ANALYZE floats"))
                    maintenance_results["tasks_completed"].append("Database vacuum completed")
        except Exception as e:
            maintenance_results["tasks_failed"].append(f"Database vacuum failed: {e}")
        
        # Task 4: Clean empty parquet directories
        try:
            parquet_dir = Path(PARQUET_DIR)
            if parquet_dir.exists():
                empty_dirs = []
                for subdir in parquet_dir.iterdir():
                    if subdir.is_dir() and not any(subdir.iterdir()):
                        subdir.rmdir()
                        empty_dirs.append(str(subdir))
                
                if empty_dirs:
                    maintenance_results["tasks_completed"].append(f"Removed {len(empty_dirs)} empty directories")
        except Exception as e:
            maintenance_results["tasks_failed"].append(f"Directory cleanup failed: {e}")
        
        print(f"✅ Maintenance completed: {len(maintenance_results['tasks_completed'])} tasks successful")
        if maintenance_results["tasks_failed"]:
            print(f"⚠️  {len(maintenance_results['tasks_failed'])} tasks failed")
        
        return maintenance_results

# Continuous monitoring class
class ContinuousMonitor:
    """
    Continuous monitoring system that can run in background.
    """
    
    def __init__(self, check_interval_minutes: int = 60):
        self.check_interval = check_interval_minutes * 60  # Convert to seconds
        self.monitor = ArgoHealthMonitor()
        self.running = False
    
    def start_monitoring(self, max_iterations: Optional[int] = None):
        """Start continuous monitoring"""
        print(f"🔄 Starting continuous ARGO health monitoring (check every {self.check_interval//60} minutes)")
        
        self.running = True
        iteration = 0
        
        while self.running and (max_iterations is None or iteration < max_iterations):
            try:
                print(f"\n🔍 Health check #{iteration + 1} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run health check
                health_report = self.monitor.run_comprehensive_health_check()
                
                # Save health report
                self._save_health_report(health_report, iteration)
                
                # Check if intervention needed
                critical_issues = [i for i in self.monitor.issues_detected if i.get("severity") == "critical"]
                if critical_issues:
                    print(f"🚨 CRITICAL ISSUES DETECTED: {len(critical_issues)} issues")
                    print("🛑 Monitoring paused - manual intervention required")
                    break
                
                # Run maintenance if needed
                if iteration % 24 == 0:  # Every 24 hours
                    self.monitor.run_automated_maintenance()
                
                # Clear issues for next iteration
                self.monitor.issues_detected = []
                self.monitor.fixes_applied = []
                
                iteration += 1
                
                # Wait for next check
                if self.running and (max_iterations is None or iteration < max_iterations):
                    time.sleep(self.check_interval)
                    
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.running = False
        print("🛑 Monitoring stopped")
    
    def _save_health_report(self, report: Dict, iteration: int):
        """Save health report to file"""
        try:
            reports_dir = Path("health_reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"health_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Keep only last 100 reports
            reports = sorted(reports_dir.glob("health_report_*.json"))
            if len(reports) > 100:
                for old_report in reports[:-100]:
                    old_report.unlink()
                    
        except Exception as e:
            logger.warning(f"Could not save health report: {e}")

# CLI interface for the health monitor
def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARGO Ingestion Health Monitor")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Health check command
    check_parser = subparsers.add_parser('check', help='Run single health check')
    check_parser.add_argument('--save-report', action='store_true', help='Save report to file')
    check_parser.add_argument('--auto-fix', action='store_true', help='Automatically fix issues')
    
    # Continuous monitoring command
    monitor_parser = subparsers.add_parser('monitor', help='Start continuous monitoring')
    monitor_parser.add_argument('--interval', type=int, default=60, help='Check interval in minutes')
    monitor_parser.add_argument('--max-iterations', type=int, help='Maximum number of checks')
    
    # Maintenance command
    maintenance_parser = subparsers.add_parser('maintain', help='Run maintenance tasks')
    
    # Recovery command
    recovery_parser = subparsers.add_parser('recover', help='Generate recovery plan')
    recovery_parser.add_argument('--execute', action='store_true', help='Execute recovery plan')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        # Single health check
        monitor = ArgoHealthMonitor()
        health_report = monitor.run_comprehensive_health_check()
        
        print("\n📋 Health Check Report:")
        print("=" * 50)
        
        # Print summary
        for component, details in health_report.items():
            if isinstance(details, dict) and "status" in details:
                status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨", "unknown": "❓"}.get(details["status"], "❓")
                print(f"{status_emoji} {component.replace('_', ' ').title()}: {details['status']}")
        
        # Print issues
        if health_report.get("issues_detected"):
            print(f"\n🔍 Issues Detected ({len(health_report['issues_detected'])}):")
            for issue in health_report["issues_detected"]:
                severity_emoji = {"critical": "🚨", "moderate": "⚠️", "warning": "⚠️"}.get(issue.get("severity"), "ℹ️")
                print(f"   {severity_emoji} {issue['type']} ({issue.get('severity', 'unknown')})")
        
        # Print recommendations
        if health_report.get("recommendations"):
            print(f"\n💡 Recommendations:")
            for rec in health_report["recommendations"]:
                print(f"   {rec}")
        
        # Save report if requested
        if args.save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"health_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(health_report, f, indent=2, default=str)
            print(f"\n💾 Report saved to: {report_file}")
    
    elif args.command == 'monitor':
        # Continuous monitoring
        monitor = ContinuousMonitor(args.interval)
        try:
            monitor.start_monitoring(args.max_iterations)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    
    elif args.command == 'maintain':
        # Run maintenance
        monitor = ArgoHealthMonitor()
        results = monitor.run_automated_maintenance()
        
        print("\n🔧 Maintenance Results:")
        for task in results["tasks_completed"]:
            print(f"   ✅ {task}")
        for task in results["tasks_failed"]:
            print(f"   ❌ {task}")
    
    elif args.command == 'recover':
        # Generate recovery plan
        monitor = ArgoHealthMonitor()
        monitor.run_comprehensive_health_check()  # Populate issues
        
        recovery_plan = monitor.create_recovery_plan()
        
        print("\n🏥 Recovery Plan:")
        print("=" * 50)
        print(f"Issues detected: {recovery_plan['issues_summary']}")
        print(f"Estimated recovery time: {recovery_plan['estimated_time']}")
        
        if recovery_plan["recovery_steps"]:
            print(f"\n📋 Recovery Steps:")
            for i, step in enumerate(recovery_plan["recovery_steps"], 1):
                print(f"   {i}. {step['action']}")
                print(f"      Command: {step['command']}")
                print(f"      Time: {step['estimated_time']}")
        
        if args.execute:
            print(f"\n🚀 Executing auto-fixable issues...")
            monitor._auto_fix_issues()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()