#!/usr/bin/env python3
"""
Bot Log Cleaner - Safely empties bot log files without interrupting the bot's operation
Compatible with Hetzner Cloud environments
"""

import os
import sys
import glob
import time
import argparse
from pathlib import Path
from typing import List, Optional
import fcntl
import errno

class BotLogCleaner:
    def __init__(self, log_directory: str = "./logs", backup: bool = False):
        """
        Initialize the log cleaner.
        
        Args:
            log_directory: Directory containing log files
            backup: Whether to create backups before clearing
        """
        self.log_directory = Path(log_directory)
        self.backup = backup
        self.cleared_files = []
        self.failed_files = []
    
    def find_log_files(self, patterns: List[str] = None) -> List[Path]:
        """
        Find all log files matching the specified patterns.
        
        Args:
            patterns: List of file patterns to match (e.g., ['*.log', '*.txt'])
        
        Returns:
            List of Path objects for found log files
        """
        if patterns is None:
            patterns = ['*.log', '*.txt', '*.err', '*.out']
        
        log_files = []
        
        if not self.log_directory.exists():
            print(f"Warning: Log directory '{self.log_directory}' does not exist!")
            return log_files
        
        for pattern in patterns:
            log_files.extend(self.log_directory.glob(pattern))
        
        return log_files
    
    def safe_truncate_file(self, file_path: Path) -> bool:
        """
        Safely truncate a log file using file locking to prevent conflicts.
        This method ensures the bot can continue writing to the file.
        
        Args:
            file_path: Path to the log file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open file in write mode but don't truncate immediately
            with open(file_path, 'r+') as file:
                # Try to acquire a non-blocking exclusive lock
                try:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # If we got the lock, truncate the file
                    file.truncate(0)
                    file.flush()
                    
                    # Release the lock
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                    
                    return True
                    
                except IOError as e:
                    if e.errno in (errno.EAGAIN, errno.EACCES):
                        # File is locked by another process (likely your bot)
                        # Try alternative method: write empty content
                        file.seek(0)
                        file.truncate(0)
                        file.flush()
                        return True
                    else:
                        raise
                        
        except PermissionError:
            print(f"Permission denied: Cannot modify {file_path}")
            return False
        except Exception as e:
            print(f"Error clearing {file_path}: {str(e)}")
            return False
    
    def backup_file(self, file_path: Path) -> Optional[Path]:
        """
        Create a backup of the log file before clearing.
        
        Args:
            file_path: Path to the log file
        
        Returns:
            Path to backup file if successful, None otherwise
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        
        try:
            # Read original file content
            with open(file_path, 'rb') as original:
                content = original.read()
            
            # Write to backup
            with open(backup_path, 'wb') as backup:
                backup.write(content)
            
            print(f"Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            print(f"Failed to create backup for {file_path}: {str(e)}")
            return None
    
    def clear_log_file(self, file_path: Path) -> bool:
        """
        Clear a single log file with optional backup.
        
        Args:
            file_path: Path to the log file
        
        Returns:
            True if successful, False otherwise
        """
        # Get file size before clearing
        try:
            original_size = file_path.stat().st_size
            if original_size == 0:
                print(f"Skipping {file_path.name} - already empty")
                return True
        except:
            original_size = 0
        
        # Create backup if requested
        if self.backup and original_size > 0:
            backup_path = self.backup_file(file_path)
            if backup_path is None and original_size > 1024 * 1024:  # Only fail for large files
                print(f"Skipping {file_path.name} - backup failed for large file")
                return False
        
        # Clear the file
        if self.safe_truncate_file(file_path):
            size_str = self.format_size(original_size)
            print(f"✓ Cleared {file_path.name} ({size_str} freed)")
            self.cleared_files.append(file_path)
            return True
        else:
            print(f"✗ Failed to clear {file_path.name}")
            self.failed_files.append(file_path)
            return False
    
    def format_size(self, bytes: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} TB"
    
    def clear_all_logs(self, patterns: List[str] = None, exclude: List[str] = None):
        """
        Clear all log files matching the patterns.
        
        Args:
            patterns: List of file patterns to match
            exclude: List of file names to exclude from clearing
        """
        if exclude is None:
            exclude = []
        
        log_files = self.find_log_files(patterns)
        
        if not log_files:
            print("No log files found to clear.")
            return
        
        print(f"\nFound {len(log_files)} log file(s) to process:")
        print("-" * 50)
        
        total_size_freed = 0
        
        for log_file in log_files:
            # Skip excluded files
            if log_file.name in exclude:
                print(f"Skipping excluded file: {log_file.name}")
                continue
            
            # Get size before clearing
            try:
                size_before = log_file.stat().st_size
            except:
                size_before = 0
            
            # Clear the file
            if self.clear_log_file(log_file):
                total_size_freed += size_before
        
        # Print summary
        print("-" * 50)
        print(f"\nSummary:")
        print(f"  Files cleared: {len(self.cleared_files)}")
        print(f"  Files failed: {len(self.failed_files)}")
        print(f"  Total space freed: {self.format_size(total_size_freed)}")
        
        if self.failed_files:
            print(f"\nFailed files:")
            for file in self.failed_files:
                print(f"  - {file.name}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Safely clear bot log files without interrupting bot operation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 clear_bot_logs.py                    # Clear all logs in ./logs directory
  python3 clear_bot_logs.py -d /var/log/mybot  # Specify log directory
  python3 clear_bot_logs.py -b                 # Create backups before clearing
  python3 clear_bot_logs.py -p "*.log" "*.err" # Only clear specific file types
  python3 clear_bot_logs.py -e "important.log" # Exclude specific files
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='./logs',
        help='Directory containing log files (default: ./logs)'
    )
    
    parser.add_argument(
        '-b', '--backup',
        action='store_true',
        help='Create backup before clearing files'
    )
    
    parser.add_argument(
        '-p', '--patterns',
        nargs='+',
        default=['*.log', '*.txt', '*.err', '*.out'],
        help='File patterns to match (default: *.log *.txt *.err *.out)'
    )
    
    parser.add_argument(
        '-e', '--exclude',
        nargs='+',
        default=[],
        help='File names to exclude from clearing'
    )
    
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("Bot Log Cleaner")
    print("=" * 50)
    print(f"Log directory: {args.directory}")
    print(f"File patterns: {', '.join(args.patterns)}")
    print(f"Backup enabled: {args.backup}")
    if args.exclude:
        print(f"Excluded files: {', '.join(args.exclude)}")
    
    # Ask for confirmation unless -y flag is used
    if not args.yes:
        response = input("\nDo you want to proceed? (y/n): ").lower()
        if response != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    # Create cleaner instance and run
    cleaner = BotLogCleaner(
        log_directory=args.directory,
        backup=args.backup
    )
    
    try:
        cleaner.clear_all_logs(
            patterns=args.patterns,
            exclude=args.exclude
        )
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
