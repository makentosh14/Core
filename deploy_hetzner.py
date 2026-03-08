#!/usr/bin/env python3
"""
deploy_hetzner.py - Hetzner Cloud Deployment Helper
=====================================================
Sets up the scanner as a systemd service on Hetzner VPS.

Usage:
    python deploy_hetzner.py setup     # Create service files
    python deploy_hetzner.py start     # Start the scanner
    python deploy_hetzner.py status    # Check status
    python deploy_hetzner.py logs      # View logs
"""

import os
import sys
import subprocess

SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_NAME = "crypto-scanner"
PYTHON_PATH = sys.executable

SERVICE_FILE = f"""[Unit]
Description=Advanced Crypto Market Scanner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={os.environ.get('USER', 'root')}
WorkingDirectory={SCANNER_DIR}
ExecStart={PYTHON_PATH} {SCANNER_DIR}/scanner_main.py --loop --top 100 --interval 300
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# Performance tuning for Hetzner
LimitNOFILE=65536
MemoryMax=2G

[Install]
WantedBy=multi-user.target
"""

REQUIREMENTS = """# requirements.txt - Scanner dependencies
aiohttp>=3.9.0
numpy>=1.24.0
"""


def cmd(command: str, check: bool = True):
    """Run a shell command."""
    print(f"  $ {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        print(f"  Command failed with exit code {result.returncode}")
    return result


def setup():
    """Set up the scanner as a systemd service."""
    print(f"\n  Setting up {SERVICE_NAME} on Hetzner...\n")

    # Write requirements
    req_path = os.path.join(SCANNER_DIR, "requirements.txt")
    with open(req_path, "w") as f:
        f.write(REQUIREMENTS)
    print(f"  Created {req_path}")

    # Install dependencies
    print("\n  Installing Python dependencies...")
    cmd(f"{PYTHON_PATH} -m pip install -r {req_path} --break-system-packages")

    # Write systemd service
    service_path = f"/etc/systemd/system/{SERVICE_NAME}.service"
    with open(service_path, "w") as f:
        f.write(SERVICE_FILE)
    print(f"\n  Created {service_path}")

    # Reload systemd
    cmd("systemctl daemon-reload")
    cmd(f"systemctl enable {SERVICE_NAME}")

    print(f"\n  Setup complete!")
    print(f"  Start with: python deploy_hetzner.py start")
    print(f"  Or:         systemctl start {SERVICE_NAME}")


def start():
    cmd(f"systemctl start {SERVICE_NAME}")
    print(f"  {SERVICE_NAME} started. Check: python deploy_hetzner.py status")


def stop():
    cmd(f"systemctl stop {SERVICE_NAME}")
    print(f"  {SERVICE_NAME} stopped.")


def status():
    cmd(f"systemctl status {SERVICE_NAME}", check=False)


def logs():
    cmd(f"journalctl -u {SERVICE_NAME} -f --no-pager -n 100", check=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: python deploy_hetzner.py [setup|start|stop|status|logs]")
        return

    action = sys.argv[1].lower()
    actions = {
        "setup": setup,
        "start": start,
        "stop": stop,
        "status": status,
        "logs": logs,
    }

    if action in actions:
        actions[action]()
    else:
        print(f"Unknown action: {action}")
        print(f"Available: {', '.join(actions.keys())}")


if __name__ == "__main__":
    main()
