#!/usr/bin/env python3
"""Full MCP protocol test for gdrag server."""

import json
import subprocess
import sys
import time
import threading

def test_mcp_full():
    proc = subprocess.Popen(
        ["python3", "-m", "src.mcp.v2.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/home/gdrick/gdrag"
    )
    
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    
    stdout_lines = []
    
    def read_stdout():
        for line in proc.stdout:
            stdout_lines.append(line.strip())
            print(f"< {line.strip()}")
            if len(stdout_lines) >= 2:
                break
    
    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    
    for msg in messages:
        msg_str = json.dumps(msg)
        print(f"> {msg_str}")
        proc.stdin.write(msg_str + "\n")
        proc.stdin.flush()
        time.sleep(0.5)
    
    time.sleep(3)
    proc.terminate()
    
    print(f"\n=== Received {len(stdout_lines)} responses ===")
    for line in stdout_lines:
        try:
            resp = json.loads(line)
            print(f"Response: {json.dumps(resp, indent=2)[:500]}")
        except:
            print(f"Raw: {line[:200]}")
    
    return len(stdout_lines) > 0

if __name__ == "__main__":
    success = test_mcp_full()
    print(f"\n{'✅ Success' if success else '❌ Failed'}")
    sys.exit(0 if success else 1)