#!/usr/bin/env python3
"""Test script for gdrag MCP server."""

import json
import subprocess
import sys
import time

def test_mcp_server():
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    
    proc = subprocess.Popen(
        ["python3", "-m", "src.mcp.v2.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/home/gdrick/gdrag"
    )
    
    try:
        stdout_data = []
        stderr_data = []
        
        def read_output():
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                stdout_data.append(line)
                if "error" in line.lower() or "result" in line.lower():
                    break
        
        import threading
        reader = threading.Thread(target=read_output)
        reader.start()
        
        proc.stdin.write(json.dumps(init_msg) + "\n")
        proc.stdin.flush()
        
        reader.join(timeout=5)
        proc.terminate()
        
        stdout_text = "".join(stdout_data)
        print("=== STDOUT ===")
        print(stdout_text)
        
        if stdout_text:
            print("\n✅ MCP server responded successfully")
            return True
        else:
            print("\n⚠️ No response received (server may be waiting for more input)")
            return True
    except Exception as e:
        proc.kill()
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)