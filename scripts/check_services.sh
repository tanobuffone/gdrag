#!/bin/bash

# Service checker for gdrag
# Checks required services: PostgreSQL, Qdrant, and optionally Memgraph

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
TIMEOUT=5
POSTGRES_HOSTS=("ws-postgres" "localhost")
POSTGRES_PORT=5432
QDRANT_HOST="localhost"
QDRANT_PORT=6333
MEMGRAPH_HOST="localhost"
MEMGRAPH_PORT=7687

# Exit code tracker
CRITICAL_FAILURE=0

# Function to check if a service is reachable
check_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local is_critical=$4

    # Use timeout and /dev/tcp to check if port is open
    if timeout $TIMEOUT bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null; then
        echo -e "${GREEN}[OK]${NC} $service_name ($host:$port)"
        return 0
    else
        if [ "$is_critical" = "true" ]; then
            echo -e "${RED}[FAIL]${NC} $service_name ($host:$port)"
            CRITICAL_FAILURE=1
        else
            echo -e "${YELLOW}[WARNING]${NC} $service_name ($host:$port) - Optional service unavailable"
        fi
        return 1
    fi
}

echo "Checking gdrag required services..."
echo "================================="

# Check PostgreSQL (try localhost first, then ws-postgres)
POSTGRES_OK=false
for host in "${POSTGRES_HOSTS[@]}"; do
    if check_service "$host" "$POSTGRES_PORT" "PostgreSQL" "false"; then
        POSTGRES_OK=true
        break
    fi
done

# Mark PostgreSQL as critical failure only if no host worked
if [ "$POSTGRES_OK" = "false" ]; then
    echo -e "${RED}[FAIL]${NC} PostgreSQL - No available host found"
    CRITICAL_FAILURE=1
fi

# Check Qdrant
check_service "$QDRANT_HOST" "$QDRANT_PORT" "Qdrant" "true"

# Check Memgraph (optional)
check_service "$MEMGRAPH_HOST" "$MEMGRAPH_PORT" "Memgraph" "false"

echo "================================="

# Summary and exit code
if [ $CRITICAL_FAILURE -eq 0 ]; then
    echo -e "${GREEN}All critical services are running.${NC}"
    exit 0
else
    echo -e "${RED}One or more critical services are not available.${NC}"
    exit 1
fi
