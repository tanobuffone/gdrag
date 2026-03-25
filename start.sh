#!/bin/bash
# gdrag v2 - Quick Start Script

set -e

echo "🚀 Starting gdrag v2..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env with your configuration before running again."
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.12+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Check database connections
echo "🔍 Checking database connections..."

# Check PostgreSQL
if command -v psql &> /dev/null; then
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1" &> /dev/null; then
        echo "✅ PostgreSQL connected"
    else
        echo "⚠️  PostgreSQL not connected. Make sure it's running."
    fi
else
    echo "⚠️  psql not found. Skipping PostgreSQL check."
fi

# Check Qdrant
if curl -s "http://$QDRANT_HOST:$QDRANT_PORT/collections" &> /dev/null; then
    echo "✅ Qdrant connected"
else
    echo "⚠️  Qdrant not connected. Make sure it's running on $QDRANT_HOST:$QDRANT_PORT"
fi

# Start the API server
echo ""
echo "🎯 Starting API server..."
echo "   Host: $GDRAG_API_HOST"
echo "   Port: $GDRAG_API_PORT"
echo "   Docs: http://localhost:$GDRAG_API_PORT/docs"
echo ""

uvicorn src.api.main:app \
    --host "$GDRAG_API_HOST" \
    --port "$GDRAG_API_PORT" \
    --reload