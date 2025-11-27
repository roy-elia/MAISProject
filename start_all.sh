#!/bin/bash
# Script to start backend, frontend, and ngrok for the Reddit Year Prediction project

echo "🚀 Starting Reddit Year Prediction Services..."
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected. Activate it first:"
    echo "   source .venv/bin/activate"
    echo ""
fi

# Start backend in background
echo "📡 Starting backend on port 8000..."
cd "$(dirname "$0")"
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 2

# Start frontend in background
echo "🎨 Starting frontend on port 3000..."
cd frontend
npm run dev > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

# Wait a moment for frontend to start
sleep 3

# Start ngrok
echo "🌐 Starting ngrok tunnel..."
cd ..
ngrok http 3000

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    pkill -f "ngrok http" 2>/dev/null
    echo "✅ All services stopped"
    exit
}

# Trap Ctrl+C to cleanup
trap cleanup SIGINT SIGTERM

# Wait for user to stop
echo ""
echo "✅ All services running!"
echo "   Backend: http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   ngrok: Check the URL above"
echo ""
echo "Press Ctrl+C to stop all services"
wait

