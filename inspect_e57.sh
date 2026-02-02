#!/bin/bash

echo "==============================================="
echo "   E57 STRUCTURE INSPECTION AUTOMATION SCRIPT"
echo "==============================================="

# Check file argument
if [ -z "$1" ]; then
    echo "❌ ERROR: No E57 file provided."
    echo "Usage: ./inspect_e57.sh \"path/to/file.e57\""
    exit 1
fi

E57FILE="$1"

if [ ! -f "$E57FILE" ]; then
    echo "❌ ERROR: File not found:"
    echo "   $E57FILE"
    exit 1
fi

echo "👉 Inspecting E57 file:"
echo "   $E57FILE"
echo ""

echo "-----------------------------------------------"
echo "[1] Updating apt…"
sudo apt update -y

echo "-----------------------------------------------"
echo "[2] Installing E57 tools…"
sudo apt install -y libe57format-dev e57-tools

echo "-----------------------------------------------"
echo "[3] Checking e57dump…"
if ! command -v e57dump &> /dev/null; then
    echo "❌ ERROR: e57dump did not install"
    exit 1
fi

echo "-----------------------------------------------"
echo "[4] Dumping E57 metadata → e57_info.txt"
e57dump "$E57FILE" > e57_info.txt

echo "✔ DONE: Saved to e57_info.txt"
echo ""

echo "-----------------------------------------------"
echo "[5] Showing first 200 lines:"
head -n 200 e57_info.txt

echo ""
echo "==============================================="
echo "   DONE. Paste the output above to ChatGPT."
echo "==============================================="
