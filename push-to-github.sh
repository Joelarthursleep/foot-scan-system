#!/bin/bash

# Push Foot Scan System to GitHub
# This script will help you push your code to GitHub

echo "========================================="
echo "Push to GitHub: foot-scan-system"
echo "========================================="
echo ""

# Check if remote exists
if git remote | grep -q "origin"; then
    echo "✓ Remote 'origin' already configured"
else
    echo "Adding remote..."
    git remote add origin https://github.com/Joelarthursleep/foot-scan-system.git
fi

echo ""
echo "Pushing to GitHub..."
echo "You may be prompted for your GitHub credentials."
echo ""

# Configure credential helper for macOS
git config credential.helper osxkeychain

# Push to GitHub
git push -u origin main

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Successfully pushed to GitHub!"
    echo "========================================="
    echo ""
    echo "Your repository is now at:"
    echo "https://github.com/Joelarthursleep/foot-scan-system"
    echo ""
    echo "Next step: Deploy to Streamlit Cloud"
    echo "1. Go to: https://share.streamlit.io"
    echo "2. Sign in with GitHub"
    echo "3. Click 'New app'"
    echo "4. Select your repository: Joelarthursleep/foot-scan-system"
    echo "5. Set main file: app.py"
    echo "6. Deploy!"
    echo ""
else
    echo ""
    echo "========================================="
    echo "❌ Push failed!"
    echo "========================================="
    echo ""
    echo "If you see an authentication error, you need a Personal Access Token:"
    echo ""
    echo "1. Go to: https://github.com/settings/tokens/new"
    echo "2. Generate a new token with 'repo' scope"
    echo "3. Copy the token"
    echo "4. Run this script again"
    echo "5. When prompted for password, paste your token (not your GitHub password)"
    echo ""
fi
