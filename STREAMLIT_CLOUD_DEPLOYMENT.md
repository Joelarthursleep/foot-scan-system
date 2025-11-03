# Streamlit Cloud Deployment Guide

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository details:
   - **Name**: `foot-scan-system`
   - **Description**: "Medical-grade foot scanning and analysis system with AI-powered risk assessment"
   - **Visibility**: Choose **Private** (recommended for medical data) or Public
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

## Step 2: Push Code to GitHub

After creating the repository, run these commands in your terminal:

```bash
cd /Users/joellewis/foot-scan-system

# Add GitHub as remote (replace YOUR-USERNAME with your actual GitHub username)
git remote add origin https://github.com/YOUR-USERNAME/foot-scan-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your actual password)
  - Generate one at: https://github.com/settings/tokens/new
  - Select scopes: `repo` (full control of private repositories)
  - Copy the token and use it as your password

## Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io

2. **Sign in with GitHub**: Click "Continue with GitHub"

3. **Authorize Streamlit Cloud**: Grant access to your repositories

4. **Create New App**:
   - Click "New app" button
   - Select your repository: `YOUR-USERNAME/foot-scan-system`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: Choose a custom URL (e.g., `foot-scan-system`)

5. **Advanced Settings** (click "Advanced settings" before deploying):
   - **Python version**: 3.10
   - **Secrets**: Leave empty (no secrets needed for now)

6. **Click "Deploy!"**

## Step 4: Wait for Deployment

- Initial deployment takes 5-10 minutes
- Streamlit Cloud will:
  - Install all dependencies from requirements.txt
  - Start your app
  - Provide you with a public URL

## Your App URL

After deployment, your app will be available at:
```
https://YOUR-APP-NAME.streamlit.app
```

## Post-Deployment

### Update Your App
Any time you push changes to GitHub, Streamlit Cloud will automatically redeploy:

```bash
# Make your changes
git add .
git commit -m "Your commit message"
git push
```

### Monitor Your App
- View logs: https://share.streamlit.io/
- Check resource usage
- See deployment history

### Custom Domain (Optional)
You can add a custom domain in the Streamlit Cloud settings.

## Troubleshooting

### Build Fails
- Check the logs in Streamlit Cloud dashboard
- Verify requirements.txt has all dependencies
- Ensure Python version is 3.10+

### App Crashes
- Check if data files are too large (GitHub limit: 100MB per file)
- Review Streamlit Cloud logs for errors
- Check memory usage (free tier: 1GB RAM limit)

### Slow Performance
- Free tier has resource limits
- Consider upgrading to paid plan for better performance
- Optimize your code (cache expensive operations)

## Current Configuration

Your app is already configured with:
- ✅ requirements.txt with all dependencies
- ✅ .gitignore to exclude unnecessary files
- ✅ SQLite databases included in repo
- ✅ Pre-trained ML models included
- ✅ Sample scan data for demonstration

## Data Persistence Note

Streamlit Cloud's file system is **ephemeral** - any data written during runtime will be lost on restart. For production:

1. Use Streamlit Cloud's secrets management for API keys
2. Consider external database (e.g., Supabase, PlanetScale) for persistent storage
3. Use cloud storage (e.g., AWS S3, Google Cloud Storage) for uploaded files

## Free Tier Limits

- **1GB RAM**
- **1 CPU core**
- **Public apps only** (unless you upgrade)
- **Sleep after inactivity** (app wakes up on visit)

For production use with sensitive medical data, consider:
- Upgrading to Streamlit Cloud Teams ($250/month)
- Or deploying to Google Cloud Run with authentication

## Support

- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud
- Community Forum: https://discuss.streamlit.io/
- Status Page: https://status.streamlit.io/

---

Need help? Check the Streamlit Community Cloud documentation or reach out on the forum.
