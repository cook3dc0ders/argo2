#!/bin/bash
# deploy.sh - Quick deployment script for ARGO RAG Dashboard

set -e

echo "🌊 ARGO RAG Dashboard - Quick Deploy Script"
echo "==========================================="

# Check if we're in the right repository
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found. Are you in the right directory?"
    echo "Please run this script from your argo2 repository root."
    exit 1
fi

echo "✅ Found streamlit_app.py"

# Create necessary directories
echo "📁 Creating deployment directories..."
mkdir -p .streamlit
mkdir -p data

# Create Streamlit config if it doesn't exist
if [ ! -f ".streamlit/config.toml" ]; then
    cat > .streamlit/config.toml << EOF
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
EOF
    echo "✅ Created .streamlit/config.toml"
fi

# Create packages.txt for system dependencies
if [ ! -f "packages.txt" ]; then
    cat > packages.txt << EOF
libpq-dev
python3-dev
build-essential
EOF
    echo "✅ Created packages.txt"
fi

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    cp requirements_cloud.txt requirements.txt
    echo "✅ Created requirements.txt from requirements_cloud.txt"
fi

# Add sample data to dataset repo directory
if [ ! -f "data/sample_profiles.json" ]; then
    echo "📊 Creating sample data file..."
    # Copy the sample_profiles.json we created earlier
    echo "Please manually add sample_profiles.json to your cook3dc0ders/dataset repository"
fi

# Check git status
echo "📝 Git repository status:"
git status --porcelain

# Ask user if they want to commit and push
echo ""
read -p "Do you want to commit and push these changes to GitHub? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Committing changes..."
    git add .
    git commit -m "Deploy: Add cloud-optimized dashboard and API

- Add streamlit_app.py for Streamlit Cloud deployment
- Add cloud_api.py for Railway backend deployment
- Add deployment configuration files
- Add sample data population script
- Configure for free tier hosting"

    echo "📤 Pushing to GitHub..."
    git push origin main
    echo "✅ Changes pushed to GitHub"
else
    echo "⏭️ Skipping git commit. You can commit manually when ready."
fi

echo ""
echo "🎉 Deployment preparation complete!"
echo ""
echo "Next Steps:"
echo "=========="
echo ""
echo "1. DATABASE (Railway):"
echo "   • Go to https://railway.app"
echo "   • Create new project → Add PostgreSQL"
echo "   • Copy DATABASE_URL from variables"
echo "   • Run: python populate_railway_db.py [DATABASE_URL]"
echo ""
echo "2. API (Railway):"
echo "   • In same project → New Service → GitHub Repo"
echo "   • Select your cook3dc0ders/argo2 repository"
echo "   • Add DATABASE_URL to environment variables"
echo "   • Test at: https://your-api.railway.app/health"
echo ""
echo "3. FRONTEND (Streamlit Cloud):"
echo "   • Go to https://share.streamlit.io"
echo "   • New app → cook3dc0ders/argo2 → streamlit_app.py"
echo "   • Add secrets with DATABASE_URL and API URL"
echo "   • Deploy and test"
echo ""
echo "4. DATASET (GitHub):"
echo "   • Add sample_profiles.json to cook3dc0ders/dataset repo"
echo "   • Ensure it's publicly accessible"
echo ""
echo "🔗 Your live URLs will be:"
echo "   Dashboard: https://your-app.streamlit.app"
echo "   API: https://your-api.railway.app"
echo "   Dataset: https://github.com/cook3dc0ders/dataset"
echo ""
echo "Need help? Check DEPLOYMENT_GUIDE.md for detailed instructions."
