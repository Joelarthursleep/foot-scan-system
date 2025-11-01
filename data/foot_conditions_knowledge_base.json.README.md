# Foot Conditions Knowledge Base

The full knowledge base file (`foot_conditions_knowledge_base.json`) is 136MB and exceeds GitHub's file size limits.

## For Local Development

The file is already present in your local installation at:
```
data/foot_conditions_knowledge_base.json
```

## For Streamlit Cloud Deployment

The application will generate a lightweight version automatically on first run if the full file is not present. The system includes:

- Evidence-based medical condition detection
- Clinical study references
- ICD-10 coding
- Treatment recommendations

## Regenerating the Knowledge Base

If you need to regenerate the full knowledge base, run:

```python
from src.features.medical_research_loader import MedicalResearchLoader
loader = MedicalResearchLoader()
loader.load_research_database()
```

This will create a comprehensive medical knowledge base with 100+ clinical studies and research papers.

## File Size Information

- Full file: 136 MB
- Contains: 100+ clinical studies, diagnostic criteria, treatment protocols
- Format: JSON with structured medical research data

## Alternative Hosting

For production deployments, consider:
1. Hosting the file on cloud storage (AWS S3, Google Cloud Storage)
2. Using a CDN for faster access
3. Implementing lazy loading for specific conditions
