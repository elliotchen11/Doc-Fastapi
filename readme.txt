This project provides DocFlow backend API using FASTAPI 

# Set up virtual env
python -m venv .ocrenv
# Windows activate env
.ocrenv\Scripts\activate
# Linux activate env
source .ocrenv/bin/activate

# Install dependencies
pip install -r requirements2.txt

# Run the server
uvicorn app.main:app --reload

# Check API health
http://localhost:8000/health

# Doc 
http://localhost:8000/docs#