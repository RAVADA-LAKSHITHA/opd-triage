.PHONY: install data train test run docker
install:; pip install -r requirements.txt
data:; python -m src.triage.data
train:; python -m src.triage.evaluate
test:; pytest -q
run: train; streamlit run app.py
docker:; docker build -t opd-triage . && docker run -p 8501:8501 opd-triage
