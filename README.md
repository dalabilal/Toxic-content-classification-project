# Toxic Content Classification

A Streamlit app that classifies text or images into toxic content categories.

- **Text input:** the text is classified directly.
- **Image input:** BLIP generates a caption for the image, then the caption is classified.
- Every input and its result is saved in a SQLite database and can be viewed in the app.

## Live Demo

https://toxic-content-classification-projectgit-cc4t2hunv9eytrwcuuapph.streamlit.app/

## How It Works

```
Image  -> BLIP-1 caption --\
                            +--> LSTM classifier --> SQLite database
User text -----------------/
```

## Project Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app (text, image, and database pages) |
| `imagecaption.py` | Image captioning with BLIP-1 |
| `classifier.py` | Loads the trained LSTM models and predicts the category |
| `database.py` | SQLite database functions |
| `train_LSTM.py` | Script used to train the LSTM |
| `cellula_toxic_data.csv` | Training dataset |
| `requirements.txt` | Python libraries |

## Installation

1. Clone the repository:

   ```bash
   git clone <your-repository-link>
   cd <repository-folder>
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac / Linux
   ```

3. Install the libraries from the requirements file:

   ```bash
   pip install -r requirements.txt
   ```

## Run the App

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

**Notes:**
- The first time you caption an image, the BLIP model is downloaded from Hugging Face, so you need an internet connection.
- The trained model files must be in the `artifacts_lstm/` folder. To create them again, run `python train_LSTM.py`.

## Database

Results are stored in `toxic_logs.db` (SQLite) with these columns: `id`, `created_at`, `input_type`, `content`, `predicted_label`, `confidence`. You can see all records in the **View database** page of the app and download them as a CSV file.
