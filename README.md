# LangChain PDF & Word Reader

Upload a PDF or Word document, get an instant summary, and chat with it. Answers are grounded in the document and cite the page numbers they came from. Built with Streamlit, LangChain and OpenAI chat models.

![Image](Screenshot.png)

## Features

- **PDF and DOCX** – Word paragraphs and tables are both extracted.
- **Automatic summary** – small documents are summarised in one call; long ones use a map-reduce pass so there is no context-length limit.
- **Chat with your document** – multi-turn Q&A that remembers recent questions.
- **Page citations** – every answer lists the source pages, with an expander showing the exact excerpts used.
- **Indexed once per file** – embeddings are computed only when a new file (or chunk size) is uploaded, not on every question.
- **Bring your own key** – enter your OpenAI API key in the sidebar, or configure it via secrets / environment variable.
- **Model choice and tuning** – pick `gpt-4o-mini`, `gpt-4.1-mini` or `gpt-4o`, and adjust chunk size and number of excerpts.
- **Token usage** – the sidebar shows chat tokens used this session.
- Clear error for scanned PDFs that contain no selectable text.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/lenkazuma/PDFReader.git
cd PDFReader
pip install -r requirements.txt
```

## Configuration

The OpenAI API key can be provided in any of these ways (first match wins):

1. Typed into the sidebar (kept only in your browser session);
2. `OPENAI_API_KEY` in `.streamlit/secrets.toml` (used by Streamlit Community Cloud);
3. `OPENAI_API_KEY` environment variable or `.env` file.

Get a key at <https://platform.openai.com/api-keys>. Never commit keys to the repository.

## Usage

```bash
streamlit run streamlit_app.py
```

Upload a file, read the summary, then ask questions in the chat box.

## Deployment

A deployed version is available at [PDFReader-LongChain](https://pdfreader-longchain.streamlit.app/) (yes, "LangChain" is misspelt in the URL).

To deploy your own on [Streamlit Community Cloud](https://streamlit.io/cloud), point it at `streamlit_app.py` and add `OPENAI_API_KEY` under **App settings → Secrets**, or leave it empty so each visitor enters their own key.

## Project structure

```
├── streamlit_app.py   # Streamlit UI
├── reader_core.py     # parsing, chunking, indexing, Q&A and summarisation
└── tests/             # pytest tests (no API key needed)
```

## Tests

```bash
pip install pytest
pytest -q
```

## Acknowledgment

Inspired by [Alejandro AO's langchain-ask-pdf](https://github.com/alejandro-ao/langchain-ask-pdf) and [his tutorial](https://www.youtube.com/watch?v=wUAUdEw5oxM).

## Roadmap

- OCR text extraction for scanned PDFs in different languages.

## License

[MIT](LICENSE)
