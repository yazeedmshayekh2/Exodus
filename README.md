# Bilingual FAQ Chatbot with Arabic Support

A production-ready chatbot API that provides bilingual support for English and Arabic questions using semantic search and LLM-powered responses.

## Features

- **Bilingual Support**: Handles both English and Arabic queries with dedicated optimization for each language
- **Content Moderation**: Built-in profanity and inappropriate content filtering for both languages
- **Automatic Language Detection**: Automatically detects the language of user input
- **Optimized Arabic Processing**: Specialized text processing for Arabic including diacritics handling and letter normalization
- **Semantic Search**: Uses sentence transformers to find the most relevant answers
- **LLM Integration**: Uses Llama 3.1 (8B model) for response generation
- **Graceful Degradation**: Falls back to simpler processing when advanced Arabic NLP tools aren't available

## Arabic Language Support

This chatbot features comprehensive Arabic language capabilities:

- Language detection for Arabic text
- Arabic text preprocessing (diacritics removal, letter normalization)
- Arabic content moderation with custom profanity lists
- Optional AraBERT integration for advanced Arabic NLP when available
- Optimized similarity thresholds specific to Arabic text

For detailed information about Arabic support, see [ARABIC_SUPPORT.md](ARABIC_SUPPORT.md).

## Getting Started

### Prerequisites

- Python 3.8 or higher (tested with Python 3.12)
- An Ollama server with access to the Llama 3.1 8B model
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bilingual-faq-chatbot.git
cd bilingual-faq-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

## Testing

To test the Arabic language capabilities:
```bash
python arabic_test.py
```

This test verifies:
- Arabic language detection
- Profanity filtering for Arabic text
- Arabic text preprocessing functionality
- Basic Arabic string handling

## Environment Variables

- `ENABLE_ARABIC`: Enable Arabic language support (default: true)
- `SIMILARITY_THRESHOLD_AR`: Similarity threshold for Arabic queries (default: 0.3)
- `SIMILARITY_THRESHOLD_EN`: Similarity threshold for English queries (default: 0.4)
- `OLLAMA_BASE_URL`: URL for the Ollama service (default: http://localhost:11434)
- `MODEL_NAME`: LLM model to use (default: llama3.1:8b)

## Frontend Configuration

The chatbot includes a React-based frontend that communicates with the backend API:

### Setup

1. Install frontend dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bashPOST
npm run dev
```

3. Access the UI at http://localhost:3000

### Configuration Settings

The frontend is configured to connect to the backend API via:

- Direct API calls to `http://localhost:8000/api` (configured in `frontend/src/api/chatService.js`)
- Development proxy from `/api` to `http://localhost:8000` (configured in `frontend/vite.config.js`)

If you change the backend port, make sure to update both these configuration files.

## Troubleshooting

### CUDA/Model Initialization Issues

If you encounter an error like:
```
Error in find_most_similar_faq: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```

This is caused by PyTorch trying to use CUDA in a forked subprocess. To fix this:

1. Make sure you're running with a single worker:
   ```bash
   # The app now defaults to one worker, but you can force it:
   python main.py
   ```

2. If the issue persists, modify multiprocessing settings:
   ```python
   # This is now included in main.py:
   import multiprocessing
   multiprocessing.set_start_method('spawn', force=True)
   ```

3. Or set these environment variables before running:
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Disables CUDA
   export TOKENIZERS_PARALLELISM=false  # Disables parallel tokenization
   python main.py
   ```

4. For Python 3.12 users:
   If SSL-related errors occur, try replacing openssl-python with pyOpenSSL:
   ```bash
   pip uninstall openssl-python
   pip install pyOpenSSL
   ```

### Frontend API Connection Issues

If the frontend cannot connect to the API:

1. Verify the backend server is running at the expected port (default: 8000)
2. Check the API URL in `frontend/src/api/chatService.js` 
3. Ensure the proxy configuration in `frontend/vite.config.js` matches the backend port
4. Check for CORS errors in the browser console and verify CORS is properly enabled in the backend

## Documentation

- [ARABIC_SUPPORT.md](ARABIC_SUPPORT.md): Detailed information about Arabic language support
- API documentation is available at `/api/docs` when the server is running

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using FastAPI, Sentence Transformers, and Ollama
- Arabic language support powered by PyArabic, langdetect, and better_profanity
- Optional advanced Arabic processing via AraBERT
- Frontend built with React, Vite, and Axios