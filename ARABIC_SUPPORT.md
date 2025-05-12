# Arabic Language Support

This document outlines the Arabic language support features in the bilingual chatbot and provides information about compatibility issues and solutions.

## Status of Arabic Language Support

The chatbot has been successfully enhanced with comprehensive Arabic language capabilities:

- **Language Detection**: Successfully identifies Arabic text (using langdetect)
- **Profanity Filtering**: Works correctly with both English and Arabic content (using better_profanity and custom Arabic word lists)
- **Text Preprocessing**: Properly handles Arabic text normalization, including diacritics removal and letter form standardization (using PyArabic)
- **AraBERT Support**: Optional advanced Arabic processing when AraBERT is available, with graceful fallbacks when not

## Python 3.12 Compatibility

There are some compatibility issues with Python 3.12 and the SSL/asyncio modules that affect the full chatbot functionality. The consolidated test script includes workarounds for these issues through environment variable settings.

## How to Test Arabic Support

To test Arabic language functionality, run the consolidated test script:

```bash
python arabic_test.py
```

This script tests:
- Arabic language detection
- Profanity filtering (using better_profanity with Arabic words)
- Arabic text preprocessing (using PyArabic)
- Basic Arabic string operations

The test script uses environment variables to avoid SSL compatibility issues and provides a comprehensive verification of all Arabic language support features.

## Dependencies for Arabic Support

The following dependencies are required for Arabic language support:

```
PyArabic
better_profanity
langdetect
```

Optional advanced dependencies:
```
arabert
farasapy
transformers
tensorflow
```

## Environment Variables

- `ENABLE_ARABIC`: Set to "true" to enable Arabic language support (default: true)
- `SIMILARITY_THRESHOLD_AR`: Similarity threshold for Arabic queries (default: 0.3)
- `SIMILARITY_THRESHOLD_EN`: Similarity threshold for English queries (default: 0.4)
- `PYTHONHTTPSVERIFY`: Set to "0" to bypass SSL verification issues (for testing only)

## Known Issues

1. **SSL Compatibility**: Python 3.12 has changes to SSL module that can cause issues with asyncio and FastAPI
2. **AraBERT Installation**: AraBERT may fail to install on some systems due to its dependencies
3. **Language Detection Accuracy**: The language detection can sometimes misidentify languages, though it reliably identifies Arabic
4. **Arabic Profanity Detection**: Requires explicit addition of Arabic words to the profanity filter's custom word list

## Future Improvements

1. Improve error handling for missing AraBERT components
2. Create a more robust language detection solution
3. Expand the Arabic profanity list with regional variations
4. Add dialectal Arabic support for different regions
5. Implement a more comprehensive setup script for Arabic language dependencies 