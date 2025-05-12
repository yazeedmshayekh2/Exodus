"""
Content moderation utilities using established libraries for filtering inappropriate content
with enhanced AraBERT support for Arabic language processing
"""

import logging
import os
import json
import re
import sys
import subprocess
from pathlib import Path
from better_profanity import profanity
from langdetect import detect

# Configure logging
logger = logging.getLogger(__name__)

# Try to import AraBERT-related libraries
try:
    from arabert.preprocess import ArabertPreprocessor
    from farasa.segmenter import FarasaSegmenter
    import pyarabic.araby as araby
    
    ARABERT_AVAILABLE = True
    logger.info("Successfully imported AraBERT and related libraries")
except ImportError as e:
    ARABERT_AVAILABLE = False
    logger.warning(f"AraBERT import failed: {e}. Attempting to install required packages...")
    
    # Try to install missing packages if in development environment
    try:
        missing_package = str(e).split("'")[1]
        subprocess.check_call([sys.executable, "-m", "pip", "install", missing_package])
        logger.info(f"Successfully installed {missing_package}")
        
        # Try importing again
        try:
            from arabert.preprocess import ArabertPreprocessor
            from farasa.segmenter import FarasaSegmenter
            import pyarabic.araby as araby
            
            ARABERT_AVAILABLE = True
            logger.info("Successfully imported AraBERT after installation")
        except ImportError as retry_error:
            logger.error(f"AraBERT import still failed after installation: {retry_error}")
            ARABERT_AVAILABLE = False
    except Exception as install_error:
        logger.error(f"Failed to install missing package: {install_error}")
        ARABERT_AVAILABLE = False
        logger.info("Falling back to basic Arabic preprocessing")

# Initialize the profanity filter with custom settings
profanity.load_censor_words()

# Path to the Arabic profanity JSON file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARABIC_PROFANITY_FILE = os.path.join(SCRIPT_DIR, "arabic_profanity.json")

# Load comprehensive Arabic profanity list
try:
    if os.path.exists(ARABIC_PROFANITY_FILE):
        try:
            with open(ARABIC_PROFANITY_FILE, 'r', encoding='utf-8') as f:
                ARABIC_PROFANITY = json.load(f)
                logger.info(f"Loaded {len(ARABIC_PROFANITY)} Arabic profanity words from file")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            # Basic list if file is invalid
            ARABIC_PROFANITY = [
                "كس", "طيز", "زب", "شرموط", "عرص", "كلب", "خول", "منيك", "متناك", 
                "عير", "خرا", "شاذ", "لوطي", "عاهرة", "قحبة", "منيوك", "حيوان", "حقير"
            ]
            # Try to fix the malformed file
            try:
                with open(ARABIC_PROFANITY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(ARABIC_PROFANITY, f, ensure_ascii=False, indent=2)
                    logger.info("Created new Arabic profanity file with basic list")
            except Exception as write_err:
                logger.error(f"Failed to create new file: {write_err}")
    else:
        # Basic list if file not found
        ARABIC_PROFANITY = [
            "كس", "طيز", "زب", "شرموط", "عرص", "كلب", "خول", "منيك", "متناك", 
            "عير", "خرا", "شاذ", "لوطي", "عاهرة", "قحبة", "منيوك", "حيوان", "حقير"
        ]
        logger.warning("Using basic Arabic profanity list. Consider adding a comprehensive list.")
        
        # Create the directory if it doesn't exist
        try:
            os.makedirs(os.path.dirname(ARABIC_PROFANITY_FILE), exist_ok=True)
            
            # Create a basic file for future enhancement
            with open(ARABIC_PROFANITY_FILE, 'w', encoding='utf-8') as f:
                json.dump(ARABIC_PROFANITY, f, ensure_ascii=False, indent=2)
                logger.info("Created basic Arabic profanity file")
        except Exception as e:
            logger.error(f"Failed to create Arabic profanity file: {e}")
            
except Exception as e:
    logger.error(f"Error handling Arabic profanity list: {e}")
    ARABIC_PROFANITY = [
        "كس", "طيز", "زب", "شرموط", "عرص", "كلب", "خول", "منيك", "متناك", 
        "عير", "خرا", "شاذ", "لوطي", "عاهرة", "قحبة", "منيوك", "حيوان", "حقير"
    ]

# Define hate speech patterns
hate_patterns = [
    # Racism
    r'\b(whites?|blacks?|arabs?|jews?|muslims?|asians?|mexicans?|latinos?|hispanics?)\s+(are|is|all)\s+(bad|evil|stupid|dumb|lazy|violent|criminals?|terrorists?)\b',
    # Religious discrimination
    r'\b(christians?|muslims?|jews?|hindus?|buddhists?|atheists?)\s+(are|is|all)\s+(bad|evil|stupid|dumb|violent|terrorists?)\b',
    # Sexism
    r'\b(women|men|girls|boys|females?|males?)\s+(are|is|all)\s+(bad|evil|stupid|dumb|lazy|inferior|weak|objects?|property)\b',
    # Homophobia
    r'\b(gays?|lesbians?|bisexuals?|transgenders?|queers?|lgbt)\s+(are|is|all)\s+(bad|evil|sick|wrong|sinners?|perverts?|abnormal|disease)\b'
]

# Arabic hate speech patterns
arabic_hate_patterns = [
    # Racism/nationalism patterns
    r'(عرب|يهود|مسلمين|مسيحيين|اجانب|سود|بيض).*?(حقيرين|أغبياء|قذرين|إرهابيين|مجرمين)',
    # Religious discrimination
    r'(مسلمين|مسيحيين|يهود|ملحدين).*?(كفار|نجسين|ضالين|وسخين)',
    # Sexism
    r'(نساء|رجال|بنات|اولاد).*?(ناقصات|ضعفاء|أغبياء|مش كفو)',
    # Homophobia
    r'(شواذ|مثليين|لوطيين).*?(مرضى|منحرفين|شاذين)'
]

# Initialize AraBERT and related components
if ARABERT_AVAILABLE:
    try:
        # Initialize AraBERT preprocessor with the best available model
        model_names = [
            "bert-base-arabertv2",  # Try this first
            "bert-base-arabert",     # Fall back to this
            "arabic-bert-base"       # Last resort
        ]
        
        arabert_initialized = False
        farasa_initialized = False
        
        # Try models in order until one works
        for model_name in model_names:
            try:
                arabert_preprocessor = ArabertPreprocessor(model_name=model_name)
                logger.info(f"Successfully initialized ArabertPreprocessor with model: {model_name}")
                arabert_initialized = True
                break
            except Exception as model_error:
                logger.warning(f"Failed to initialize ArabertPreprocessor with model {model_name}: {model_error}")
        
        # If we couldn't initialize AraBERT with any model, set to None
        if not arabert_initialized:
            arabert_preprocessor = None
            logger.warning("Failed to initialize ArabertPreprocessor with any model")
        
        # Try to initialize Farasa segmenter
        try:
            farasa_segmenter = FarasaSegmenter()
            farasa_initialized = True
            logger.info("Successfully initialized FarasaSegmenter")
        except Exception as farasa_error:
            farasa_segmenter = None
            logger.warning(f"Failed to initialize FarasaSegmenter: {farasa_error}")
        
    except Exception as e:
        arabert_preprocessor = None
        farasa_segmenter = None
        logger.error(f"Error initializing Arabic NLP components: {e}")
else:
    arabert_preprocessor = None
    farasa_segmenter = None
    logger.info("Using basic Arabic preprocessing (without AraBERT)")

# Add custom words to the profanity filter
profanity.add_censor_words(ARABIC_PROFANITY)

# Prepare Arabic word pattern variations (for different forms of the same word)
ARABIC_VARIATIONS = {}
for word in ARABIC_PROFANITY:
    # Create variations with different diacritics and letter forms
    variations = [
        word,  # Original word
        ''.join(c for c in word if '\u0600' <= c <= '\u06ff'),  # Without diacritics
        word.replace('ا', 'أ').replace('ا', 'إ').replace('ي', 'ى').replace('ة', 'ه')  # Common replacements
    ]
    ARABIC_VARIATIONS[word] = variations

def preprocess_arabic(text: str) -> str:
    """
    Preprocess Arabic text for better content moderation using AraBERT if available.
    
    Args:
        text (str): The text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Use AraBERT if available for sophisticated preprocessing
    if arabert_preprocessor:
        try:
            return arabert_preprocessor.preprocess(text)
        except Exception as e:
            logger.error(f"Error preprocessing with ArabertPreprocessor: {e}")
    
    # Use pyarabic if available but AraBERT failed
    if ARABERT_AVAILABLE and 'araby' in globals():
        try:
            # Use pyarabic's basic normalization
            text = araby.strip_tashkeel(text)  # Remove diacritics
            text = araby.normalize_hamza(text)  # Normalize hamza
            text = araby.normalize_lamalef(text)  # Normalize lam-alef
            text = araby.strip_tatweel(text)  # Remove tatweel
            logger.info("Used pyarabic for basic normalization")
            return text
        except Exception as e:
            logger.error(f"Error preprocessing with pyarabic: {e}")
    
    # Fallback to simple normalization if all else fails
    text = re.sub('[إأآا]', 'ا', text)  # Normalize alef
    text = re.sub('ى', 'ي', text)  # Normalize ya
    text = re.sub('ة', 'ه', text)  # Normalize ta marbuta
    text = re.sub('گ', 'ك', text)  # Normalize kaf
    text = re.sub('[ؤئ]', 'ء', text)  # Normalize hamza
    
    # Remove diacritics (tashkeel)
    text = re.sub('[\u064B-\u0652]', '', text)
    
    return text

def contains_arabic_profanity(text: str) -> bool:
    """
    Advanced check for Arabic profanity with preprocessing and variation handling.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if Arabic profanity is detected
    """
    if not text:
        return False
    
    # Preprocess the text for better matching
    preprocessed_text = preprocess_arabic(text.lower())
    
    # Check for exact word matches
    for word in ARABIC_PROFANITY:
        if word in preprocessed_text:
            logger.warning(f"Arabic profanity detected: {word}")
            return True
    
    # Check with word boundaries to avoid false positives
    words = preprocessed_text.split()
    for word in words:
        if word in ARABIC_PROFANITY:
            logger.warning(f"Arabic profanity detected as word: {word}")
            return True
    
    # Check for variations with word boundaries 
    for base_word, variations in ARABIC_VARIATIONS.items():
        for variation in variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            if variation and re.search(pattern, preprocessed_text):
                logger.warning(f"Arabic profanity variation detected: {variation}")
                return True
    
    # Check for common Arabizi replacements (Latin characters for Arabic)
    arabizi_patterns = [
        # Common obscene words in Arabizi
        r'\b(kos|tiz|zeb|shar[mn]oo[t]?)\b',
        r'\b(wes[h]?[ck]h?a|wes5a|ga7ba)\b',
        r'\b([ck]alb|[ck]lab|7ayawan|7qeer)\b'
    ]
    
    for pattern in arabizi_patterns:
        if re.search(pattern, text.lower()):
            logger.warning(f"Arabizi profanity detected with pattern: {pattern}")
            return True
    
    return False

def contains_profanity(text: str) -> bool:
    """
    Check if text contains any profanity or offensive language using better_profanity.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if profanity is detected, False otherwise
    """
    if not text:
        return False
    
    # Check for profanity using better_profanity (handles English and common profanity)
    if profanity.contains_profanity(text):
        logger.warning("Profanity detected using better_profanity")
        return True
    
    # Specialized check for Arabic profanity
    if contains_arabic_profanity(text):
        return True
            
    # No profanity detected
    return False

def detect_inappropriate_patterns(text: str) -> bool:
    """
    Detect other inappropriate patterns like excessive punctuation, 
    repeated sexual innuendo, or threatening language.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if inappropriate patterns detected
    """
    # Check for excessive punctuation/shouting (sign of aggressive behavior)
    if len(re.findall(r'[!?]', text)) > 5:
        logger.warning("Excessive punctuation detected (possible aggressive content)")
        return True
    
    # Check for repeated characters (possible attempt to bypass filters)
    if re.search(r'(.)\1{4,}', text):
        logger.warning("Repeated characters detected (possible filter evasion)")
        return True
        
    # Check for threatening keywords
    threat_patterns = [
        r'\b(kill|murder|hurt|harm|attack|beat|threaten|stalk|hunt|track)\b.*\b(you|yourself|him|her|them|people)\b',
        r'\b(die|death|dead|suicide|hang|shoot)\b',
        r'\b(bomb|explosion|terror|terrorist|attack|weapon)\b'
    ]
    
    # Arabic threat patterns
    arabic_threat_patterns = [
        r'(اقتل|قتل|اموت|موت|اذبح|ذبح|اضرب|ضرب|اهجم|هجوم|تهديد|تهدد|طعن)',
        r'(سلاح|مسدس|بندقية|انتحار|متفجر|قنبلة|ارهاب|ارهابي)'
    ]
    
    for pattern in threat_patterns:
        if re.search(pattern, text.lower()):
            logger.warning("Threatening content detected")
            return True
    
    for pattern in arabic_threat_patterns:
        if re.search(pattern, text):
            logger.warning("Arabic threatening content detected")
            return True
    
    return False

def detect_hate_speech(text: str) -> bool:
    """
    Detect hate speech, slurs, and discriminatory language
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if hate speech is detected
    """
    # Patterns to detect discriminatory language
    hate_patterns = [
        # Racism
        r'\b(whites?|blacks?|arabs?|jews?|muslims?|asians?|mexicans?|latinos?|hispanics?)\s+(are|is|all)\s+(bad|evil|stupid|dumb|lazy|violent|criminals?|terrorists?)\b',
        # Religious discrimination
        r'\b(christians?|muslims?|jews?|hindus?|buddhists?|atheists?)\s+(are|is|all)\s+(bad|evil|stupid|dumb|violent|terrorists?)\b',
        # Sexism
        r'\b(women|men|girls|boys|females?|males?)\s+(are|is|all)\s+(bad|evil|stupid|dumb|lazy|inferior|weak|objects?|property)\b',
        # Homophobia
        r'\b(gays?|lesbians?|bisexuals?|transgenders?|queers?|lgbt)\s+(are|is|all)\s+(bad|evil|sick|wrong|sinners?|perverts?|abnormal|disease)\b'
    ]
    
    # Arabic hate speech patterns
    arabic_hate_patterns = [
        # Racism/nationalism patterns
        r'(عرب|يهود|مسلمين|مسيحيين|اجانب|سود|بيض).*?(حقيرين|أغبياء|قذرين|إرهابيين|مجرمين)',
        # Religious discrimination
        r'(مسلمين|مسيحيين|يهود|ملحدين).*?(كفار|نجسين|ضالين|وسخين)',
        # Sexism
        r'(نساء|رجال|بنات|اولاد).*?(ناقصات|ضعفاء|أغبياء|مش كفو)',
        # Homophobia
        r'(شواذ|مثليين|لوطيين).*?(مرضى|منحرفين|شاذين)'
    ]
    
    for pattern in hate_patterns:
        if re.search(pattern, text.lower()):
            logger.warning("Hate speech detected")
            return True
    
    for pattern in arabic_hate_patterns:
        if re.search(pattern, text):
            logger.warning("Arabic hate speech detected")
            return True
    
    return False

def detect_pii(text: str) -> bool:
    """
    Detect personally identifiable information (PII)
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if PII is detected
    """
    # Check for email addresses
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        logger.warning("Email address detected")
        return True
        
    # Check for phone numbers (various formats)
    phone_patterns = [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US/Canada: 123-456-7890
        r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',  # International: +1 123 456 7890
        r'\b\d{8,15}\b'  # Generic number sequence that could be a phone number
    ]
    
    # Arabic region specific phone patterns
    arabic_phone_patterns = [
        r'\b(05|5)\d{8}\b',  # Saudi format
        r'\b(01|1)\d{9}\b',  # Egypt format
        r'\b(07|7)\d{8}\b',  # Iraq/Jordan format
        r'\b(09|9)\d{7}\b'   # Syria/Lebanon format  
    ]
    
    for pattern in phone_patterns + arabic_phone_patterns:
        if re.search(pattern, text):
            logger.warning("Phone number detected")
            return True
            
    # Check for credit card numbers
    if re.search(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', text):
        logger.warning("Possible credit card number detected")
        return True
        
    # Check for SSN/ID numbers
    if re.search(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', text):  # SSN format
        logger.warning("Possible SSN detected")
        return True
    
    # Check for Arabic ID formats
    if re.search(r'\b\d{10}\b', text):  # Saudi ID format (10 digits)
        logger.warning("Possible Saudi ID detected")
        return True
    
    if re.search(r'\b\d{14}\b', text):  # Egyptian ID format (14 digits)
        logger.warning("Possible Egyptian ID detected")
        return True
        
    return False

def moderate_content(text: str) -> tuple:
    """
    Comprehensive moderation of content for different types of inappropriate content.
    Uses a confidence scoring system to reduce false positives.
    
    Args:
        text (str): The text to moderate
        
    Returns:
        tuple: (is_inappropriate, reason)
            - is_inappropriate (bool): True if content is inappropriate
            - reason (str): Reason for flagging, or None if appropriate
    """
    # Only filter explicit profanity with high confidence
    if contains_profanity(text):
        # Count number of profane words to determine confidence
        profanity_count = 0
        for word in ARABIC_PROFANITY:
            if word in text.lower():
                profanity_count += 1
                
        # High number of matches or very short text with profanity indicates high confidence
        if profanity_count > 1 or (profanity_count > 0 and len(text.split()) < 5):
            return True, "profanity"
    
    # Use a higher standard for hate speech - only block if multiple patterns match
    hate_speech_count = 0
    for pattern in hate_patterns:
        if re.search(pattern, text.lower()):
            hate_speech_count += 1
    
    for pattern in arabic_hate_patterns:
        if re.search(pattern, text):
            hate_speech_count += 1
    
    if hate_speech_count > 1:  # Only block if multiple hate speech patterns match
        return True, "hate_speech"
        
    # Only block explicit threats
    is_threatening = False
    explicit_threat_patterns = [
        r'\b(kill|murder|hurt|harm|attack)\b.*\b(you|yourself|him|her|them)\b',
        r'\b(bomb|explosion|terror|terrorist|attack)\b.*\b(building|school|hospital|public)\b'
    ]
    
    arabic_explicit_threat_patterns = [
        r'(اقتل|قتل|اذبح|ذبح).*?(انت|انتم|هو|هي)',
        r'(متفجر|قنبلة|ارهاب).*?(مبنى|مدرسة|مستشفى)'
    ]
    
    for pattern in explicit_threat_patterns + arabic_explicit_threat_patterns:
        if re.search(pattern, text.lower()):
            is_threatening = True
            break
    
    if is_threatening:
        return True, "threatening_content"
        
    # Don't block for PII unless it's clearly problematic (like credit card numbers)
    if re.search(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', text):  # Credit card pattern
        return True, "personal_information"
    
    # No issues detected or confidence too low
    return False, None 