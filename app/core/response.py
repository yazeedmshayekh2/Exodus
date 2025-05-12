"""
Response formatting utilities for the chatbot
"""

import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

def format_response(response: str, language: str) -> str:
    """
    Format the response based on language.
    
    Args:
        response (str): Raw response text
        language (str): Language code ('en' or 'ar')
        
    Returns:
        str: Formatted response with proper markdown and styling
    """
    if language == 'ar':
        return format_arabic_response(response)
    return format_english_response(response)

def format_english_response(response: str) -> str:
    """
    Format English response with markdown styling.
    
    Args:
        response (str): Raw English response
        
    Returns:
        str: Formatted response with proper markdown and styling
    """
    # Format phone numbers first
    response = format_phone_numbers_en(response)
    
    # Pre-process section titles and ensure proper spacing around bold text
    response = re.sub(r'\*\*\s*(.*?)\s*\*\*', r'**\1**', response)
    
    # Convert bullet points to markdown
    lines = response.split('\n')
    formatted_lines = []
    
    in_list = False
    list_indent = "  "  # Standard indentation for list items
    
    for line in lines:
        line = line.strip()
        
        # Skip multiple empty lines
        if not line:
            if not formatted_lines or formatted_lines[-1] != "":
                formatted_lines.append("")
            continue
        
        # Handle section titles (enclosed in **)
        if re.match(r'^\*\*.*\*\*$', line):
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            formatted_lines.append(line)
            formatted_lines.append("")
            continue
        
        # Handle numbered lists
        if re.match(r'^\d+\.', line):
            number, content = line.split('.', 1)
            formatted_lines.append(f"{number.strip()}. {content.strip()}")
            in_list = True
            continue
        
        # Handle bullet points
        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
            content = line.lstrip('•- *').strip()
            formatted_lines.append(f"{list_indent}* {content}")
            in_list = True
            continue
        
        # Handle continuation of list items
        if in_list and not line.startswith(('*', '-', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
            if line.strip():
                formatted_lines.append(f"{list_indent}  {line}")
            else:
                in_list = False
            continue
        
        # Regular text (not part of a list)
        in_list = False
        formatted_lines.append(line)
    
    response = '\n'.join(formatted_lines)
    
    # Clean up formatting
    response = re.sub(r'\s+:', ':', response)
    response = re.sub(r'\s+,', ',', response)
    response = re.sub(r'\s+\.', '.', response)
    response = re.sub(r'\s+\)', ')', response)
    response = re.sub(r'\(\s+', '(', response)
    
    # Add proper spacing after punctuation
    response = re.sub(r'([.,!?:])(?!\s|$)', r'\1 ', response)
    
    # Ensure proper spacing around bold text
    response = re.sub(r'(?<!\*)\*\*(?!\s)', '** ', response)  # Add space after **
    response = re.sub(r'(?<!\s)\*\*(?!\*)', ' **', response)  # Add space before **
    
    # Clean up multiple spaces and newlines
    response = re.sub(r' {2,}', ' ', response)
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Clean up markdown
    response = re.sub(r'\*{3,}', '**', response)  # Fix multiple asterisks
    response = re.sub(r'\*\*\s+\*\*', '', response)  # Remove empty bold tags
    
    # Remove any remaining email-like formatting
    response = re.sub(r'\n\s*Best regards,?\s*\n', '\n', response, flags=re.IGNORECASE)
    response = re.sub(r'\n\s*\[.*?\]\s*\n', '\n', response)
    response = re.sub(r'\n\s*Customer Service.*?\n', '\n', response, flags=re.IGNORECASE)
    
    return response.strip()

def format_phone_numbers_en(text: str) -> str:
    """Enhanced phone number formatting"""
    # Format different phone number patterns
    patterns = [
        (r'\b\d{8}\b', lambda m: f"{m.group(0)[:4]} {m.group(0)[4:]}"),  # 12345678 -> 1234 5678
        (r'\b\d{4}\s?\d{4}\b', lambda m: m.group(0).replace(" ", "") [:4] + " " + m.group(0).replace(" ", "") [4:]),  # Standardize 8-digit format
        (r'\+\d{3}\s?\d{8}\b', lambda m: f"+{m.group(0).replace(' ', '')[1:4]} {m.group(0).replace(' ', '')[4:8]} {m.group(0).replace(' ', '')[8:]}"),  # International format
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    
    return result

def format_arabic_response(response: str) -> str:
    """
    Format Arabic response with markdown styling.
    
    Args:
        response (str): Raw Arabic response
        
    Returns:
        str: Formatted response with proper markdown and RTL styling
    """
    # Convert Western numbers to Arabic
    western_to_arabic = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
    response = response.translate(western_to_arabic)
    
    # Pre-process section titles
    response = re.sub(r'\*\*\s*(.*?)\*\*\s*', r'\n**\1**\n\n', response)
    
    lines = response.split('\n')
    formatted_lines = []
    
    in_list = False
    list_indent = "  "  # Standard indentation for list items
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_section:
                formatted_lines.append("")
            continue
        
        # Handle section titles (enclosed in **)
        if re.match(r'^\*\*.*\*\*$', line):
            current_section = line
            formatted_lines.extend(["", line, ""])
            continue
        
        # Handle headings (Arabic style)
        if line.endswith(':') and len(line.split()) <= 4:
            formatted_lines.extend(["", f"**{line}**", ""])
            continue
        
        # Handle numbered lists (RTL format)
        if any(line.startswith(f'{i}.') for i in range(10)):
            number, content = line.split('.', 1)
            formatted_lines.append(f"{number.strip()}. {content.strip()}")
            in_list = True
            continue
        
        # Handle bullet points
        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
            content = line.lstrip('•- *').strip()
            # If content starts with bold text, handle it specially
            if content.startswith('**'):
                content = content.replace('**', '', 2)  # Remove first pair of **
            formatted_lines.append(f"{list_indent}* {content}")
            in_list = True
            continue
        
        # Handle continuation of list items
        if in_list and not line.startswith(('*', '-', '•')):
            if line.strip():  # Only append non-empty continuation lines
                formatted_lines.append(f"{list_indent}  {line}")
            else:
                in_list = False  # End list on empty line
            continue
        
        # Handle bold text
        if '**' in line:
            line = re.sub(r'(?<!\*)\*\*(?!\*)', ' **', line)  # Add space before **
            line = re.sub(r'\*\*(?!\s)', '** ', line)  # Add space after **
            line = re.sub(r'\s+\*\*\s+', ' **', line)  # Clean up extra spaces
        
        # Regular text
        in_list = False
        formatted_lines.append(line)
    
    response = '\n'.join(formatted_lines)
    
    # Clean up formatting
    response = re.sub(r'\s+:', ':', response)
    response = re.sub(r'\s+،', '،', response)  # Arabic comma
    response = re.sub(r'\s+\.', '.', response)
    response = re.sub(r'\s+\)', ')', response)
    response = re.sub(r'\(\s+', '(', response)
    
    # Add proper spacing after Arabic punctuation
    response = re.sub(r'([.،؛؟!:])(?!\s|$)', r'\1 ', response)
    
    # Clean up multiple spaces and newlines
    response = re.sub(r'\s{2,}', ' ', response)
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)  # Fix multiple newlines
    
    # Clean up markdown
    response = re.sub(r'\*{4,}', '**', response)
    response = re.sub(r'\*\*\s+\*\*', '', response)
    
    # Add RTL mark and additional formatting for Arabic
    response = '\u200F' + response  # RTL mark
    
    return response.strip()

def get_fallback_response(language: str) -> str:
    """Enhanced fallback response with customer service referral"""
    if language == 'ar':
        return """
                عذراً، أقترح التواصل مع فريق خدمة العملاء للحصول على المساعدة المتخصصة.
                """
    return """
            I apologize, I recommend contacting our customer service team for specialized assistance.
            """ 