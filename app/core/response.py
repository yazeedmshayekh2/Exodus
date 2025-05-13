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
        formatted = format_arabic_response(response)
    else:
        formatted = format_english_response(response)
    
    # Apply post-processing to fix common issues
    return post_process_formatting(formatted, language)

def post_process_formatting(response: str, language: str) -> str:
    """
    Apply final formatting fixes to ensure clean, visually appealing responses.
    
    Args:
        response (str): Formatted response text
        language (str): Language code ('en' or 'ar')
        
    Returns:
        str: Clean formatted response
    """
    # Fix title format - remove extra stars and ensure proper bold formatting
    lines = response.split('\n')
    processed_lines = []
    
    for i, line in enumerate(lines):
        # Fix titles with stars (e.g., "** Title **" -> "**Title**")
        if '**' in line:
            # Remove any stray asterisks outside of proper bold markers
            line = re.sub(r'(?<!\*)\*(?!\*)', '', line)
            
            # Fix ** text ** -> **text** (remove spaces inside bold markers)
            line = re.sub(r'\*\*\s+([^*]+?)\s+\*\*', r'**\1**', line)
            
            # More aggressive fixing for spaces after opening **
            line = re.sub(r'\*\*\s+', r'**', line)
            
            # More aggressive fixing for spaces before closing **
            line = re.sub(r'\s+\*\*', r'**', line)
            
            # Fix ***text*** -> **text** (remove extra *)
            line = re.sub(r'\*\*\*+([^*]+?)\*\*\*+', r'**\1**', line)
            
            # If it's a title (standalone bold text line)
            if line.strip().startswith('**') and line.strip().endswith('**') and i < 5:
                # Add extra spacing for titles at the beginning of the message
                if i == 0 or (i > 0 and not lines[i-1].strip()):
                    processed_lines.append(line)
                    if i < len(lines) - 1 and lines[i+1].strip():
                        processed_lines.append('')  # Add empty line after title
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)
    
    response = '\n'.join(processed_lines)
    
    # Fix bullet points to ensure proper markdown formatting
    response = re.sub(r'^\s*[-•]\s*', '* ', response, flags=re.MULTILINE)
    
    # Fix numbered lists to ensure proper spacing
    response = re.sub(r'(\d+)\.(\S)', r'\1. \2', response)
    
    # Ensure phone numbers are properly formatted
    if language == 'en':
        # Fix common phone number formats
        phone_patterns = [
            (r'\b(\d{4})\s*(\d{4})\b', r'**\1 \2**'),  # 1234 5678 -> **1234 5678**
            (r'\b(\+\d{3})\s*(\d{4})\s*(\d{4})\b', r'**\1 \2 \3**'),  # +123 1234 5678 -> **+123 1234 5678**
        ]
        
        for pattern, replacement in phone_patterns:
            response = re.sub(pattern, replacement, response)
    
    # One final pass to fix any remaining bold text issues
    response = re.sub(r'\*\*\s+', r'**', response)  # Remove spaces after **
    response = re.sub(r'\s+\*\*', r'**', response)  # Remove spaces before **
    response = re.sub(r'\*{3,}', r'**', response)   # Fix excessive asterisks
    
    # Clean up extra whitespace
    response = re.sub(r' {2,}', ' ', response)
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Ensure proper spacing around messages
    response = response.strip()
    
    return response

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
    
    # Fix improper bold text formatting
    # Replace ** text ** (with spaces inside) with **text** (no spaces inside)
    response = re.sub(r'\*\*\s+([^*]+?)\s+\*\*', r'**\1**', response)
    
    # More aggressive fixing for spaces after opening ** and before closing **
    response = re.sub(r'\*\*\s+', r'**', response)
    response = re.sub(r'\s+\*\*', r'**', response)
    
    # Remove extra stars around text that's already bold
    response = re.sub(r'\*\s*\*\*([^*]+?)\*\*\s*\*', r'**\1**', response)
    
    # Fix titles with stars - replace "** Title **" with "**Title**"
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
        
        # Clean up stars from titles (e.g. ** Insurance Issue ** to **Insurance Issue**)
        if line.startswith('**') and line.endswith('**'):
            # Remove extra spaces inside bold markers
            title = re.sub(r'\*\*\s*(.*?)\s*\*\*', r'**\1**', line)
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            formatted_lines.append(title)
            formatted_lines.append("")
            continue
        
        # Handle numbered lists - make sure there's a space after the period
        if re.match(r'^\d+\.', line):
            number, content = line.split('.', 1)
            formatted_lines.append(f"{number.strip()}. {content.strip()}")
            in_list = True
            continue
        
        # Handle bullet points with proper markdown format
        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
            content = line.lstrip('•- *').strip()
            # Fix bullet points with bold text
            if content.startswith('**') and content.endswith('**'):
                content = content[2:-2].strip()  # Remove the bold markers
                formatted_lines.append(f"* **{content}**")  # Add bold inside the bullet
            else:
                formatted_lines.append(f"* {content}")
            in_list = True
            continue
        
        # Handle continuation of list items
        if in_list and not line.startswith(('*', '-', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
            if line.strip():
                formatted_lines.append(f"  {line}")
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
    
    # Fix bold text - remove spaces between asterisks and text
    response = re.sub(r'\*\*\s+', '**', response)  # Remove space after **
    response = re.sub(r'\s+\*\*', '**', response)  # Remove space before **
    
    # Fix for consecutive bold texts
    response = re.sub(r'\*\*\s*\*\*', '', response)  # Remove empty bold markers
    
    # Clean up multiple spaces and newlines
    response = re.sub(r' {2,}', ' ', response)
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Clean up markdown - remove unnecessary stars
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
    
    # Fix improper bold text formatting
    response = re.sub(r'\*\*\s+([^*]+?)\s+\*\*', r'**\1**', response)
    
    # More aggressive fixing for spaces after opening ** and before closing **
    response = re.sub(r'\*\*\s+', r'**', response)
    response = re.sub(r'\s+\*\*', r'**', response)
    
    # Remove extra stars around text that's already bold
    response = re.sub(r'\*\s*\*\*([^*]+?)\*\*\s*\*', r'**\1**', response)
    
    # Clean up pre-process section titles
    response = re.sub(r'\*\*\s*(.*?)\s*\*\*', r'**\1**', response)
    
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
        
        # Handle section titles - clean up stars
        if line.startswith('**') and line.endswith('**'):
            current_section = line
            formatted_lines.extend(["", line, ""])
            continue
        
        # Handle headings (Arabic style)
        if line.endswith(':') and len(line.split()) <= 4:
            formatted_lines.extend(["", f"**{line}**", ""])
            continue
        
        # Handle numbered lists with proper spacing
        if any(line.startswith(f'{i}.') for i in range(10)):
            number, content = line.split('.', 1)
            formatted_lines.append(f"{number.strip()}. {content.strip()}")
            in_list = True
            continue
        
        # Handle bullet points with clean formatting
        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
            content = line.lstrip('•- *').strip()
            # If content is bold, format it properly
            if content.startswith('**') and content.endswith('**'):
                content = content[2:-2].strip()  # Remove the bold markers
                formatted_lines.append(f"* **{content}**")  # Add bold inside the bullet
            else:
                formatted_lines.append(f"* {content}")
            in_list = True
            continue
        
        # Handle continuation of list items
        if in_list and not line.startswith(('*', '-', '•')):
            if line.strip():  # Only append non-empty continuation lines
                formatted_lines.append(f"  {line}")
            else:
                in_list = False  # End list on empty line
            continue
        
        # Handle bold text - fix spacing
        if '**' in line:
            line = re.sub(r'\*\*\s+', '**', line)  # Remove space after **
            line = re.sub(r'\s+\*\*', '**', line)  # Remove space before **
        
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
    
    # Clean up markdown - remove unnecessary stars
    response = re.sub(r'\*{3,}', '**', response)
    response = re.sub(r'\*\*\s+\*\*', '', response)
    
    # Add RTL mark and additional formatting for Arabic
    response = '\u200F' + response  # RTL mark
    
    return response.strip()

def get_fallback_response(language: str) -> str:
    """Enhanced fallback response with customer service referral"""
    if language == 'ar':
        return """
**أود مساعدتك في ذلك**

أعتذر، لا يبدو أن لدي معلومات كافية عن هذا الموضوع في قاعدة بياناتي الحالية.

يمكنك:
* تجربة طرح سؤالك بطريقة أخرى
* الاستفسار عن خدمات أخرى متوفرة لدينا
* استخدام عبارات أكثر وضوحاً متعلقة بخدماتنا

سأكون سعيداً بالإجابة عن أسئلتك المتعلقة بخدماتنا وعملياتنا الأساسية.
"""
    return """
**I'd like to help with that**

I'm sorry, I don't seem to have enough information about this topic in my current database.

You can:
* Try phrasing your question differently
* Ask about other services we offer
* Use more specific terms related to our services

I'm happy to answer any questions about our core services and processes.
""" 