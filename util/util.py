"""
util.py

A collection of utility functions to streamline operations in the dp-fall-detection project.
These utilities assist with various common tasks, such as formatting and displaying data in the terminal.

Author: Ellie Liang
Date: 2025-06-09
"""

from datetime import datetime

# --------------------------------------------------------------------
# ANSI Style Codes for Terminal Output Text Adjustments
# --------------------------------------------------------------------
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
PURPLE = '\033[35m'
BRIGHT_BLUE = '\033[94m'
BRIGHT_MAGENTA = '\033[95m'
BOLD = '\033[1m'
RESET = '\033[0m'

# --------------------------------------------------------------------
# Dictionary Containing ANSI Color Codes
# --------------------------------------------------------------------
ANSI_COLOR_DICT = {
    "RED": RED,
    "GREEN": GREEN,
    "YELLOW": YELLOW,
    "BLUE": BLUE,
    "PURPLE": PURPLE,
    'BRIGHT_BLUE': BRIGHT_BLUE,
    "BRIGHT_MAGENTA": BRIGHT_MAGENTA,
    "RESET": RESET
}

def bold_text(text):
    """
    Applies bold formatting to the given text using ANSI escape codes.

    This function wraps the input text with the ANSI escape code for bold 
    formatting, making it appear bold in terminal output.

    Args:
        text (str): The text to be bolded.

    Returns:
        str: The input text wrapped with the bold escape code and reset code.
    """
    return f"{BOLD}{text}{RESET}"

def color_text(text, color):
    """
    Wraps a given string with the given color using ANSI escape codes to display colored text in the terminal.

    Available Colors:
        - "RED"
        - "GREEN"
        - "YELLOW"
        - "BLUE"
        - "PURPLE"
        - "BRIGHT_BLUE"
        - "BRIGHT_MAGENTA"

    Args:
        text (str): The text to be colored.
        color (str): The desired text color.

    Returns:
        str: The input text wrapped with the color escape code and a reset code.
    """
    color = color.upper()
    if color not in ANSI_COLOR_DICT:
        print(f"Warning: '{color}' is not a valid color. Returning uncolored text.")
        return text
    return f'{ANSI_COLOR_DICT[color]}{text}{RESET}'

def color_text_with_code(text, color_code):
    """
    Wraps a given string with the provided ANSI escape code to display colored text in the terminal.

    Args:
        text (str): The text to be colored.
        color_code (str): The ANSI escape code representing the desired text color.

    Returns:
        str: The input text wrapped with the provided color code and a reset code to return to default styling.
    """
    return f'{color_code}{text}{RESET}'

def get_timestamp_now():
    """
    Returns the current date and time as a formatted string.

    The timestamp is formatted as 'YYYYMMDD_HHMMSS', which is commonly used 
    for generating unique or time-specific identifiers.

    Returns:
        str: The current date and time formatted as 'YYYYMMDD_HHMMSS'.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def color_timestamp_now():
    """
    Returns the current date and time formatted as a timestamp ('[YYYY-MM-DD HH:MM:SS]'), 
    wrapped in green color for terminal output.

    Returns:
        str: A string containing the formatted timestamp wrapped in color escape codes.
    """
    return color_text_with_code(f"[{get_timestamp_now()}]", GREEN)

def print_with_timestamp(text):
    """
    Prints the given text with a leading timestamp ('[YYYY-MM-DD HH:MM:SS]') wrapped in green color for terminal 
    output of the current date and time. 

    Args:
        text (str): The text to print with timestamp in terminal.
    
    Returns:
       None 
    """
    print(f"{color_timestamp_now()} {text}")

def print_color_text_with_timestamp(text, color):
    """
    Prints the given text wrapped in the given color with a leading timestamp ('[YYYY-MM-DD HH:MM:SS]') 
    wrapped in green color for terminal output of the current date and time. 

    Args:
        text (str): The text to print with timestamp in terminal.
        color (str): The desired text color.

    Returns:
       None 
    """
    print_with_timestamp(color_text(text, color))

def print_color_text(text, color):
    """
    Prints the given text wrapped in the specified color for terminal output.

    This function utilizes ANSI escape codes to display colored text in the terminal. 
    The color is applied based on the provided color string, which must be a valid color defined in the ANSI color dictionary.

    Available Colors:
        - "RED"
        - "GREEN"
        - "YELLOW"
        - "BLUE"
        - "PURPLE"
        - "BRIGHT_BLUE"
        - "BRIGHT_MAGENTA"

    Args:
        text (str): The text to be printed with the specified color in the terminal.
        color (str): The desired text color.

    Returns:
        None
    """
    print(color_text(text, color))

def print_bold_text(text):
    """
    Prints the given text with bold formatting using ANSI escape codes.

    This function wraps the input text with the ANSI escape code for bold 
    formatting and prints it to the terminal.

    Args:
        text (str): The text to be printed in bold.

    Returns:
        None
    """
    print(bold_text(text))