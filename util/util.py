"""
util.py

A collection of utility functions to streamline operations in the dp-fall-detection project.
These utilities assist with various common tasks, such as formatting and displaying data in the terminal.

Author: Ellie Liang
Date: 2025-06-09
"""

from datetime import datetime

# --------------------------------------------------------------------
# ANSI Color Codes for Terminal Output Color Adjustments
# --------------------------------------------------------------------
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
PURPLE = '\033[35m'
BRIGHT_BLUE = '\033[94m'
BRIGHT_MAGENTA = '\033[95m'
RESET = '\033[0m'

# --------------------------------------------------------------------
# Dictionary Containing ANSI Color Codes
# --------------------------------------------------------------------
ANSI_COLOR_DICT = {
    "RED": RED,
    "GREEN": GREEN,
    "BLUE": BLUE,
    "PURPLE": PURPLE,
    'BRIGHT_BLUE': BRIGHT_BLUE,
    "BRIGHT_MAGENTA": BRIGHT_MAGENTA,
    "RESET": RESET
}

def color_text(text, color):
    """
    Wraps a given string with the given color using ANSI escape codes to display colored text in the terminal.

    Available Colors:
        - "RED"
        - "GREEN"
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