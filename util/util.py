
from datetime import datetime

# --------------------------------------------------------------------
# ANSI Color Codes for Terminal Output Color Adjustments
# --------------------------------------------------------------------
GREEN = '\033[32m'
BLUE = '\033[34m'
RED = '\033[31m'
PURPLE = '\033[35m'
BRIGHT_MAGENTA = '\033[95m'
BRIGHT_BLUE = '\033[94m'
RESET = '\033[0m'

def color_text(text, color_code):
    """
    Wraps a given string with ANSI escape codes to display colored text in the terminal.

    Args:
        text (str): The text to be colored.
        color_code (str): The ANSI escape code representing the desired text color.

    Returns:
        str: The input text wrapped with the provided color code and a reset code to return to default styling.
    """
    return f'{color_code}{text}{RESET}'

def color_timestamp_now():
    """
    Returns the current date and time formatted as a timestamp ('[YYYY-MM-DD HH:MM:SS]'), 
    wrapped in green color for terminal output.

    Returns:
        str: A string containing the formatted timestamp wrapped in color escape codes.
    """
    return color_text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", GREEN)

def print_with_timestamp(text):
    """
    Prints the given text with a leading timestamp ('[YYYY-MM-DD HH:MM:SS]') wrapped in green color for terminal 
    output of the current date and time. 

    Args:
        text (str): The text to print with timestamp in terminal
    
    Returns:
       None 
    """
    print(f"{color_timestamp_now()} {text}")