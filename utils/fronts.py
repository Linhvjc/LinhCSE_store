from colorama import Fore, Back, Style
import shutil
columns, _ = shutil.get_terminal_size()
class Font():
    def __init__(self) -> None:
        pass
    
    def inline_text(self, text):
        return text* columns
    
    def bool_text(self, text):
        return f"\033[1m{text}\033[0m"
    
    def underline_text(self, text):
        return f"\033[4m{text}\033[0m"
    
    def info_text(self, text):
        return f"{Fore.BLUE}{text}{Style.RESET_ALL}"
    
    def warning_text(self, text):
        return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"
    
    def error_text(self, text):
        return f"{Fore.RED}{text}{Style.RESET_ALL}"