import logging
from transformers import pipeline, AutoModel, PhobertTokenizer

def main():
    from colorama import Fore, Back, Style

    def print_colored_text():
        print(f"{Fore.RED}This is red text{Style.RESET_ALL}")
        print(f"{Back.GREEN}This has a green background{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}This is bold text{Style.RESET_ALL}")

    print_colored_text()
    
    

if __name__ == '__main__':
    main()