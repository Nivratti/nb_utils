import os
import json
from loguru import logger
from datetime import datetime as dt

def add_today_date_in_filename(file_name, format="%Y-%m-%d"):
    """
    Add today's date to the filename.

    Args:
        file_name (str): The original file name.
        format (str): Optional. The format in which the date should be added to the filename. 
                      Default is "%Y-%m-%d".

    Returns:
        str: The modified file name with the date appended. in case any code exception it return original filename as it is

    Example:
        >>> add_today_date_in_filename("file_util_logger.txt")
        'file_util_logger_2023-06-12.txt'
    """
    try:
        today_date = dt.now().strftime(format)
        
        filename_no_ext, file_extension = os.path.splitext(file_name)
        filename_no_ext = f"{filename_no_ext}-{today_date}"
        
        name_with_date = filename_no_ext + file_extension
        return name_with_date
    
    except Exception as e:
        logger.error("Error:", e)
        return file_name

def generate_today_date_log_folder(root_log_folder_withpath, exist_ok=True):
    """
    Create a log folder with today's date.

    Args:
        root_log_folder_withpath (str): The root log folder path.
        exist_ok (bool): Optional. Default true means no error if folder already exists

    Returns:
        str: The path of the created log folder.

    Example:
        >>> generate_today_date_log_folder("logs")
        'logs/June-2023/2023-06-12'
    """  
    now = dt.now()
    current_month_year = now.strftime("%B-%Y")

    today_date = now.strftime("%Y-%m-%d")
    
    # list flating * before list required -- list must not be empty
    today_log_folder = os.path.join(
        root_log_folder_withpath,
        *[current_month_year, today_date]
    )
    # make recursive dir
    os.makedirs(today_log_folder, exist_ok=exist_ok)
    return today_log_folder

def check_create_file_with_all_permission(file_name_with_path):
    """
    Check if the file exists. If not, create the file with read and write permissions.

    Args:
        file_name_with_path (str): The file name along with its path.

    Returns:
        bool: True if the file is created successfully, False if the file already exists.

    Example:
        >>> check_create_file_with_all_permission("file.txt")
        True
    """
    if not os.path.exists(file_name_with_path):
        # The default umask is 0o22 which turns off write permission of group and others
        os.umask(0)
        with open(os.open(file_name_with_path, os.O_CREAT | os.O_APPEND, 0o777), 'w') as fh:
            pass
            # print(f"file {file_name_with_path} created with all permissions")
        return True
    else:
        return False

def generate_unique_filename(extension=".txt", base_folder=None):
    """
    Generate a unique filename.

    Args:
        extension (str, optional): Extension for the filename. Defaults to ".txt".
        base_folder (str, optional): Base folder path. Defaults to None.

    Returns:
        str: The generated unique filename.

    Example:
        >>> generate_unique_filename(".txt", "data/")
        'data/abcdefgh.txt'
    """
    from text_processing import generate_unique_str
    unique_str = generate_unique_str()

    if not extension.startswith('.'):
        extension = "." + extension

    filename = unique_str + extension

    if base_folder:
        return os.path.join(base_folder, filename)
    else:
        return filename