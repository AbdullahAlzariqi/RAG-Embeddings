from collections import Counter

def get_duplicate_strings(string_list):
    """
    Finds and returns duplicate strings in a list of strings.

    Args:
        string_list: A list of strings.

    Returns:
        A list containing the duplicate strings.  
        Returns an empty list if there are no duplicates.
    """
    counts = Counter(string_list)
    duplicates = [string for string, count in counts.items() if count > 1]
    return duplicates