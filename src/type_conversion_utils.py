"""
Type Conversion Utilities
Provides safe type conversion functions to prevent type errors in mathematical operations
"""

def safe_int_conversion(value, default=0):
    """
    Safely convert a value to integer
    
    Args:
        value: Value to convert (can be str, int, float, etc.)
        default: Default value to return if conversion fails
        
    Returns:
        int: Converted integer value or default
    """
    try:
        if isinstance(value, str):
            # Handle string representations of numbers
            return int(float(value))  # Convert through float to handle "1.0" strings
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            return default
    except (ValueError, TypeError):
        return default

def safe_float_conversion(value, default=0.0):
    """
    Safely convert a value to float
    
    Args:
        value: Value to convert (can be str, int, float, etc.)
        default: Default value to return if conversion fails
        
    Returns:
        float: Converted float value or default
    """
    try:
        if isinstance(value, str):
            return float(value)
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return default
    except (ValueError, TypeError):
        return default

def ensure_numeric_for_math(value, target_type=int, default=None):
    """
    Ensure a value is numeric before mathematical operations
    
    Args:
        value: Value to check and convert
        target_type: Target type (int or float)
        default: Default value if conversion fails (uses 0 for int, 0.0 for float if None)
        
    Returns:
        Numeric value ready for mathematical operations
    """
    if default is None:
        default = 0 if target_type == int else 0.0
    
    if target_type == int:
        return safe_int_conversion(value, default)
    else:
        return safe_float_conversion(value, default)