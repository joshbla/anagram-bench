#!/usr/bin/env python3
"""
Get OpenRouter API rate limits
"""

import os
import json
import requests
from dotenv import load_dotenv

def get_openrouter_rate_limits():
    """
    Get rate limit information from OpenRouter API
    
    Returns:
        tuple: (requests_per_interval, interval) or (None, None) if error
    """
    # Load environment variables from .env.local
    load_dotenv('.env.local')
    
    # Get API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        return None, None
    
    # Make request to OpenRouter API
    response = requests.get(
        url="https://openrouter.ai/api/v1/key",
        headers={
            "Authorization": f"Bearer {api_key}"
        }
    )
    
    # Check if request was successful
    if response.status_code == 200:
        data = response.json()
        
        # Extract rate limit information
        if 'data' in data and 'rate_limit' in data['data']:
            rate_limit = data['data']['rate_limit']
            return rate_limit.get('requests'), rate_limit.get('interval')
    
    return None, None


if __name__ == "__main__":
    # For direct execution, just return the values
    requests, interval = get_openrouter_rate_limits()
    if requests and interval:
        # Just exit with success - no output
        exit(0)
    else:
        # Exit with error code if failed
        exit(1)