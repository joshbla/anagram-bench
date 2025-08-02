#!/usr/bin/env python3
"""
Example of using the rate limit checker
"""

from check_openrouter_limits import get_openrouter_rate_limits

# Get the rate limits
requests, interval = get_openrouter_rate_limits()

print(f"Result: requests={requests}, interval={interval}")