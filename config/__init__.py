"""Configuration module"""
from .settings import settings, get_settings
from .aws import bedrock, opensearch, s3

__all__ = ['settings', 'get_settings', 'bedrock', 'opensearch', 's3']

