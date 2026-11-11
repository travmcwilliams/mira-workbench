# platform_scrapers package - one module per major aggregator that consolidates many sources
from .platform_c_scraper import scrape_platform_c

__all__ = ["scrape_platform_c"]
