# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class LogoScrapeItem(scrapy.Item):
    site_id = scrapy.Field()
    link_id = scrapy.Field() 
    status = scrapy.Field()
    current_link = scrapy.Field()
    link_img = scrapy.Field()
    pattern_id = scrapy.Field()
    found_img = scrapy.Field()

