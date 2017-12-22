from logo_scrape.items import LogoScrapeItem

import datetime
import scrapy


class LogoSpider(scrapy.Spider):
    name = "logo_spider"
    # start_urls = ["http://time.com"]

    def start_requests(self):
        yield scrapy.Request("http://time.com", self.parse)

    def parse(self, response):
        #Here we try to find the logo in pattern:
        #<div id="header"><a><img>
        headers = ["#Header", "#header", "#HEADER"]
        for header in headers:
            pattern_header=response.css(header)
            if pattern_header:
                if pattern_header.xpath("//a/img/@src"):
                    url = response
                    file_url = pattern_header.xpath("//a/img/@src").extract_first()
                    yield LogoScrapeItem(title=url, file_urls=file_url)

        # Here we try to find logo if site uses <header>

        if response.xpath("//header"):
            #<header><a><img>
            header_a_img = response.xpath("//header//a/img/@src")
            if header_a_img:
                url = response
                file_url = header_a_img.extract_first()

        # harcode_img()
