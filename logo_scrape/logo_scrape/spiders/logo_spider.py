from logo_scrape.items import LogoScrapeItem
import datetime
import scrapy


class LogoSpider(scrapy.Spider):
    name = logo_spider
    start_urls = ["https://advokatarnaudov.com/категории-правни-услуги/търговско-и-банково-право/"]

    def parse(self, response):
        
        def harcode_img():

            headers = ["#Header", "#header", "#HEADER"]
            for header in headers:
                pattern_header=response.css(header)
                if pattern_header:
                    if pattern_header.xpath("//a/img/@src"):
                        url = response
                        file_url = pattern_header.xpath("//a/img/@src").extract_first()
                        yield LogoScrapeItem((title=url, file_urls=file_url)

        harcode_img()
