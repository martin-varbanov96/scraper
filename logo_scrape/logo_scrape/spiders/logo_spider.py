from logo_scrape.items import LogoScrapeItem

import datetime
import scrapy


class LogoSpider(scrapy.Spider):
    name = "logo_spider"
    #start_urls = ["http://time.com"]

    def start_requests(self):
        yield scrapy.Request("http://time.com", self.parse)
        yield scrapy.Request("https://git-scm.com/docs/git-merge", self.parse)
        yield scrapy.Request("https://www.pythoncentral.io/one-line-if-statement-in-python-ternary-conditional-operator/", self.parse)

    def parse(self, response):
        # TODO:: Here these methods make may copy burger button or similar

        patterns_trees = [response.css("#Header").xpath("//a/img/@src"), 
                            response.css("#HEADER").xpath("//a/img/@src"),    
                            response.css("#header").xpath("//a/img/@src"),
                            response.xpath("//header//a/img/@src"),


        ]
        for pattern_tree in patterns_trees:
            if pattern_tree:
            #<header><a><img>
                url = response
                file_url = pattern_tree.extract_first()
                yield LogoScrapeItem(title=url, file_urls=file_url)
        
