from logo_scrape.items import LogoScrapeItem

import datetime
import scrapy

class PatternUrl:

    def __init__(self, path_to_img="", list_of_conditionals=[]):
        self.url_pattern = ""
        self.file_url = ""
        self.path_to_img = path_to_img
        self.list_of_conditionals = list_of_conditionals

    def find_obj(self, response):
        print("asdds"*50)
        for el in self.list_of_conditionals:
            if el:
                if self.path_to_img:
                    print("500"*500)
                    url = response
                    file_url = str(self.path_to_img)
                    print(url)
                    print(file_url)
                    yield LogoScrapeItem(url=url, file_url=file_url)

class LogoSpider(scrapy.Spider):
    name = "logo_spider"
    #start_urls = ["http://time.com"]

    def start_requests(self):
        yield scrapy.Request("http://time.com", self.parse)
        yield scrapy.Request("https://git-scm.com/docs/git-merge", self.parse)

    def parse(self, response):
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
                yield LogoScrapeItem(url=url, file_url=file_url)
        home_page_url_patter = "" 
#        a = PatternUrl(response.css("header").xpath("//a[@href='"+response.url+'/'+"']/img/@src").extract_first(), [response.css("header").xpath("//a[@href='"+response.url+'/'+"']")] )
 #       a.find_obj(response)
        # response.css("header").xpath("//a[contains(@href, 'http://time.com/')]").re(r'"http://time.com/"')


