from logo_scrape.items import LogoScrapeItem
import datetime
import scrapy
import csv
import re
import logging

def get_data():
        path = 'logo_scrape/static/test_data.csv'
        output_list = list()
        with open(path) as f:
            data_read = csv.reader(f, delimiter='\n')
            for row in data_read:
                el_http = "http://" + str(row[0])
                output_list.append(el_http)
                # DO I NEED THIS:
                # el_https = "https://" + str(row[0])
                # output_list.append(el_https)
        return output_list

class PatternUrl:

    def __init__(self, path_to_img="", list_of_conditionals=[]):
        self.url_pattern = ""
        self.file_url = ""
        self.path_to_img = path_to_img
        self.list_of_conditionals = list_of_conditionals

    def find_obj(self, response):
        for el in self.list_of_conditionals:
            if el:
                if self.path_to_img:
                    url = response
                    file_url = str(self.path_to_img)
                    yield LogoScrapeItem(url=url, file_url=file_url)

class LogoSpider(scrapy.Spider):
    name = "logo_spider"
    #start_urls = ["http://time.com"]

    def start_requests(self):
       # for el in get_data():
        #    yield scrapy.Request(el, self.parse)
         yield scrapy.Request("https://time.com", self.parse)
         #yield scrapy.Request("http://git-scm.com/docs/git-merge", self.parse)
     #   try:
      #      yield scrapy.Request("https://www.wyndhamhotels.com/en-uk", self.parse)
       # except TimeoutError as e:
        #    logging.exception("MSG"*500)

            
    def parse(self, response):
        pattern_regex = r"<[0-9][0-9][0-9]\W"
        response_status_code=int(re.match(pattern_regex, str(response)).group(0)[1:4])
        if response_status_code != 200:
            yield LogoScrapeItem(reponse_code=response_status_code)

        #Here we iterate over 10 random links
        iterated_links_counter=0
        for href in response.xpath("//a"):
            iterated_links_counter +=1
            if iterated_links_counter >= 10:
                break
            #TODO:: Implement second parse
            #yield scrapy.Requst(href.xpath("@href").extract_first(), self.PARSE_FUNCTION_IMPLEMENT)
            pass


        # TODO:: We add false data

        #here we set possible patterns and check if they are found:
        patterns_trees = [response.css("#Header").xpath("//a/img/@src"), 
                            response.css("#HEADER").xpath("//a/img/@src"),    
                            response.css("#header").xpath("//a/img/@src"),
                            response.xpath("//header//a/img/@src"),
                            response.xpath("//a[@href='"+response.url+'/'+"']/img/@src"),
                            response.xpath("//a[@href='"+response.url+'/'+"']/img/@src"),
                            response.css("header").xpath("//a[@href='/']/img/@src"),
                            response.xpath("//a[@href='/']/img/@src")
        ]
        # TODO:: check status codes
        #if response.getcode() != 200:
         #       url = response
          #      response_code=response.getcode()
           #     yield LogoScrapeItem(url=url, response_code=response_code)
        pattern_number = 0 
        for pattern_tree in patterns_trees:
            if pattern_tree:
                url = response

                # TODO:: RETURN STATUS CODE
                #response_code=response.getcode()
                
                pattern_type = pattern_number
                pattern_number+=1
                file_url = pattern_tree.extract_first()
                # TODO:: Add response code
                yield LogoScrapeItem(url=url, file_url=file_url,pattern_type=pattern_type, response_code=response_status_code, found_img=True, is_http="", is_https="")
        home_page_url_patter = "" 
#        a = PatternUrl(response.css("header").xpath("//a[@href='"+response.url+'/'+"']/img/@src").extract_first(), [response.css("header").xpath("//a[@href='"+response.url+'/'+"']")] )
 #       a.find_obj(response)
        # response.css("header").xpath("//a[contains(@href, 'http://time.com/')]").re(r'"http://time.com/"')


