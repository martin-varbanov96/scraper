from logo_scrape.items import LogoScrapeItem
import datetime
import scrapy
import csv
import re
from collections import Counter
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

class LogoSpider(scrapy.Spider):
    name = "logo_spider"

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
        response_status_code = int(re.match(pattern_regex,
                                   str(response)).group(0)[1:4])
        if response_status_code != 200:
            yield LogoScrapeItem(reponse_code=response_status_code)
        # TODO:: We add false data

        # here we set possible patterns and check if they are found:
        patterns_trees = [response.css("#Header").xpath("//a/img/@src"), 
                          response.css("#HEADER").xpath("//a/img/@src"),    
                          response.css("#header").xpath("//a/img/@src"),
                          response.xpath("//header//a/img/@src"),
                          response.xpath("//a[@href='"+response.url+'/'+"']/img/@src"),
                          response.xpath("//a[@href='"+response.url+'/'+"']/img/@src"),
                          response.css("header").xpath("//a[@href='/']/img/@src"),
                          response.xpath("//a[@href='/']/img/@src")
                          ]
        pattern_number = 0
        for pattern_tree in patterns_trees:
            def parse_within_page(self, response):
                return pattern_tree

            def getFinalImage(input_dict):
                greatest_count = 0
                greatest_string = ""
                for el in input_dict:
                    if greatest_count < input_dict[el]:
                        greatest_count = input_dict[el]
                        greatest_string = el
                return greatest_string, greatest_count
            # Here we iterate over 10 random links
            iterated_links_counter = 0
            links_list = list()
            flag_to_yield = False
            most_common_image_found = ""
            amount_most_common_image_found = 0
            for href in response.xpath("//a"):
                    iterated_links_counter += 1
                    if iterated_links_counter >= 10:
                        break
                    pattern_response = scrapy.Request(href.xpath("@href").extract_first(),
                                                      self.parse_within_page())
                    if pattern_response:
                        flag_to_yield = True
                        links_list.append(pattern_response)
            links_dict = Counter(links_list)
            most_common_image_found, amount_most_common_image_found = getFinalImage(links_dict)

            if flag_to_yield:
                url = response
                pattern_type = pattern_number
                pattern_number += 1
                file_url = pattern_tree.extract_first()
                yield LogoScrapeItem(url=url,
                                     file_url=file_url,
                                     pattern_type=pattern_type,
                                     response_code=response_status_code,
                                     found_img=True,
                                     is_http="",
                                     is_https="",
                                     succsses_rate=amount_of_same_image)

